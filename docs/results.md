# Results

Measurements, with the commands that produce them. Every projection elsewhere in this
repository is calibrated against these numbers rather than estimated.

This file is chronological, which means it records wrong turns as well as results. That is
deliberate: several of the numbers here replaced earlier numbers of mine that were wrong,
and a log that quietly deleted them would suggest a straighter path than there was.

## The headline numbers

| | |
|---|---|
| Whole model on chain | Qwen3-30B-A3B — 60,465,686,912 bytes, 7,318 transactions, 17.1e12 gas |
| Gas a token | 3.79e9 for the 30B mixture, 277M for dense Qwen3-0.6B |
| Agreement with float64 | exact tokens, every position, both models |
| Reproduced | Intel Xeon and AMD EPYC: same state root, same gas, same tokens |
| Determinism | one digest across aarch64/macOS, x86 Intel, x86 AMD, 12 thread counts |
| Routing | same 8 experts of 128, all three machines, every position |
| Native | 0.27 s a token (30B), 21.6 tokens/sec (0.6B) |
| Kernel | 196 GB/s, against the EVM's 63 gas per multiply-accumulate |

## The four silent failures

The bugs worth reading are the ones that produced plausible output rather than an error.
None was in the arithmetic, which is where all the early design attention went.

| what | how it looked | where |
|---|---|---|
| Requantisation shifted by `32 - s` instead of `s` | agreed with its own reference; 1137% off float64 | [Phase 0](#phase-0--native-kernel-against-the-evm-baseline) |
| Norm vectors read as a chunk when the layer was a directory | fluent, well-formed, wrong tokens | [generating on a live chain](#a-model-generating-tokens-on-a-live-chain-and-the-bug-that-found) |
| Deploy length wrapped at 65,536, truncating a directory | deploy succeeded, 111 of 13,219 chunks stored | [Qwen3-30B-A3B layer](#qwen3-30b-a3b-a-real-mixture-layer-on-chain) |
| Softmax over all experts rather than the chosen | every self-consistency test passed | [MoE routing](#the-router-precompile-8x-the-experts-for-100x-the-gas) |

Two of the four were in the storage format, which the original plan had no phase for. Every
one was found by running something real rather than by testing.

## Phase 0 — native kernel against the EVM baseline

**Verdict: the premise holds.** Native integer matmul is roughly four orders of magnitude
faster than the same operation in the EVM, and the gap is in the interpreter rather than in
the Solidity.

### Hardware

Apple M4 Pro, 14 cores, 24 GB unified memory. Theoretical memory bandwidth ~273 GB/s.

### The EVM side

`forge test --match-contract MatmulBench -vv` in the companion `on-chain-llm` repository.

| shape | weight bytes | gas | gas/MAC |
|---|---|---|---|
| 576 × 576 | 663,552 | 20,972,802 | 63 |
| 1024 × 1024 | 2,097,152 | 66,128,962 | 63 |
| 2048 × 2048 | 8,388,608 | 264,116,290 | 62 |

**63 gas per multiply-accumulate, flat across three orders of magnitude of shape.**

Worth stressing that this is not a strawman. `Transformer.matmul` is already the optimised
implementation: inline assembly, one `mload` unpacking sixteen `int16` weights at a time,
inner loop fully unrolled. The flatness across shapes confirms there is no per-call
overhead left to remove. This is what the operation costs in the EVM.

### The native side

`cargo run --release --bin bench`

| shape | MiB | scalar GB/s | parallel GB/s | speedup |
|---|---|---|---|---|
| SmolLM2-135M dim 576 | 0.6 | 39.9 | 40.7 | 1.0× |
| SmolLM2-135M ffn | 1.7 | 40.4 | 40.5 | 1.0× |
| Qwen3-4B dim 2560 | 12.5 | 41.9 | 86.9 | 2.1× |
| Qwen3-32B dim 5120 | 50.0 | — | 161.2 | — |
| Qwen3-32B ffn | 250.0 | — | 196.3 | — |

**196 GB/s at the large shapes** — about 72% of the machine's theoretical peak, so the
kernel is genuinely memory-bound rather than leaving headroom on the table. At two bytes
per weight that is **98 billion multiply-accumulates per second**.

A single core saturates near 40 GB/s and stays there regardless of shape, which is the
signature of a memory-bound kernel rather than a compute-bound one. Parallelism only helps
once the working set exceeds cache — hence the threshold below.

**This stopped being true when the accumulator widened to `i128`.** The table above was
measured with products formed in `i64`, and `tensor-kernel`'s own header notes that `i128`
costs roughly a quarter of the throughput. It cost more than that: the kernel is now bound
by forming products, not by fetching weights, and three separate measurements say so.

Batching sixteen activation vectors through one weight matrix reads the weights once
instead of sixteen times. If loads were the cost that is most of a 16×. On Qwen3-4B's own
shapes:

| shape | 16 separate | batched | |
|---|---|---|---|
| attention proj 2560×2560, 13.1 MB | 6.1 ms | 3.8 ms | 1.62× |
| ffn gate 9728×2560, 49.8 MB | 13.1 ms | 11.3 ms | 1.16× |

Both do the same 398M multiply-accumulates. Collapsing sixteen passes over 49.8 MB into one
saved 1.8 ms, so the reads were never what was being paid for.

Thread scaling says the same thing from the other side — a bandwidth-bound kernel plateaus
when the memory system saturates, and this one does not:

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| ffn gate, batch 16 | 88.0 ms | 46.0 ms | 23.3 ms | 12.2 ms |

Near-linear to 8, which is this machine's performance-core count. **Cores are the lever.**

And the arithmetic does not want cleverness. Splitting the multiply to avoid the 128-bit one
— `x = (x >> 32) * 2^32 + (x as u32)`, exact for negative `x` and assuming nothing about
magnitude, unlike the three bounded attempts the kernel header records — ran 20% slower:
89.1 s of node CPU for a Qwen3-4B reply became 104.6, 115.7, 109.0. LLVM already lowers a
widened `i16 * i64` to a single widening multiply.

The practical consequences are that prefill batching is not worth its contract surface, that
int8 weights would not help either since quantising operands does not reduce the number of
multiply-accumulates, and that a machine with more performance cores is the cheapest real
speedup available and needs no code change at all.

### Speedup

Two comparisons, measuring different things, both worth stating.

**Against chain capacity.** Ethereum sustains roughly 3M gas/sec (36M gas blocks every
12 seconds), which at 63 gas/MAC is 47,600 MAC/sec of chain throughput.

```
98.1e9 / 47.6e3  ≈  2,000,000x
```

**Against raw EVM execution.** Forge executed 4.39 billion gas of this test in 11.5
seconds, so revm runs at roughly 380M gas/sec — about 6M MAC/sec.

```
98.1e9 / 6.06e6  ≈  16,000x
```

The first number is the one that governs what fits in a block. The second isolates
interpreter overhead from the gas schedule, and is the honest figure for "how much of the
cost is the EVM being an interpreter."

### What this means per token

A forward pass performs roughly one multiply-accumulate per parameter per token.

| model | EVM gas/token | EVM chain time/token | native |
|---|---|---|---|
| SmolLM2-135M | 8.5e9 | 47 minutes | 0.003 s |
| Qwen3-4B | 2.5e11 | 23 hours | 0.04 s |
| Qwen3-32B | 2.0e12 | 7.8 days | 0.33 s |
| DeepSeek-V3 (37B active) | 2.3e12 | 9.0 days | 0.38 s |
| dense 1T | 6.3e13 | 8 months | 10.2 s |

The SmolLM2 row is a useful sanity check: it is consistent with the companion project's
observed behaviour, where a single token spans thousands of transactions.

**Phase 1 is worth building.** A 32B model goes from 7.8 days of chain time per token to a
third of a second on a laptop.

## Findings

**Thread dispatch dominates small shapes.** The first run showed the parallel path at 0.2×
and 0.4× scalar speed on shapes under 2 MiB — a real regression, not noise, and it would
have hit every attention projection in a small model. Rayon's per-call cost is fixed, so
the fix is a threshold in absolute work rather than a ratio: `PARALLEL_THRESHOLD`, set at
2M weights, below which the kernel runs scalar. After the fix those shapes measure 1.0×.

Because rows are partitioned but never split, switching between paths cannot change the
result, so the threshold is safe to tune without affecting consensus. Both sides of it are
covered by `parallel_matches_scalar`.

**Determinism holds where it must.** `thread_count_does_not_change_result` verifies
bit-identical output at 1, 2, 3, 7, and 14 threads on a shape above the threshold, so it
exercises the parallel path rather than trivially passing through the scalar one.

## Phase 1 — the precompile, in a real EVM

**Verdict: reachable and correct, and it exposed the constraint that defines phase 2.**

The precompile is registered at `0x100`, executes under the interpreter's own gas
accounting, and returns correct results both to a direct transaction and to contract
bytecode performing a `STATICCALL` — the path Solidity takes. Seventeen tests cover it,
including ten malformed inputs that would each panic an unchecked implementation.

### Passing weights as calldata does not work

Measured at 256 × 256, with the weights in the transaction's input:

| | gas |
|---|---|
| total transaction | 5,174,310 |
| the matmul itself | 164 |
| transport (calldata, intrinsic) | 5,174,146 |
| **ratio** | **31,549×** |

Making the arithmetic 16,000× cheaper accomplishes nothing when moving the operands costs
31,549× more than performing the operation. Calldata is priced per byte against scarce
block space; the matmul is priced against measured throughput. Those two schedules are not
close to each other, and at any real model width the gap only widens.

The property is sharper than the ratio. EIP-7623 places a floor on a transaction's gas
derived from its calldata, and at this width the floor already exceeds the cost of the
matmul by four orders of magnitude — so a gas limit low enough to starve the arithmetic is
**rejected before execution begins**. The computation is not merely cheap relative to
transport; it is unpriceable. `the_calldata_floor_makes_the_matmul_unpriceable` pins this.

**Therefore weights cannot travel as arguments.** They have to already be on the chain,
with the caller passing a reference and the precompile reading from state directly. That
was already the plan for phase 2; it is now the only viable design rather than an
optimisation.

### A consequence for the implementation

The simple precompile signature — `fn(&[u8], u64, u64)` — receives only its input bytes and
has no access to chain state. Reading weights from storage therefore requires implementing
the full `PrecompileProvider` trait, which is handed `&mut CTX` and can reach the journal.
That is a larger change than extending a precompile set, and it is the substance of phase 2.

### Incidental finding

revm's default spec enforces a per-transaction gas cap of 2²⁴ (16,777,216) under EIP-7825.
A purpose-built chain would raise it — worth noting because it is a second place, besides
the block gas limit, where a chain's configuration bounds how much work one transaction may
contain.

## Phase 2 — weights read from chain state

**Verdict: works, and moves the bottleneck to activations.**

Weights live in contract code. The caller passes an address and an offset; the precompile
reads them through the journal. Nothing about the model appears in the transaction.

This could not be an ordinary precompile. The `fn(&[u8], u64, u64)` signature receives only
its input bytes and never sees the chain, so reaching state meant implementing the
`PrecompileProvider` trait, which is handed the context. The result wraps Ethereum's own
precompiles rather than replacing them, and both matmul forms coexist: `0x100` takes
weights as calldata, `0x101` reads them from state.

Measured at 256 × 256:

| | calldata form (`0x100`) | state form (`0x101`) |
|---|---|---|
| input bytes | 132,360 | **1,056** |
| total gas | 5,174,310 | **55,200** |
| of which matmul | 164 | 164 |
| of which cold account | — | 2,600 |

**93× cheaper, and 125× less input.** The saving grows with `rows`, because calldata for the
state form scales with `cols` alone rather than `rows × cols`.

Price is independent of blob size — a call touching a small tensor inside a 64 GB model
costs the same as one against a 64 KB model. Without that, large models would be unusable
regardless of how little of them any single call reads.

### What the remaining 55,200 gas is — and why it does not matter

Almost all of it is the activation vector arriving as *transaction* calldata at 16 gas per
byte. That is the correct price for a transaction sent straight to the precompile, and the
wrong price for a forward pass, which does not re-enter the transaction between layers: the
contract holds the residual stream in memory and hands it over from there, where arguments
are priced as memory rather than calldata.

Measured directly, by running the same call a varying number of times in one transaction
and taking the increment:

| | gas |
|---|---|
| marginal cost of a call | 2,365 |
| of which matmul | 2,148 |
| of which everything else | **217** |

And that 217 is flat. At 32 columns it is 217; at 2048 columns — sixty-four times the
activation width — it is still 217, byte for byte:

| activation width | matmul | overhead |
|---|---|---|
| 32 columns | 228 | 217 |
| 2048 columns | 228 | 217 |

**So activations do not need to follow the weights into state.** Corrected projection for
Qwen3-32B, ~450 matmuls per token:

| per token | gas |
|---|---|
| matmul, all layers | ~31.3M |
| per-call overhead | ~0.1M |

Transport is about 0.3% of the cost, not 500%. Phase 2.5 is unnecessary and has been
dropped; the next step is the model registry.

### Two measurement traps worth recording

**EIP-7623's floor silently swallowed the first attempt.** A transaction pays the greater of
what it executed and a floor derived from its input. With a wide activation vector the floor
exceeded execution entirely, so one call and five calls reported *identical* gas — which
reads as "calls are free" when it means "nothing was measured". Carrying the arguments in
the calling contract's own code leaves the transaction empty and makes the increment real.

**A precompile absent from the set fails silently.** Before the state matmul was registered
as a member of `Precompiles`, a `STATICCALL` to `0x101` from contract bytecode found an
ordinary account with no code, returned success, and did nothing. Overriding `contains` was
not enough — membership in the set is what makes the interpreter treat the address as a
precompile at all. The symptom was the same as the floor's: an inner loop that appeared to
cost zero.

### A pricing bug the measurement caught

The weights account was being charged the cold access price on *every* call. A forward pass
touches the same account hundreds of times per token, so all but the first were overcharged
26x — roughly a million gas per token at 32B, for nothing. Access is now priced from the
`is_cold` flag the journal already returns, which took the per-call overhead from 2,716 to
217.

## Phase 3 — Solidity through the precompile

**Verdict: the stack closes.** A contract compiled by `solc`, executing on the EVM that
carries the precompile, gets bit-identical results to the reference kernel.

`TensorMatmul` wraps the call; `ModelRegistry` holds models and sessions. Five tests cover
registration, immutability, unregistered models, shapes the blob cannot satisfy, and the
round trip itself.

### Why the tests are in Rust

`forge` cannot test this. Its EVM is stock revm with no matmul precompile, so a Solidity
test would find an empty account at `0x101`, receive success with no data, and **pass while
computing nothing**. That is not hypothetical — it is exactly the failure that made an inner
loop appear free earlier in this project. The contracts are compiled with `solc` and run
against the EVM that actually carries the precompile.

The library defends against the same thing at runtime: it checks the returned length rather
than the success flag, so a contract deployed to a chain without the precompile reverts
instead of quietly producing a forward pass of zeros.

### Two corrections to the companion project

**One runtime, many models.** That project bakes a model's configuration into a constructor,
so every model is a separate deployment carrying its own copy of the inference code. Adding
a model is now a transaction rather than a deploy.

**Sessions are keyed and owned.** That project holds generation state in globals — one
generation chain-wide, and any address may call `start` and destroy another's work. Keying
state by session gives concurrency and access control from the same change.

**Registration is one-way and commits to `extcodehash` of the weight blob.** Both properties
serve the same end. Re-executing a computation proves the arithmetic and says nothing about
*which* parameters were used; a model whose weights could be repointed would make every hash
previously published against it a claim about something that no longer exists.

## Phase 4 — real weights

**Verdict: correct against the model, and it caught a bug every prior test had missed.**

Slices of the published Qwen3-0.6B checkpoint, quantised to `int16` with per-row
power-of-two scales, registered as a model, and projected through Solidity → precompile →
kernel. Compared two ways: bit-exactly against an independently derived integer answer, and
numerically against float64 from the original bf16 weights.

| fixture | shape | vs float64 |
|---|---|---|
| `self_attn.q_proj` | 48 × 1024 | 5.16e-06 |
| `self_attn.k_proj` | 32 × 1024 | 4.87e-06 |
| `mlp.gate_proj` | 64 × 1024 | 1.06e-05 |

### The bug float64 found and integers could not

The kernel requantised by `32 - shift` rather than `shift`.

A weight is stored as `w * 2^s` and an activation as `v * 2^32`, so their product carries
`s + 32` fractional bits. Returning to Q32 sheds exactly `s` — the activation's own 32 are
the ones being kept. The two expressions agree at `s = 32` and nowhere else.

Every integer test passed, because the reference vector generator implemented the same
mistake. The two agreed with each other and disagreed with the model by **1137%**. After the
fix, 7e-06.

The lesson is about test design rather than fixed point: two implementations sharing a
specification cannot validate that specification, however carefully each is written. Only an
external reference can.

### int16 encodes bf16 essentially losslessly

| | mean relative error |
|---|---|
| int16 weights vs bf16 weights | ~1e-06 |
| full integer path vs float64 | ~7e-06 |

bf16 carries eight mantissa bits; int16 with a per-row scale gives fifteen of magnitude. The
quantisation is not discarding anything the checkpoint had — the residual error is Q32
rounding of the accumulator, not the weights. This matters for the wider argument: the
`int16` representation is not a compromise made to fit on chain, it is a faithful encoding
of what these models already are.

### Qwen3 is not Llama

The companion project's exporter targets Llama and would have produced fluent nonsense here:

- `head_dim` is 128 while `hidden_size / num_attention_heads` is 64, so `q_proj` is
  2048 × 1024 and `o_proj` is 1024 × 2048 — **neither is square**, which that exporter
  assumes throughout.
- `q_norm` and `k_norm` are present at `head_dim`, applied per head before rotary.
- `rope_theta` is 1e6 rather than 1e4.

Each is silent when unhandled, so `tools/quantize.py` reports them rather than proceeding
quietly.

## Phase 5 — a real layer, and the finding that changed the design

**Verdict: the layer works, and a matmul-only precompile does not.**

### The layer

One block of the published Qwen3-0.6B checkpoint, in integers, against float64:

| position | mean relative error | max |
|---|---|---|
| 0 | 4.37e-06 | 3.50e-05 |
| 1 | 5.33e-06 | 6.12e-05 |
| 2 | 2.04e-05 | 1.01e-04 |
| 3 | 1.02e-05 | 5.38e-05 |

Attention with a KV cache, QK-norm before rotary, grouped queries, real rotary tables at
theta 1e6, SwiGLU, both residual paths. The error does not grow with position, so nothing
accumulates through attention or the cache.

`cargo run --release --example real_layer -- models/layer0` — an example rather than a test,
because it needs a 31 MB export that cannot be committed and a test that silently skips is
worse than none.

### Matmul is not where a chain's time goes

Attributing a layer natively gives 83% matmul and 17% everything else. That looks like
vindication for a narrow precompile. It is not, because the question is what remains *after*
matmul is accelerated.

Measured in the EVM (`forge test --match-contract OpsCost` in the companion repository):

| | gas |
|---|---|
| `exp`, one call | 6,309 |
| `silu`, one call | 6,235 |
| `isqrt`, one call | 3,905 |

Applied to one Qwen3-0.6B layer:

| | gas |
|---|---|
| matmul in the EVM | 990,904,320 |
| matmul via precompile | 15,360 |
| SwiGLU (3,072 × SiLU) | 19,153,920 |
| RMSNorm (26 × isqrt) | 101,530 |

**A layer goes from 99% matmul to 99% SwiGLU, and what is left behind is 1,247× the thing
that was fixed.** Accelerating only matrix multiplication does not remove the bottleneck; it
relocates it three orders of magnitude in the wrong direction.

### What that changes

A SwiGLU precompile at `0x102`, vectorised — one call per layer rather than one per element,
because at 3,072 elements the calling overhead would otherwise be the entire cost of what it
replaced. Priced at 4 elements per gas, so a layer's SwiGLU falls from 19.1M gas to 768.

The original argument was right about interface *shape* and wrong about *sufficiency*. A
precompile running a whole transformer block would still need revising for every
architecture variant — that objection stands. But no amount of narrowing makes matmul enough
on its own.

SwiGLU rather than the more general `exp` is a real trade. `exp` is universal, serving
softmax and GELU-through-tanh as well, but leaving the surrounding division and multiply in
the EVM keeps the operation an order of magnitude above the matmul beside it. SwiGLU covers
the Llama, Qwen, and Mistral families as they stand; GeGLU models need a sibling, and adding
one is additive rather than a revision — which is the property the narrow-interface argument
was actually protecting.

### A stale constant, found by chasing a timing gap

The layer took 2 ms where its 30 MB of weights should stream in 0.3 ms. Part of the answer:
`PARALLEL_THRESHOLD` had been calibrated when activations were `i32`, and widening them to
`i64` slowed the kernel without changing dispatch cost, so the crossover moved.
Qwen3-0.6B's `k_proj` is exactly 1024 × 1024, which sat just under the old value — **every
key projection of every layer was running single-threaded.** A sweep put the crossover at
2^19; 2^18 regresses a 576-wide projection to 0.7× scalar, 2^21 denies `k_proj` a 1.8× gain.

### Peak throughput is not layer throughput

The 196 GB/s in phase 0 was measured on a 250 MB slab. Real layers issue much smaller
matmuls and achieve far less:

| shape | MiB | GB/s |
|---|---|---|
| Qwen3-32B ffn | 250.0 | 106.8 |
| Qwen3-0.6B down | 6.0 | 43.7 |
| Qwen3-0.6B q_proj | 4.0 | 32.7 |
| Qwen3-0.6B k_proj | 2.0 | 20.6 |

Projections built on the peak figure are optimistic by two to five times at small model
widths. They converge as models grow, since a 32B layer's matmuls are themselves large.

## Phase 6 — the whole model

**Verdict: it reproduces the model, and picks the same token.**

Qwen3-0.6B end to end in integers — 28 layers, the final norm, the 151,936-row classifier —
against float64 computed from the original bf16 checkpoint.

```
cargo run --release --example full_model -- models/full
```

| | |
|---|---|
| logits vs float64 | 1.70e-03 mean relative, 9.59e-03 max |
| top-1 token | **agrees** |
| top-5 set | **agrees** |
| throughput | 46 ms/token, **21.6 tokens/sec** |

### Depth compounds error, by about two orders of magnitude

One layer matched float64 to 2e-05. Twenty-eight and a classifier come to 1.70e-03. That
stays far below what changes an answer, but it settles a question that could not be settled
by assumption: **a layer being right does not make a stack of them right**, and the only way
to know the margin was to run the whole thing.

### Numerical agreement is necessary and not sufficient

A language model emits a token, not a vector, so the check that gates this phase is the
argmax rather than the error. Ranks six and seven do swap — their float64 logits differ by
0.004, inside the measured error — which is the honest boundary of what this precision buys.
Reported rather than tuned away.

### Where the time goes at model scale

46 ms/token for 0.6B, of which 393 ms was a one-time load of 1.2 GB. The classifier is a
single 151,936 x 1024 matmul — 155M multiply-accumulates, more than a third of the model's
parameters — so at this size a vocabulary is not a detail.

## Qwen3-4B

Six point seven times the parameters of Qwen3-0.6B, running unchanged. The only code that
needed touching was the exporter, which could read one safetensors file and Qwen3-4B ships
as three.

| | Qwen3-0.6B | Qwen3-4B |
|---|---|---|
| layers | 28 | 36 |
| dim | 1024 | 2560 |
| parameters | 0.6B | 4.0B |
| gas per token | 112,574,489 | 176,377,176 |
| peak memory | — | 10.2 GB |
| tokens | all match float64 | all match float64 |

**Gas grows far slower than the model.** Six point seven times the parameters costs 1.57
times the gas, because arithmetic was never what a token was paying for — the per-layer
overhead and the cache dominate at small sizes, and only the arithmetic scales with
parameters. The same shape appears in wall clock: a bigger model is *relatively* cheaper on
this design, and the fixed costs stop mattering somewhere above here.

That also means the earlier per-token figures should not be extrapolated by parameter count.
A 32B model is not 53 times a 0.6B one.

### What bounds this machine

10.2 GB resident for a 4B model, because the harness holds every weight in memory at once.
A real node would not — weights are contract code, read through the state database — so this
is a property of the test harness rather than of the design. It is still what decides which
models can be checked here: 26 GB of RAM puts the ceiling around 8B, and the mixture-of-
experts variants need a router that does not exist yet.

## Caveats

- Projections assume weights stream at the measured rate and count nothing for attention,
  softmax, KV cache traffic, or weights that do not fit in memory. They are ceilings a real
  implementation is measured against, not predictions.
- 24 GB of RAM caps local work at roughly 7B parameters in `int16`. Anything larger streams
  from disk and will not reach these figures.
- The gas-to-time conversion uses Ethereum's throughput. A purpose-built chain would set a
  different gas schedule, which is the point — but the EVM figures are what a general chain
  charges today.

## Cross-architecture determinism — the premise, finally measured

Every claim in this repository rests on one property: two honest nodes on different
hardware compute identical bits. Until now that was an argument from the design — integers
have no reduction-order or FMA hazards, and the kernel partitions output rows without ever
splitting an accumulator — backed by a single-machine regression test. The argument was
never wrong, but it had never been run anywhere but one laptop, which is exactly the kind
of assumption that turns out to be false in the one place it matters.

`cargo run --release -p tensor-model --bin determinism` hashes three stages: the kernel
across seven shapes straddling the parallel threshold, the fixed-point primitives swept
rather than spot-checked, and the whole four-layer model over every position — logits *and*
residual stream, so a drift too small to change a token is still caught. It needs neither
revm nor solc nor a network, because determinism is a property of the arithmetic rather
than of the EVM around it, which is what makes it runnable on a machine you were handed
five minutes ago.

| machine | arch | OS | threads | OVERALL |
|---|---|---|---|---|
| Apple M4 | aarch64 | macOS | 1, 2, 3, 5, 8, 14 | `0146f1d1…dcc14` |
| Intel Xeon Platinum 8481C (c3) | x86_64 | Debian 12 | 1, 3, 4 | `0146f1d1…dcc14` |
| AMD EPYC 7B13 (n2d) | x86_64 | Debian 12 | 1, 3, 4 | `0146f1d1…dcc14` |

```
matmul     4424 values  ae138fc81759bf09829ddf7f5878b1402a092261c728bb7423ff771c96a406ee
fixed     16448 values  bd67b028a88362e2e2897c41ef3b72194005eee6fa82389e45951a2ef42bf750
model       480 values  05acb9b13248d6cab6e405f9ea23f0cac56e14af1fc1535aedb1051f1a4e779b
OVERALL                 168febfc534015d705e80c45e8543c51d9629c14b086fd1f9550f8edaea5f589
tokens generated        [124, 124, 42]
```

Three vendors, two instruction sets, two operating systems, twelve thread configurations:
one digest. Thread counts include 3 and 5 deliberately, since a count that divides the work
evenly would hide precisely the bug this is looking for.

What it establishes is narrower than it looks, and worth stating exactly. It shows the
arithmetic is portable — the property floating point cannot offer and the reason this
design is integer-only. It does not show that two *nodes* agree, because agreement between
nodes also covers block execution, state ordering and gas accounting, none of which this
binary touches. That remains the next thing to measure, and it is now the only part of
"verifiable" still resting on an argument rather than a number.

## Two nodes agreeing — verification by re-execution, measured

Cross-architecture determinism showed the *arithmetic* is portable. That is necessary and
not sufficient: agreeing on a matmul is not agreeing on a chain, which also requires nodes
to agree on which accounts exist, what code they hold, what every storage slot contains,
how many times each account has sent, and what everything cost. A state root covers all of
it in 32 bytes, so two nodes either produce the same hash or they have forked.

The node now records the raw signed transactions it accepted, and `--replay` executes such
a journal from genesis on a machine that never spoke to the original node. The journal
carries signatures, so a replaying node recovers each sender rather than being told who
they were — it is checking the transactions, not trusting a report about them.

`testdata/demo.journal` is the recording of `demo.sh`: deploy an uploader, store a weight
tensor as contract code, deploy the registry, register the model under its hash.

| machine | arch | block 1 | block 4 (FINAL) |
|---|---|---|---|
| Apple M4 | aarch64 | `0x6df3b9ee…` | `0x5618b7f8…` |
| Intel Xeon Platinum 8481C | x86_64 | `0x6df3b9ee…` | `0x5618b7f8…` |
| AMD EPYC 7B13 | x86_64 | `0x6df3b9ee…` | `0x5618b7f8…` |

```
block           gas  state root
    0             0  0x03e3b4afc4c929aba6511949b48be3d78a51b9c96757bff7e5ed398acbab210d
    1        471234  0x6df3b9ee7b8b49e6becfeed95048ee55f3cd960cf97c849b40b605afb8c1da70
    2         85334  0x63ae551d55fb314fd99cfbca8996fbfc5faf2503bfb0120a6c1d68c1307f1209
    3       1322369  0xb9ca365d01527196e68a71c320ca0ba72e2b261de894cbbd31036e8b725f57d2
    4        143157  0x5618b7f89cf47864ee58611808c9263d2d2d006b1df06aadd574c586c2810839
```

The gas column matches too, which the root does not cover and which would diverge first if
pricing depended on anything machine-specific. And the roots the replaying machines
computed are the roots the live node had already served over `eth_getBlockByNumber` — so
the agreement is with what the chain published, not merely between two replays.

The root is a real Merkle-Patricia root, computed as Ethereum computes it, verified against
the known empty-trie constant. A sorted digest would have detected divergence just as well
and been far easier, but it would have meant serving a field called `stateRoot` that no
Ethereum tool could check — and a number that looks standard while meaning something else
is worse than an obviously custom one.

### What this closes, and what it does not

Verification by re-execution now works end to end: transactions in, independent execution
on unrelated hardware, agreement on a published commitment. That was the premise the whole
design rests on, and it is no longer an argument.

It is still one implementation. Three machines running the same binary agreeing rules out
hardware, thread count, and OS — it does not rule out a bug in that binary, which a second
independent client would. That is the honest remaining gap, and it is the same gap every
young chain has; Ethereum's answer was multiple clients, and nothing here can substitute
for one.

There is also still no peer-to-peer and no consensus. Nodes agree here because a journal
was handed to them, not because they gossiped and settled on an ordering. Ordering is the
part a real deployment inside reth brings, and it is the last structural piece missing.

### A correction the state root forced

Making the header honest exposed that it was not. `transactionsRoot` was being reported as
the empty root on blocks that contained transactions, and `eth_getTransactionByHash` was
serving a nonce of zero and an empty input for every transaction — fabrications no client
could detect, since nothing about a well-formed JSON object says it is false. Transactions
are now serialised out of the signed bytes themselves, which is the only account of a
transaction that cannot drift from the truth, and `transactionsRoot` is computed. This also
fixed `cast block <n>`, which had been failing to deserialise every block but genesis.

`receiptsRoot` is still the empty root. It would need full receipt RLP with blooms and
nothing here consumes it — it is named as a shortcut in the code rather than left looking
like a result.

## A model generating tokens on a live chain, and the bug that found

Until now the model always ran in the same process that held the state. That proves the
arithmetic and the contracts, and it cannot prove the chain: a mistake shared between the
executor and the state it executes against is invisible from inside.

`tensor-cli` is a client and nothing else. It signs its own transactions, speaks only
JSON-RPC over HTTP, and encodes every call by hand rather than importing the encoder the
tests use — sharing that code would quietly reintroduce the shared assumption the client
exists to remove. It uploads a transformer to a running node as contract code, drives
`start`, `step` and `emitToken`, and compares what the chain emits against float64.

```
uploading weights as contract code
  layer 0     0x603e1bd7…  25664 bytes in 4 chunks
  …
  total       110976 bytes on chain

generating
  position 0: token   124   3 transactions   1377684 gas
  position 1: token   124   3 transactions   1208563 gas
  position 2: token    42   3 transactions   1232173 gas

expected    [124, 124, 42]
chain       [124, 124, 42]
```

34 transactions, 34 blocks, 31.9M gas. Replaying the node's journal from genesis reaches
`0xd89f39ad…`, the same root the live node published — so the tokens are not merely correct
once, they are reproducible by anyone holding the transactions.

### The bug: every real model would have produced wrong tokens

A blob larger than 24,000 bytes is stored as several chunks behind a directory. Every test
in this repository used a single chunk, and no real model can: a Qwen3-0.6B layer is 31.5MB,
so **every layer of every real model is a directory**. The one configuration that mattered
was the one never exercised.

`Transformer._norm` read norm vectors with a direct `extcodecopy`, the only place a contract
touches weights without going through a precompile. It understood single chunks only. Given
a directory it copied the address table and used it as norm weights.

Nothing reverted. The chain produced `[110, 15, 14]` where the model produces
`[124, 124, 42]` — fluent, well-formed, wrong. On a chain whose purpose is auditability
that is the worst failure mode available: an answer that looks like an answer.

Two things are worth recording about how it was found. It was not found by testing, but by
building a client that had to store weights the way a real deployment would, and the size
limit forced the directory path into use. And the code carried a comment noting the
single-chunk assumption — it identified the constraint as being about the *norms'* size,
when the real constraint was the *layer's*, which is thirty times larger and never
satisfiable.

`_norm` now resolves both forms. The regression is `chunked_storage_generates_the_same_tokens`,
which generates at four chunk sizes chosen to split tensors mid-row and leave a short final
chunk; it was confirmed to fail against the old code before being kept. The precompile's own
directory handling turned out to be correct at every offset and chunk size — that is now
tested too, rather than assumed, since it had never been.

## The real Qwen3-0.6B, on a live chain

1.19 GB of weights uploaded to a running node as contract code, then five tokens generated
by signed transactions over JSON-RPC, checked against float64.

```
layer 0 … 27   31,488,000 bytes each, 1,337 chunks each
lm_head       311,316,864 bytes, 13,219 chunks
total       1,192,980,864 bytes on chain

position 0: token 21806    326,732,122 gas
position 1: token   358    322,660,807 gas
position 2: token   829    323,053,187 gas
position 3: token   374    323,446,399 gas
position 4: token   444    323,842,735 gas

expected    [21806, 358, 829, 374, 444]
chain       [21806, 358, 829, 374, 444]
```

1,215 transactions, 1,215 blocks, 270.7 billion gas. The upload took about three minutes.

### The bug this found, which no fixture could

The first attempt failed at `emitToken`, after all 28 layers had run. The classifier's
directory listed **111 chunks where it should have listed 13,219**.

`WeightStore._deploy` wrote the code length into the SSTORE2 prologue as a `uint16`. A chunk
is capped at 24,000 bytes so it always fitted; a directory has no cap and grows with the
model. Qwen3-0.6B's classifier needs 13,219 chunks, so its directory is 264,381 bytes, and
264,381 mod 65,536 is 2,237 — the size actually deployed.

Nothing reverted. The deploy succeeded and returned an address. Twenty-eight layers, whose
directories are 26,741 bytes and fit, ran correctly. Only the classifier read weights that
were not there.

The prologue now uses a four-byte length and reverts past it rather than wrapping. `PUSH4`
costs three gas more than `PUSH2`, once per upload; being unable to express a length is not
a saving worth making.

Two things are worth recording. The comment above `_deploy` warned about exactly this family
of bug — it had shipped once already, as an offset that disagreed with a length — and it
guarded the offset while leaving the length to wrap. And no fixture could have caught it: a
directory only exceeds 65,536 bytes past about 3,277 chunks, which is 77 MB of weights. The
test now brackets that boundary directly, at 3,276 and 3,277 entries, rather than through a
model large enough to reach it.

### Directory resolution is O(chunks), and it dominates

| | gas |
|---|---|
| seed a position | 1,448,190 |
| 28 layers | 286,749,930 |
| classifier | 35,248,279 |
| **per token** | **~323,800,000** |

That is 10.2M gas a layer, against **648K measured earlier** for the same layer. The earlier
figure was not wrong; it was measured with the layer in a single chunk, which no real model
can be. The difference is the directory.

`resolve` loads each chunk's code in order to learn its size, including the chunks it is
about to skip, so reading a tensor near the end of a blob touches every chunk before it.
A 31.5MB layer at 23,552-byte chunks is 1,337 accounts, and each of a layer's seven
projections walks them again. The cost is O(chunks) per tensor read where it should be
O(chunks actually read).

This is a fixable property of the format rather than of the design: if each directory entry
carried its chunk's cumulative offset, resolution could seek to the first relevant chunk
instead of walking to it. I guessed that was worth roughly an order of magnitude. It was
worth 16%, and the next section is why.

## Seeking directories, and a prediction that was wrong

Directory entries now carry the blob's cumulative length through each chunk: twenty bytes of
address, four of offset. Where a byte lives is answerable from the directory alone, so a
reader seeks to the first chunk that matters instead of loading every chunk before it to
learn how long they were. The offsets are read from the chunks themselves at upload time
rather than taken from the caller, since a caller who mis-stated them would produce a
directory that resolves to the wrong bytes while looking perfectly well-formed.

Measured on the real Qwen3-0.6B, same 1.19 GB, same five tokens, same output:

| | before | after |
|---|---|---|
| seed a position | 1,448,190 | 1,445,730 |
| 28 layers | 286,749,930 | 240,742,126 |
| classifier | 35,248,279 | 35,248,279 |
| **per token** | **323,842,735** | **277,436,135** |

A 16% saving on a token, and **none at all on the classifier**. I had predicted an order of
magnitude. The reasoning behind that prediction was correct as far as it went — reading a
tensor near the end of a blob did touch every chunk before it — and it missed the thing that
actually determines the cost.

A layer's seven projections do not each read a small part of the blob. Together they read
*all* of it, because the blob is exactly those seven tensors. So the chunks a layer must
genuinely read were always about 1,337, and skipping is only ever saving the overhead of
passing chunks on the way to ones you were going to read anyway. The classifier is the
clearest case: it reads its whole 311MB starting at offset zero, so there is nothing in
front of it to skip, and its cost did not move by a single gas.

Seeking is still worth having — it removes an O(chunks) term per read, it takes the norm
vectors' repeated walks with it, and its benefit grows with how selectively a blob is read,
which is exactly what a mixture-of-experts model will do. But the dominant cost of a token
is reading the bytes the model actually needs, and no addressing scheme reduces that. That
is a floor set by the model's size, not by the storage format.

The general lesson is worth keeping: an optimisation aimed at work that was never the
bottleneck returns what that work cost, which here was a sixth of it. The measurement was
cheap and the prediction was free; only one of them was right.

## The router precompile: 8x the experts for 1.00x the gas

Expert selection is a precompile at `0x107` rather than contract code, and not for speed.
The router's own arithmetic is trivial — one row per expert against the model width. What
selection decides is *which parameters a token reads*, so two nodes that break a tie
differently do not round differently: they run different weights and emit different text.
Putting the comparison in the VM means every node runs the same comparison, written once,
rather than whatever a contract author's sort happened to do.

The measurement that justifies routing existing at all, taken through the precompiles in a
real EVM — score, choose, then read only the chosen experts:

| experts stored | chosen | gas |
|---|---|---|
| 8 | 2 | 110,230 |
| 64 | 2 | 110,260 |

Eight times the weights for 1.0002 times the gas. The thirty gas of difference is the router
scoring 56 more rows; the experts' weights are never touched, and they are the megabytes.

This is where the seeking directory format finally pays. Against dense layers it was worth a
sixth, because a layer's seven projections collectively read the whole blob and there was
almost nothing to skip. A router taking 8 of 256 skips 97% of a layer, and every skipped
chunk is an account never touched. The two changes are the same idea applied twice: cost
should follow what a token reads, not what the model contains.

### What had to be got right

`top_k` breaks ties by lowest index and is a selection sort rather than `sort_unstable_by`,
because "unstable" is precisely the property that must not appear anywhere in consensus.

The routing weights are a softmax over the chosen scores alone, not over all experts with
the losers dropped. The second form is self-consistent, plausible, and silently scales the
whole feed-forward down. Every self-consistency test passes against it; only the float64
comparison catches it, at 8.4e-4 against a mean magnitude of 8.7e-4.

Asking for more experts than exist is refused rather than satisfied with fewer. A caller
handed `k-1` weights would normalise over the wrong count and scale its feed-forward down
without noticing — the same failure by a different route.

### Still a mechanism, not a model

This is the router and the reference block, tested and measured. No published MoE checkpoint
runs here yet: that needs the layout in `Transformer.sol`, an arch flag in the registry, and
an exporter that writes experts as addressable offsets. The mechanism is the part that had to
be proven deterministic, and it is.

## A mixture layer through the contracts

`Transformer` now has a mixture branch: `nExperts` of zero keeps the dense feed-forward,
anything else routes. The layout needs no per-expert table. Experts are identical in shape,
so expert `e`'s gate, up and down each sit one stride past expert zero's, and adding an
expert to a model is appending bytes.

`Config` gained `nExperts`, `topK` and `expertStride`; `Layer` gained the router's offset.
All four sit after the fields a dense model uses, so a dense encoding is unchanged up to the
point it stops caring.

### Testing it without a second reference

Checking a mixture layer would normally mean writing the whole layer again in Rust —
attention, norms and all — which is a lot of code whose only purpose is to be wrong in
different ways.

There is a cheaper check that is also stronger. With `topK` of one the chosen expert's
routing weight is exactly one, so a mixture layer must produce *precisely* what a dense
layer produces when handed that expert's tensors. Same arithmetic, two paths through the
contract, and the comparison is exact rather than approximate. Any error in the stride, the
offset arithmetic or the scaling shows up immediately.

The trap is that the check passes trivially if the router happens to pick expert zero, since
then it compares expert zero against expert zero. The first version of the test did exactly
that. It now runs eight inputs, asserts equality on each, and requires that more than one
expert was actually reached — otherwise the stride was never exercised at all. A second test
requires that visiting two experts changes the output, which fails if the routing weights
are ignored.

### The offset that broke everything, and how it showed up

Widening `Config` from eight fields to eleven moved the offset to the layer array inside the
`Model` struct, which is written as a literal. Missing it produced panic `0x32` — an array
index out of bounds — from inside Solidity, on a dense model that had nothing to do with
mixtures.

That is the third time in this project a struct change has broken encodings written by hand
in two places at once. The failure is at least loud: an out-of-bounds panic rather than a
plausible number. The generation tests caught it before anything else ran.

## A mixture model, generating what float64 says

The layer tests establish that routing reads the right experts and sums them correctly.
This is the bar the dense path is held to: a whole model, every position, against float64.

`testdata/moe` is three layers routing two experts of six, so a token reads a third of each
layer and four sixths of every layer are never touched. It emits `[12, 46, 71]`, and so does
the chain — at one, two and all layers per transaction, and with the weights stored in
4,096-byte chunks behind directories.

`synth_model.py --experts N --top-k K` writes the fixture and its float64 reference together,
so the expected tokens come from a mixture computed in float64 rather than from the integer
path agreeing with itself. Its reference breaks ties by lowest index and softmaxes over the
chosen experts alone, matching the chain — the two mistakes that are self-consistent and
silently wrong.

### A fixture that predates the feature

`testdata/tiny` has no `n_experts` field, and the pinned determinism digest is taken against
it. Regenerating it to gain a field whose absence already means "dense" would have moved a
constant that exists to catch arithmetic changes. The harness defaults the field instead.
That is the right trade: a fixture is a fixed point, and changing one to make code tidier
spends the thing it is for.

## Qwen3-30B-A3B: a real mixture layer, on chain

A 30B model is tractable here for the reason mixtures exist: 30.5B parameters stored, about
3B read per token. Storage is the axis paid once; reading is the axis paid again by every
node on every token.

Layer 0 is 2048 wide with 128 experts of 768, eight visited. 1.25 GB, of which a token reads
76 MB.

### Natively, against float64

```
 pos         ms      rel err  experts chosen
   0        4.6      3.11e-4  [112, 64, 77, 67, 82, 0, 12, 124]
   1        4.6      7.80e-5  [51, 72, 44, 19, 15, 46, 64, 79]
   2        4.6      9.53e-5  [51, 121, 89, 111, 117, 4, 19, 41]
   3        4.4      1.37e-4  [72, 110, 64, 50, 62, 97, 43, 11]

  experts chosen        exactly float64's, every position
  whole layer error     2.78e-6
  5.7 ms a position, so 48 layers is 0.27 s a token
```

The error figure is the lesser result. The one that matters is that the integer router chose
the same eight experts of a hundred and twenty-eight, every time. Agreement to a tolerance
means nothing if the token then visits different parameters, and a trained router's eighth
and ninth experts are separated by very little — unlike a synthetic one, whose scores come
from a random matrix and are spread far apart.

### On the chain

```
uploading the layer as contract code
  1.25 GB, 1207 transactions, 281,516,279,237 gas

running the layer, on chain
  position 0:    145,382,234 gas   rel err 2.78e-6
  position 1:    136,970,704 gas   rel err 1.68e-6
  position 2:    137,345,500 gas   rel err 1.18e-6
  position 3:    137,712,272 gas   rel err 1.36e-6
```

The chain reproduces the native result to the digit: 2.78e-6 at position 0, from a layer
whose experts were chosen inside the VM by the router precompile.

137M gas a layer against 8.6M for a dense Qwen3-0.6B layer, for a model fifty times the
size — and 76 MB read rather than 1,208 MB. Extrapolating to 48 layers puts a token near
6.6 billion gas, which is a real number rather than an aspirational one: it is what a chain
with these limits would charge, and it is the cost of *reading 5.5 GB per token*, which no
scheme reduces.

### What could not be done, and why

The whole model is 61 GB, which this node cannot hold. Not for want of memory: every chunk
is an account, and the state root is recomputed over all of them each block at 0.53 µs an
account, so two and a half million accounts would spend roughly a day inside the trie rather
than in the upload. That is a property of this development node, not of the design — real
clients keep an incremental trie and pay only for what changed. It is now the first thing
that would have to be built to run a whole large model rather than a layer of one.

A layer was the right unit anyway. Layer 0 lives entirely in one of sixteen shards, so
answering the question cost a 3.7 GB download rather than 61 GB.

## Batching an upload: what it buys and what it costs

Uploading 1.25 GB, the same layer four ways:

| batch | transactions | upload gas | wall clock |
|---|---|---|---|
| 1 MB | 1,207 | 281.5e9 | 69 s |
| 2 MB | 598 | 291.6e9 | 60 s |
| 4 MB | 301 | 311.7e9 | 57 s |
| 8 MB | 152 | 351.7e9 | 54 s |

Eight times the batch is eight times fewer blocks and 22% less wall clock, for **25% more
gas**. The gas is not overhead in the chain's accounting — the EVM charges for memory
quadratically, and a `storeMany` call holds its whole batch in memory while decoding it.

The trade-off inverts with scale, which is the part worth keeping. The trie work an upload
does is blocks times accounts-so-far, and at a layer's 53,000 accounts the trie is not what
the time goes on — hence only 22%. At a whole model's 2.6 million it is nearly all of it, and
the same change is worth hours rather than seconds. Eight megabytes is what the gas cap
allows: at about 226 gas a byte, that is 1.81 billion against the node's 4 billion.

So the default is 8 MB, chosen for the case this exists for. It is the right default for a
reason worth stating plainly: upload gas is paid once by whoever publishes a model, while
token gas is paid again by every node on every token, and batching moves cost onto the
cheaper of the two. Nothing here changes what a token costs.

### Where a root's time goes

| | 20,000 accounts | 100,000 accounts |
|---|---|---|
| preparing leaves | 2.89 ms | 16.19 ms |
| building the trie | 6.89 ms | 33.81 ms |

Seventy per cent is the trie itself, so caching leaves — the obvious optimisation — could
recover at most the other thirty. The real fix is not to rebuild the trie at all: keep its
nodes and update the path from the changed leaves to the root, which is what every
production client does and what this node does not. That is the single thing standing
between a layer of a large model and a whole one, and it is not a flag.

## An incremental trie: 872x at the scale that mattered

The node rebuilt the whole account trie every block. That is correct, and it is what stood
between a layer of a large model and a whole one: a block cost time proportional to every
account that had ever existed, so an upload cost blocks times accounts.

It now keeps the trie between blocks and rebuilds only the path from a changed leaf to the
root. The set of accounts to update is the one revm already hands back — the accounts a
transaction touched — so the dirty set costs nothing to compute.

| accounts | one block (44 new accounts + root) | per account |
|---|---|---|
| 50,000 | 688 µs | 15.6 µs |
| 200,000 | 804 µs | 18.3 µs |
| 800,000 | 1.17 ms | 26.6 µs |
| 2,600,000 | 1.58 ms | 35.9 µs |

Fifty-two times the accounts for 2.3 times the cost: logarithmic, which is the shape it
should be. Rebuilding cost 0.53 µs an account, so at 2.6 million a block spent 1,378 ms in
the trie against 1.58 ms now — **872 times**. For a 61 GB model in 8 MB batches that is
about 84 minutes of trie work reduced to roughly 9 seconds.

Measured on the real 1.25 GB layer, the upload went from 69 s to 54 s at 1 MB batches. The
more useful result is what it does to the trade-off recorded above: 1 MB now costs the same
wall clock as 8 MB did and 20% less gas, because the reason to batch has largely gone.

### Two implementations, one root

`alloy-trie`'s one-shot builder was kept rather than replaced, and it is now the oracle. The
trie is checked against it after *every* insert and removal across four sequences —
insertion, scattered removal, in-place update, and interleaved random operations — because a
trie can be wrong in a way that only appears at one shape, and the final state is unlikely
to be that shape.

Removal is where the shapes live. A branch left with one child must stop being a branch, and
an extension onto an extension must merge, or two states with identical contents hash
differently — a fork between a node that took the shortcut and one that did not.

A separate test replays the recorded journal and compares the incremental root against the
from-scratch root after every transaction. The trie's own tests prove the data structure;
this proves the wiring, which is a different claim: that the accounts a transaction reports
touching are exactly the accounts whose leaves changed. An account mutated without being
reported would pass every test in `trie.rs` and fork the chain.

There is no invalidation step, and that is deliberate. Insert and remove rebuild every node
on the path they touched and leave the rest alone, so a stale cache cannot outlive a change
beneath it — the node that held it was replaced.

## The whole Qwen3-30B-A3B, on chain

60,465,686,912 bytes of weights uploaded to a running chain as contract code, then five
tokens generated by signed transactions over JSON-RPC. Every one is the token float64 says
the model emits.

```
layer 0 … 47   1,246,733,440 bytes each, 52,936 chunks each
lm_head          622,481,792 bytes, 26,431 chunks
total         60,465,686,912 bytes on chain

position 0: token   353    8 transactions   3,788,406,101 gas
position 1: token   358    8 transactions   3,782,193,157 gas
position 2: token   829    8 transactions   3,784,723,141 gas
position 3: token   374    8 transactions   3,787,319,501 gas
position 4: token 20445    8 transactions   3,789,883,973 gas

expected    [353, 358, 829, 374, 20445]
chain       [353, 358, 829, 374, 20445]
```

7,318 transactions, 7,318 blocks, 17.1 trillion gas, on an `n2-highmem-16` that peaked at
69 GB of its 125. About two hours for the upload; the instance was deleted immediately
after.

**3.79 billion gas a token** for a 30.5B-parameter model — against 277M for the 0.6B dense
one. Fifty times the parameters for fourteen times the gas, because a token reads eight
experts of a hundred and twenty-eight and never touches the rest.

A token needs eight transactions because 3.79 billion exceeds any single transaction's cap.
That is the granularity knob doing what it was built for, and it is also the check that it
works: splitting a token across transactions must not change the token, and here it did not.

### What this took that a layer did not

Three things, none of them the model:

**The trie.** Rebuilding it every block was fatal at 2.6 million accounts — 1.4 seconds a
block, days of trie work. Keeping it and updating only changed paths made a block's cost
flat, and 7,318 blocks spent seconds rather than days.

**Streaming the export.** 61 GB is 244 GB in float64, so the reference is computed layer by
layer with the loop inverted — layers outside, positions inside — which works because a
layer's key/value cache is used by nothing else. Peak memory is one layer.

**Not archiving transactions.** 60 GB of weights arrive as 60 GB of signed transactions.
Keeping both is 120 GB before any state, and a machine that can hold the model cannot
necessarily hold the record of how it arrived. That is now a flag, on by default, because
answering honestly about a transaction is worth more than the memory in every case but this
one.

### What it does not show

It is one node. The tokens are reproducible — the journal and the state root are there for
anyone to re-execute — but nobody has. Cross-machine determinism was established at layer
scale and on the smaller models, not by a second party running this 60 GB chain.

And it is still a development node: no peer-to-peer, no consensus, no ordering. This shows a
30B mixture model *fits* and *runs* under an ordinary Ethereum interface at a cost anyone can
read off a receipt. It does not show a network agreeing on it.

## Determinism on a trained router

The determinism digest covered synthetic weights, which is the right way to test arithmetic
and the wrong way to test routing. A random router's scores are spread far apart, so its top
eight of a hundred and twenty-eight are never in doubt. A trained router's are, and
disagreeing about them is not a rounding difference — it sends the token through different
parameters.

`determinism --moe <layer>` hashes the router's scores, the experts chosen, and the
mixture's output for a real Qwen3-30B-A3B layer. The scores are hashed as well as the
decision on purpose: if two machines ever disagree, the digest says whether they differed in
the arithmetic or only in what it led to.

| machine | arch | moe digest |
|---|---|---|
| Apple M4 | aarch64 | `5cec5618…ede0` |
| Intel Xeon Platinum 8481C | x86_64 | `5cec5618…ede0` |
| AMD EPYC 7B13 | x86_64 | `5cec5618…ede0` |

```
0: [112, 64, 77, 67, 82, 0, 12, 124]
1: [51, 72, 44, 19, 15, 46, 64, 79]
2: [51, 121, 89, 111, 117, 4, 19, 41]
3: [72, 110, 64, 50, 62, 97, 43, 11]
```

The same eight experts of a hundred and twenty-eight, on three CPU vendors and two
instruction sets, every position.

The digest is deliberately *outside* the pinned `OVERALL` value, because the stage needs a
1.25 GB export that cannot be committed — folding it in would make a pinned constant depend
on whether somebody happened to have the weights.

## The 30B chain, produced twice, on two CPU vendors

The whole model running on a chain was one machine's word for it. It has now been done
again, independently, on a different CPU vendor — from downloading the checkpoint to
generating the tokens.

| | Intel Xeon Platinum 8481C | AMD EPYC 7B13 |
|---|---|---|
| bytes on chain | 60,465,686,912 | 60,465,686,912 |
| transactions | 7,318 | 7,318 |
| total gas | 17,075,759,817,727 | 17,075,759,817,727 |
| tokens | `[353, 358, 829, 374, 20445]` | `[353, 358, 829, 374, 20445]` |
| **state root** | `0x29bb017eed6f828b74381a515f48b30ba9892e6dd1866810cfcdd58d8fd1ea68` | `0x29bb017eed6f828b74381a515f48b30ba9892e6dd1866810cfcdd58d8fd1ea68` |

Identical, to the digit, including the gas.

This turned out not to need a journal. The client signs deterministically — a fixed key,
sequential nonces, and transaction data that is a function of the weights — so running the
same pipeline is running the same transactions. The contract addresses gave it away before
the roots did: layer 41 landed at `0x29d0ea43…` on both machines, and a contract address is
derived from its sender and nonce, so identical addresses mean identical sequences.

That is a stronger check than replaying a journal would have been, because it covers more.
A journal starts after the transactions exist; this starts from the safetensors. The export
is float64 and numpy, which is *outside* the integer determinism guarantee everything else
here rests on — so if quantisation had rounded differently on one machine, the weights would
have differed, and the roots with them. They did not.

What it still does not cover: both runs were mine, and both ran the same binary. Two vendors
agreeing rules out the hardware. It does not rule out a bug in the code they both ran, which
only a second implementation would.

## The precompiles inside reth

Everything measured above runs revm directly, which is why none of it is a chain: no
peer-to-peer, no consensus, no ordering. reth supplies exactly that and asks for exactly
what this project already has — its `EvmFactory` names the precompile set as an associated
type, so carrying a custom one is a matter of saying which.

`crates/tensor-reth` does. `TensorPrecompiles` — the same provider the development node
uses, unchanged — is reth's precompile set, and `EthEvm<DB, I, TensorPrecompiles>` compiles.

The version pin is the entire reason this works. reth's latest release, v2.4.1, still
carries revm 41; these precompiles are written against 42. reth's main branch moved to
42.0.1, which is the version this project already builds against, so the dependency is
pinned to a commit rather than a tag.

That matters more than it sounds. Most of reth's own examples customise precompiles through
`PrecompilesMap`, which is a table of stateless functions — the wrong shape entirely, since
a matmul reads its weights from chain state and only a provider is handed the context to do
that. What made this a small change rather than a fork is that `EvmFactory::Precompiles` is
unconstrained at the trait level, so a state-reading provider is as acceptable as a table.

### A wall that was not one

`EthEvmConfig` fixes the precompile set to `PrecompilesMap` — a table of stateless
functions. That looked fatal: a matmul reads megabytes of weights out of chain state, and a
function handed only its calldata cannot.

alloy-evm's *dynamic* precompiles are handed an `EvmInternals`, which can load accounts, and
loading accounts is the only thing weight resolution ever needed. So the adapter is thin:
turn `EvmInternals` into the code source the resolver takes, and call the same function.

Making that possible meant a refactor worth more than the integration. Weight resolution
reached revm's journal directly, which was fine while revm was the only host. It now takes a
code source — address to code, and whether the touch was cold — so both hosts run the same
directory walk, the same seeking, and the same cold and warm accounting. The state-reading
matmul went the same way. A second implementation of what a matmul computes would be a
second opinion about what a model says, which is the one thing a chain cannot have two of.

Running a matmul through an EVM that reth built, against weights held in chain state:

```
  precompiles   0x100-0x107, inside an EVM reth built
  shape         64x96, weights read from chain state
  gas           44180
  result        matches the kernel exactly
```

### What this establishes, exactly

That reth's EVM **executes these precompiles correctly**, against weights in chain state,
with the same result as the kernel. The executor builder is written and the raised limits
are applied where reth builds its EVM.

### And it runs as a node

```
cargo run --release -p tensor-reth          # a reth node, chain 4242, RPC on 8545
RPC=http://127.0.0.1:8545 ./demo.sh         # the same demo, against it
```

The demo deploys the uploader, stores a tensor as contract code, deploys the registry,
commits to the weights by hash, and multiplies by them through the precompile. The result
matches the reference, and four blocks exist afterwards — produced by reth's own consensus
engine, from reth's own transaction pool.

Everything that makes it a chain is reth's and untouched: the network, the pool, the payload
builder, the engine. The only substitution is the executor.

Two things cost time and are worth recording.

**Dev mining is a debug capability.** `launch()` starts a node that accepts transactions into
the pool and never seals a block, which looks exactly like a stuck chain rather than a
missing flag. `launch_with_debug_capabilities()` is what mines.

**Account code is loaded lazily, and an in-memory database hides it.** `load_account` returns
an account carrying a code hash and an empty `code`, so weights came back as nothing and
every matmul reverted. The self-check above did not catch it, because `CacheDB` keeps code
inline in the account — it answered the wrong question correctly. `load_account_code`
followed by `load_code` is what actually fetches bytes, and the lesson generalises: a test
harness that stores things more conveniently than the real thing will confirm code that
cannot work against the real thing.

The crate is deliberately its own workspace. reth is hundreds of crates, and folding it into
the main one would put that build in front of every test run — including `verify.sh`, which
exists to be quick enough that nobody skips it. The main workspace still builds and tests in
seconds, unchanged.

### Two nodes, and what gossip does not decide

The node now takes `--rpc-port`, `--p2p-port`, `--peer` and `--follow`, so two can run side by
side. They connect: the follower reports `connected_peers=1` against the sealer's enode,
with discovery off and the peer given explicitly.

The sealer produced four blocks. The follower stayed at zero.

That is not a fault, and it is the most useful thing this attempt produced. **A post-merge
Ethereum node does not choose its head from gossip.** Blocks propagate over peer-to-peer,
but what advances the chain is a consensus client calling `engine_forkchoiceUpdated`. This
chain's genesis sets `terminalTotalDifficulty: 0`, so it is post-merge from block zero, and
a follower without a consensus layer has nothing to tell it which block is canonical. It
sits at genesis holding valid blocks it has no authority to adopt.

So "two nodes agreeing because they gossiped" was the wrong target — that is not how
Ethereum has worked since the merge. There are two right ones:

- **Drive the follower's engine API.** Take the sealer's blocks and feed them to the follower
  as `engine_newPayload` plus `engine_forkchoiceUpdated`. The follower re-executes each block
  with its own EVM and either accepts it or rejects it on the state root. That is exactly
  independent verification, and it is what a consensus client does.
- **Run a consensus client.** Point a real one at both nodes and let it do the same thing for
  real.

The first is a few hundred lines and needs JWT auth against the engine port; the second is
an operational task rather than a coding one. Either produces the missing measurement:
a second client accepting or rejecting blocks it did not produce.

What is established meanwhile is that the peering works and both nodes carry the
precompiles, which is everything up to the point where a consensus layer takes over.

## A model generating tokens on reth

```sh
cargo run --release -p tensor-reth                       # a reth node, chain 4242
RPC=http://127.0.0.1:8545 GAS=16000000 MAX_LAYERS=1 \
  cargo run --release -p tensor-cli -- testdata/tiny     # a model, generating on it
```

```
uploading weights as contract code
  layer 0 … 3    25,664 bytes each          lm_head  8,320 bytes
  total         110,976 bytes on chain

generating
  position 0: token   124   6 transactions   1,501,214 gas
  position 1: token   124   6 transactions
  position 2: token    42   6 transactions

expected    [124, 124, 42]
chain       [124, 124, 42]
```

Thirty transactions, thirty blocks, sealed by reth's own consensus engine from reth's own
pool. Every part of a chain is reth's; the only substitution is the executor.

### Five things a real client wanted that a development node did not

None was about the arithmetic, and every one was a place the harness had been quietly
answering an easier question.

**Receipts do not exist yet.** The development node seals a block inside the send, so a
receipt was always there when it returned. A real client mines asynchronously, and reading
the receipt immediately returns `null` — which the client read as a revert, because a
missing status and a failed one are both "not `0x1`". It now waits, and treats absence as
"not yet" rather than as failure.

**The pool enforces the gas cap itself.** Raising `tx_gas_limit_cap` in the EVM is not
enough: reth's transaction pool checks EIP-7825's 2^24 limit before execution, so a
transaction asking for more is rejected before any EVM sees it. Against a stock pool, the
granularity knob is what keeps a token inside the cap — which is what it was built for.

**A pool will not take a transaction that offers nothing.** Fees were zero, which the
development node ignored and charged nothing for either way.

**`eth_call` comes from nobody.** Without an explicit `from`, a call arrives from the zero
address, and a session that checks its owner refuses it. The development node never noticed,
because nothing it served was owned.

**Return data has to come from somewhere.** A receipt does not carry it. `tensor-node` keeps
it and serves `tensor_getOutput`; a stock client does not. The value is now read with
`eth_call` against the state the transaction is about to be applied to, and then the
transaction is sent — the same state, so the same value.

The pattern is worth more than the list: every one of these was a difference between a
harness that was convenient and a client that is real, and none would have been found by
testing against the harness.

## Where a token's gas actually went: 5.4x from one constant

A Qwen3-0.6B layer costs 15.73M multiply-accumulates, which at 1,024 MACs a gas is **15,360
gas of arithmetic**. It was measured at 8.6M. Fifty-seven times the arithmetic was going
somewhere else, and it had never been attributed.

It was account access. Every chunk of weights is an account, a cold touch is 2,600 gas, and
a 31.5MB layer stored in 24,000-byte chunks is 1,337 accounts — about 3.5M gas of touching
accounts to do 15,360 gas of arithmetic.

The chunk size was 24,000 because that is EIP-170's 24,576 with room for a tag byte. **This
chain does not have EIP-170's limit.** It raises the contract code size to 64MB, because a
layer of a real model cannot fit in 24KB — and then inherited the constant that limit
implies anyway.

Raised to 1,000,000 bytes, measured on the real Qwen3-30B-A3B layer, same weights and the
same answer to the digit:

| | 24,000-byte chunks | 1,000,000-byte chunks |
|---|---|---|
| chunks in the layer | 52,936 | 1,248 |
| upload transactions | 1,207 | 159 |
| **gas a position** | **145,382,234** | **27,050,044** |
| relative error | 2.78e-6 | 2.78e-6 |

**5.4x.** Extrapolated across 48 layers, a token goes from about 6.98 billion gas to 1.30
billion.

Two things about this are worth keeping. The optimisation was invisible for as long as
nobody asked where the gas went — every measurement in this file above it is correct and
none of them prompted the question. And the cause was a constant that was right for a
different chain: the code inherited Ethereum's limit while explicitly raising Ethereum's
limit, twelve lines apart.

## What the remaining gas is: not arithmetic

With the profiler working again, a dense Qwen3-0.6B layer breaks down as:

```
o proj                                       184,368   27.4%
down proj                                    186,805   27.7%
swiglu                                        76,671   11.4%
gate + up proj                                62,478    9.3%
attention                                     38,054    5.6%
… norms, rope, cache                          …
measured inside the contract                 673,643

  the precompiles themselves charge            18,712
  memory touched  315,136 bytes                218,964   (33%)
```

**The precompiles are 2.8% of a layer.** The other 97.2% is Solidity handing them their
arguments — packing activations into `bytes`, copying them across call boundaries, and
paying the EVM's quadratic memory price for the space.

That reframes what is left to optimise. The arithmetic was made native and is now
negligible; weight access was 3.5M gas and is now a fraction of that; what remains is
argument marshalling inside the contract. A scratch buffer is already reused, so the cost is
not repeated allocation — it is the copying itself, and the two projections that dominate
(`o` and `down`, 27% each) are exactly the two with the widest activations.

The shape of the answer has been the same at every stage of this project: whatever was made
fast stopped being the cost, and something nobody had measured took its place.

## Two nodes agreeing on inference

The gap this file kept naming is closed. A second reth node re-executed every block it did
not produce — the weight uploads, every layer, the classifier — with its own EVM against its
own copy of the state, and reached the same state root.

```
 block   txs           gas  status
    25     1       318,986  VALID
    26     1       318,986  VALID
    27     1       318,986  VALID
    28     1       318,986  VALID
    29     1        72,491  VALID

sealer   block 29  root 0x7245b34836674da6d77f8e695ef4328886062cb63f70b5c6c5f3142ef677d038
follower block 29  root 0x7245b34836674da6d77f8e695ef4328886062cb63f70b5c6c5f3142ef677d038

  agreed: the follower re-executed every block and reached the same state
```

Those 318,986-gas blocks are layers of a transformer. The follower never uploaded the model
and was never told what the answer should be; it was handed transactions and recomputed
them.

**`VALID` is the measurement.** It means the follower's own execution produced the state root
the block claims. A token generated on one machine was checked by another that had only the
transactions — which is what "verifiable inference" was supposed to mean all along.

### Why it took a consensus client rather than gossip

The two nodes were already peered, and it changed nothing. A post-merge Ethereum node
propagates blocks over peer-to-peer but adopts one only when a consensus client calls
`engine_forkchoiceUpdated`. A follower without one sits at genesis holding valid blocks it
has no authority to canonicalise — which looks like a broken sync and is correct behaviour.

`tools/engine_relay.py` is that consensus client, reduced to the one job that matters here:
hand over a payload, let the follower verify it, record the verdict, then name the head. Real
consensus decides *which* chain; this decides nothing and only asks whether the follower
agrees with the one it is given.

Three details were needed and are worth recording. The follower serves
`debug_getRawTransaction`, because an execution payload carries raw signed transactions
rather than the objects `eth_getBlock` returns — and the point is that the follower verifies
the signatures itself. reth reads its JWT secret rather than creating it, so the file has to
exist before launch. And Prague is active from genesis on this chain, so `newPayloadV4`.

### What is still not shown

Both nodes run the same binary. Two vendors and two processes agreeing rules out hardware
and rules out one process fooling itself; it does not rule out a bug they share. That needs a
second implementation, and it is now the only structural gap left.

```sh
./demo-network.sh    # a sealer, a follower, a model, and a verdict on every block
```

## Fusing the feed-forward: 18.7% off a layer

The profile said the precompiles charge 2.8% of a layer and Solidity handing them their
arguments charges the rest. The feed-forward is where that costs most: gate, up and the
activation are all `hidden` wide — three times the residual stream — and each was packed
into `bytes`, copied across a call boundary, copied back, and paid for again in the EVM's
quadratic memory price.

Precompile `0x108` does all four steps in one call. The caller passes `dim` activations and
receives `dim` back; everything of width `hidden` now exists only inside the VM, where it is
a `Vec<i64>` and costs nothing.

| | four calls | one call |
|---|---|---|
| gas a layer | 673,643 | 547,653 |
| memory touched | 315,136 bytes | 249,312 bytes |
| memory cost | 218,964 (33%) | 141,927 (26%) |

**18.7% off a layer**, and a 28-layer token from about 47M gas to 44M. The saving is entirely
in carrying arguments; the arithmetic is unchanged and the precompiles still charge 18,712
gas of it.

The equivalence is exact rather than approximate, and the test says so: the fused path must
equal the three calls it replaces bit for bit, at three shapes. It is the same arithmetic in
the same order against the same weights, so anything else would mean the fused path is a
second opinion about what a layer computes rather than a cheaper way to reach the first.

Both feed-forward paths use it — dense and mixture — and both hosts have it, so reth and the
development node still run one implementation.

### What is left of the 97%

It is now 96.6%, which is the honest way to put it: fusing moved a fifth of the cost and did
not change the shape of the problem. What remains is the attention half of a layer — the
query, key and value projections, the rotary tables, the cache, and the output projection —
each still a separate call with its own arguments. The same trick would apply, and the same
caution: every fusion is another place where one implementation could disagree with itself.

## Fusing attention: worse, then better, and why

`o_proj` was the largest single line in a layer — 184,368 gas of 673,643 — and almost none of
it arithmetic. Attention returns `heads * head_dim` values, twice the residual stream in
Qwen3-0.6B, and nothing but the output projection ever reads them. Fusing the two looked like
the same win the feed-forward gave.

It measured **2.4% worse**: 561,069 gas against 547,653.

The saving is real — one round trip of the attention result — and the wrapper gives it back.
`Tensor.attention` builds its input through `_build`, which reuses the scratch buffer a layer
already allocated. The fused wrapper cannot: its header is 44 bytes and `_build` takes a
single `bytes32`, so it allocates a fresh buffer and copies the query and both caches into
it. Those are larger than the result it saved passing.

So `_build` now accepts a head wider than one word — a second `bytes32`, written only when
the header exceeds thirty-two bytes — and the fused wrapper goes through the scratch path
like every other one.

| | two calls | fused, fresh buffer | fused, scratch |
|---|---|---|---|
| gas a layer | 547,653 | 561,069 | **525,382** |
| memory touched | 249,312 | 265,728 | 232,992 |

**4.1% off a layer**, and a 28-layer token from about 44M gas to 43M.

Sizing the scratch was the other half of it. `scratchFor` reserved sixteen bytes of header
for an attention call; the fused one needs forty-four, and the shortfall was a "scratch too
small" revert on every layer — which showed up as the test count dropping from 123 to 77
rather than as an error anybody would read. Counting passing tests rather than looking for
failures is the only reason that was caught quickly, and it is the third time in this project
that has been true.

The lesson survives the eventual win, and is worth more than it. Fusing is not a saving in
itself; it trades one copy for another, and which is larger has to be measured. This one was
2.4% worse until the buffer it allocated was the buffer the layer already had — the fusion
was never the problem, and measuring before adopting is what distinguished the two.

Both fusions together: a layer went 673,643 to 525,382, and the precompiles still charge
18,712 gas of it.

## The profile, once the labels were honest

Fusing moved work without moving the `_mark` boundaries, so the profile kept printing
"rmsnorm (ffn)" against a number that was the whole feed-forward. Corrected:

```
rmsnorm (attn)                                25,239    4.8%
k/v proj + QK-norm + rope + cache             31,408    6.0%
q proj                                        19,167    3.6%
QK-norm + rope (q)                            25,314    4.8%
attention + o proj  (fused, 0x109)           216,250   41.2%
rmsnorm (ffn)                                  8,560    1.6%
gate + up + swiglu + down  (fused, 0x108)    199,444   38.0%
                                             -------
                                             525,382
```

The two fused calls are 79% of the layer, and the precompiles inside them charge 18,712 gas
in total. So what is left is not many small costs but two large ones, each dominated by
moving a residual stream in and a residual stream out.

That points at the end of this line rather than another step along it: **a layer is one
precompile**. Input the residual stream, the cache, the rotary tables and the weight
offsets; output the new residual stream and the cache extension. Every intermediate — the
query, the keys, the values, the attention result, the gate, the activation — stays inside
the VM, and Solidity carries 8KB in and 8KB out instead of the hundred kilobytes it moves
now.

It is also where the caution bites hardest. Each fusion so far has been checked bit-for-bit
against the calls it replaced, and that gets harder as the fused thing gets larger: a
whole-layer precompile would be checked against `forward_layer`, which is the reference the
whole project is measured against. Worth doing, and not worth doing quickly.

## Everything together, on the real 30B layer

The same Qwen3-30B-A3B layer, same weights, same answer to the digit, with the chunk size
raised and both fusions in place:

| | before | after |
|---|---|---|
| first position | 145,382,234 | 26,802,436 |
| steady state | ~137,000,000 | **18,133,826** |
| relative error | 2.78e-6 | 2.78e-6 |

**7.6x** on the cost of a layer once the weight accounts are warm. Across 48 layers that puts
a token near **0.87 billion gas** rather than 3.79 billion.

The first position costs more than the rest and always will: it is the one that touches every
weight account cold, at 2,600 gas each against 100 warm. That gap is now visible precisely
because the rest got cheaper — at 137M a position it was noise.

Three changes did it, and none touched the arithmetic:

- **The chunk size**, which was Ethereum's code limit on a chain that does not have it. Fewer,
  larger accounts.
- **The feed-forward fused**, so `hidden`-wide intermediates never leave the VM.
- **Attention fused with its projection**, once the scratch buffer could carry a wider header.

Every one was found by asking where the gas went rather than by guessing, and the answer was
never the multiply-accumulates. They were 15,360 gas of a layer at the start and they are
15,360 gas of it now.

## Wiring the whole-layer precompile: what stopped it

`0x10a` is written, tested and registered on both hosts, and is not yet used by a contract.
Two things surfaced while wiring it that are worth stating before anyone starts.

**The header is three words and `_build` carries two.** Widening it again would mean packing
eleven offsets and an address across word boundaries by hand. The better answer is to build
the header with `abi.encodePacked` and pass it as the first *part* with a head length of
zero — the layout is exactly what `encodePacked` produces from six `uint32`, an `address`,
eleven `uint32` and an `int64`, and no bit shifting can go wrong.

**The cache does not line up.** A contract holds `kv.k` covering positions zero through
`pos`, because `_extendCache` writes this position's entry before attention reads it. The
whole-layer call wants the entries *before* this position and computes the current one
itself. So a caller must either slice — which in Solidity is a copy, and copying the cache is
a good part of what fusing was meant to avoid — or `forwardLayer` must stop extending the
cache itself and let the precompile return the entry to append.

The second is right, and it is a change to the contract rather than to the precompile. It is
also exactly the kind of change that was worth stopping in front of: the last time the layer
was rewired, a scratch buffer sized for the old call showed up as the test count dropping
from 123 to 77 rather than as an error, and that was caught only by counting passing tests.

## A whole block in one call: 10.3x on a dense layer

`0x10a` is now what a dense layer calls. One precompile: residual stream, cache, rotary row
and weight offsets in; new residual stream and the cache entry this position added, out.
Every intermediate — query, keys, values, attention result, gate, activation — stays inside
the VM.

| | separate calls | two fusions | one call |
|---|---|---|---|
| gas a layer | 673,643 | 525,382 | **51,183** |
| memory touched | 315,136 | 232,992 | 139,552 |

**10.3x against the fused form, 13.2x against where this started.** The precompiles still
charge 18,712 gas of it, which is now 37% of a layer rather than 2.8% — the marshalling that
dominated has mostly gone, and what is left is one call's arguments.

**Mixture layers are unchanged and still go step by step.** Which experts a token visits is
decided in the middle of a layer, and the fused form has no router — so Qwen3-30B-A3B still
measures 18.1M gas a position, exactly as before. The fusion helps dense models; a fused
mixture block would be a separate precompile that resolves its experts internally, and it is
the obvious next one.

### Three things it took

The header is ninety-six bytes, which is three words, and `_build` carries two. Building it
with `abi.encodePacked` and passing it as the first *part* avoids packing eleven offsets and
an address across word boundaries by hand.

The caches go in whole rather than sliced. A contract holds `kv.k` with room for the entry
being computed; the precompile reads the earlier entries, ignores the last slot, and returns
what belongs there. Slicing in Solidity is a copy, and copying the cache is most of what
fusing was for.

And nineteen arguments in one `encodePacked` is past what the stack can reach, as is holding
the result, the residual stream, both caches and the widths at once. Both split into two.

The check that mattered throughout: 124 tests, including a whole model generating the tokens
float64 produces. Counting them passing rather than scanning for failures is the only reason
a change this broad could be made quickly.

## What a fused mixture block needs first

Fusing a dense layer was worth 10.3x, and the same change for a mixture layer is worth more —
Qwen3-30B-A3B is 18.1M gas a position and every bit of it is marshalling. It is blocked on
something structural rather than fiddly.

`Mixture` holds `Vec<Expert>`: every expert, materialised. That is fine for a caller holding
the model in memory, and impossible for a precompile that resolves weights out of chain
state, because **which experts to resolve is not known until the router has run** — and the
router runs inside the layer, after attention. Handing the precompile all of them first would
mean reading 1.2 GB to use 76 MB, which is precisely the cost routing exists to avoid.

So the prerequisite is that a mixture produces an expert on demand — a closure or a trait
resolving from the blob when asked — rather than holding a slice of all of them. That is a
change to `tensor-model`'s API, and it has to be made there rather than worked around in the
precompile, because the whole value of the fused dense block is that it *is* `forward_layer`
rather than a second implementation of it. A fused mixture that reimplemented routing would
give that up.

Noted in `Mixture`'s own doc comment, where whoever does it will be looking.

## Real generation: a prompt in, text out

Until now the chain replayed positions an exporter had baked in. It now generates: given
token ids, it feeds the model its own output and keeps going.

```
PROMPT_IDS=9707,11 GENERATE=3 tensor-cli models/full

position 2: token  2776   3 transactions   102,551,651 gas
position 3: token  4460   3 transactions   102,927,017 gas
position 4: token   311   3 transactions   103,311,623 gas

prompt      [9707, 11]          "Hello,"
generated   [358, 2776, 4460, 311]   " I'm trying to"
```

Qwen3-0.6B, all 1.19 GB of it on chain, continuing a prompt in English. Every token is a
transaction; the gas is on the receipt.

Two details make it real rather than staged. The first two positions reproduce the baked
run's tokens exactly — 21806 and 358 — because the prompt begins with the same two ids, and
then the sequences diverge at the first position where the model is fed its own output
instead of a scripted one. And the embedding for each generated token is read from the
classifier table, which for a model with tied weights *is* the embedding table and is
already on chain. Dequantising a row is `q << (32 - scale)`, exact, and it agrees with what
the exporter wrote for the prompt to within 9 elements of 1024 at 1.9e-6 — the difference
being that the exporter quantised from the original floats while this reads what the chain
actually holds.

A row lookup is a table read rather than a multiply, which is why the contract takes the row
rather than fetching it — `Generator.feed` was built that way from the start.

### What is left for a prompt box

A tokenizer, and a page. Both are off-chain glue: text to ids with Qwen3's `tokenizer.json`,
ids to text on the way back, and a form that posts to the RPC. The chain needs no changes —
it already does the part that is hard to believe.

The one real constraint is that the rotary tables cover a fixed number of positions, so a
model exported for five cannot generate a hundred. That is an export flag, not a design
limit.

## A prompt box

```sh
cargo run --release -p tensor-node -- 8545 &
python3 tools/chat.py --prompt "The capital of France is" --generate 5
```

```
prompt   "The capital of France is"
  tokens [785, 6722, 315, 9625, 374]

position 5: token 12095   3 transactions   103,203,527 gas
position 6: token    13   3 transactions   103,584,371 gas
...

reply  " Paris. The capital of Italy"
whole  "The capital of France is Paris. The capital of Italy"
```

Qwen3-0.6B, 1.19 GB of contract code, answering a factual question. Each token is a
transaction with a receipt.

Two things are off chain and both are lookups rather than computation: text to token ids,
and ids back to text. Everything between them — twenty-eight layers, attention, the
classifier — happened inside the EVM against weights held in state. There is no path in
`chat.py` that produces a token without a transaction having produced it first: it parses
the tokens out of what the run printed, and a failed run prints nothing to parse.

The export needs to cover as many positions as the conversation uses, because the rotary
tables are precomputed — `models/chat` covers thirty-two. That is an export flag rather than
a design limit, and the error says so when a prompt exceeds it.

## Keeping the chain across restarts

Uploading a model is the expensive part — 1.19 GB and about a minute — and it only ever
needed to happen once per chain. It was happening once per *process*, because this node held
everything in memory and a restart took the contract with it. That is not a property of
chains; it is a property of a node that had never learned to write anything down.

`--state <path>` reads accounts at startup, and `tensor_saveState` writes them when the
client asks — which it does once, after an upload finishes.

```
first prompt   uploads 1.19 GB, saves 1,241 accounts     ~51 s
node killed and restarted, 1,241 accounts restored
next prompt    "Water boils at" → " 100°C"               3.0 s
```

Blocks are deliberately not restored. A chain resuming from saved state starts its numbering
again, which is honest about what the file is: the weights are there, the history is not, and
putting a block number on something that never happened would be worse than admitting the
gap.

### On the thing that looked slow

It was not slow, it was silent. `chat.py` captured the run and printed it at the end, so a
three-second generation and a fifty-second upload looked identical from outside — frozen.
Streaming the output fixed the complaint entirely without touching the chain.

That is worth recording next to the optimisations, which took a layer from 673,643 gas to
51,183. Only one of the two was ever visible to somebody sitting in front of it.

## The model was not the problem

Asked to write code, the chain produced ` and return the head of the` — a continuation of the
question rather than an answer. That looks like a weak model and mostly was not one.

Qwen3 is instruction-tuned and expects its turn markers. Fed raw text it behaves like a base
model and autocompletes. With the template — and an empty `<think>` block, which is how Qwen3
is told to answer directly rather than reason first — the same weights, unchanged, produce:

```
"What is the capital of France?"
  → "The capital of France is **Paris**."

"Write a python function to reverse a linked list."
  → Here's a Python function that reverses a linked list:
    def reverse_linked_list(head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
```

Correct code, from a 0.6B model, one token per three transactions.

Two other things were needed. The export has to cover the whole conversation, because the
rotary tables are precomputed — the template alone costs about twenty tokens, so
`models/chat128` covers a hundred and twenty-eight positions rather than thirty-two. And
generation stops at `<|im_end|>` rather than filling the budget, since anything past it is
the model starting a turn nobody asked for.

Worth recording because it is the second time in a row the visible problem was not where the
engineering had been. The gas work took a layer from 673,643 to 51,183 and nobody watching
would notice; a missing prompt wrapper made the whole thing look broken.
