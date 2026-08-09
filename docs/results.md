# Results

Measurements, with the commands that produce them. Every projection elsewhere in this
repository is calibrated against these numbers rather than estimated.

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
OVERALL                 0146f1d1cee7ad530188c43169bafdfe260fdc644c018810f8eddc6e18edcc14
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
