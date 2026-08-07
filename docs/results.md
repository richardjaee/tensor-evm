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

## Caveats

- Projections assume weights stream at the measured rate and count nothing for attention,
  softmax, KV cache traffic, or weights that do not fit in memory. They are ceilings a real
  implementation is measured against, not predictions.
- 24 GB of RAM caps local work at roughly 7B parameters in `int16`. Anything larger streams
  from disk and will not reach these figures.
- The gas-to-time conversion uses Ethereum's throughput. A purpose-built chain would set a
  different gas schedule, which is the point — but the EVM figures are what a general chain
  charges today.
