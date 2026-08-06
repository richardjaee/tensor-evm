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

### The next bottleneck is activations

The remaining 55,200 gas is almost entirely the activation vector arriving as calldata.
Projecting to Qwen3-32B (dim 5120, 64 layers, seven matmuls per layer):

| per token | gas |
|---|---|
| matmul (all layers) | ~31M |
| activations as calldata | ~147M |

**Transport still dominates, now by about 5×.** The fix is the same one that worked here:
activations belong in state too. The companion project already keeps the residual stream in
storage between phases, so the contract has them there already — passing a reference rather
than the bytes is a smaller change than it sounds, and it removes calldata from the inner
loop entirely.

That is the immediate next step, ahead of the model registry.

## Caveats

- Projections assume weights stream at the measured rate and count nothing for attention,
  softmax, KV cache traffic, or weights that do not fit in memory. They are ceilings a real
  implementation is measured against, not predictions.
- 24 GB of RAM caps local work at roughly 7B parameters in `int16`. Anything larger streams
  from disk and will not reach these figures.
- The gas-to-time conversion uses Ethereum's throughput. A purpose-built chain would set a
  different gas schedule, which is the point — but the EVM figures are what a general chain
  charges today.
