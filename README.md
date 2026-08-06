# tensor-evm

An EVM with matrix multiplication as a native operation, so that language model inference
can run entirely on chain — re-executed by every validator, with no off-chain prover and
no trusted operator.

## Why

Running a transformer inside the EVM works. A companion project does it today on Base
Sepolia: weights in contract storage, arithmetic in the EVM, every validator recomputing
every token. Gas turned out not to be the ceiling — sharded matmuls reduced the
per-transaction cost to `O(dim)` rather than `O(params)`.

The ceiling is the interpreter. The EVM dispatches one 256-bit operation at a time,
sequentially, fetching each weight through a Merkle-trie-backed store, in order to do
arithmetic on 16-bit integers. Native code is roughly four orders of magnitude faster, and
no contract-level optimisation recovers that.

So: move the inner loop into the VM. Keep everything else on chain, unchanged.

The question this project exists to answer is how far that goes.

> Can a public chain serve a frontier-scale open-weights model, with every node
> independently recomputing every token, at a cost an individual can afford?

If the answer is yes, verifiable inference needs neither zero-knowledge proofs nor trusted
hardware — just ordinary consensus over a VM that can do arithmetic at memory speed.

## Approach

- **A standalone L1, not a rollup.** Rollups must post their data to Ethereum, and the
  data here is tens to thousands of gigabytes of weights. On an L1 weights are ordinary
  state, distributed by the same peer-to-peer sync every chain already uses.
- **One narrow precompile.** `matmul(weights, activations)` and nothing more. A precompile
  that ran whole transformer blocks would need revising for every architecture variant; one
  that only multiplies matrices serves all of them, including mixture-of-experts models.
- **Integer arithmetic throughout.** `int16` weights, Q32 activations. Not a workaround —
  it is the reason consensus is possible at all. See below.
- **Every validator holds every weight.** Sharding the model across nodes would be cheaper
  and would destroy the guarantee, because verification by re-execution requires that a
  single node can re-execute.

## Determinism

This is the property everything rests on.

A chain works only if every node computes identical bits. Floating point does not provide
that across heterogeneous hardware — reduction order varies with thread count, kernel
selection varies with vendor, fused multiply-add changes results depending on what the
compiler emitted. Two honest validators on different machines disagree, and the chain
forks. This is where most attempts at verifiable inference fail.

Integer arithmetic has none of those failure modes. The kernel goes further and never
combines an accumulator across threads: output rows are partitioned but never split, so
each dot product is computed start to finish by a single thread. The result is therefore
independent of the partitioning itself, not merely reproducible for one fixed partition.

`thread_count_does_not_change_result` is the regression test. If it fails, nodes with
different core counts would fork the chain.

## Status

Phase 0 complete and the premise holds. See [`docs/results.md`](docs/results.md) for the
measurements and [`docs/design.md`](docs/design.md) for the plan.

| Phase | Deliverable | State |
|---|---|---|
| 0 | Deterministic kernel + bandwidth measurement | **done** |
| 1 | reth fork, stub precompile callable from Solidity | next |
| 2 | Real precompile against state-resident weights | |
| 3 | Model registry, session-keyed state, arch flags | |
| 4 | Scale run | |
| 5 | Mixture-of-experts support | |

### Phase 0 headline

The EVM charges **63 gas per multiply-accumulate**, flat across three orders of magnitude
of shape — measured against the companion project's already-optimised assembly
implementation, so the figure reflects the interpreter rather than the Solidity. The native
kernel sustains **196 GB/s**, about 72% of the test machine's theoretical peak, or 98
billion multiply-accumulates per second.

| model | EVM chain time/token | native |
|---|---|---|
| SmolLM2-135M | 47 minutes | 0.003 s |
| Qwen3-32B | 7.8 days | 0.33 s |
| DeepSeek-V3 (37B active) | 9.0 days | 0.38 s |

A 32B model goes from just under eight days of chain time per token to a third of a second
on a laptop. Phase 1 is worth building.

## Layout

```
crates/tensor-kernel/   deterministic integer matmul, plus the bandwidth benchmark
docs/design.md          architecture, scaling analysis, non-goals, limitations
```

## Building

```sh
cargo test  --release            # correctness, including the determinism guarantees
cargo run   --release --bin bench   # bandwidth measurement and per-model projections
```

The toolchain is pinned in `rust-toolchain.toml`.

## Scaling, in one equation

A forward pass reads every active weight exactly once and does two operations with it, so
past a few hundred million parameters the arithmetic is free and memory is the whole story:

```
seconds per token ≈ active_parameters × 2 bytes / achieved GB/s
```

`bench` measures the denominator directly. Every projection in this repository is
calibrated against that measurement rather than estimated.

This is also why mixture-of-experts models are favourable here despite their size. A dense
1T model reads 2 TB per token; DeepSeek-V3 is 671B parameters but activates ~37B, reading
~74 GB. Storage grows, and storage is the cheap axis.

## Not in scope

**Zero-knowledge proofs** — a different architecture with different trust properties. The
better answer above the size where re-execution stops being affordable; the purpose here is
to find out where that is.

**Homomorphic encryption** — measured at roughly four orders of magnitude above plaintext,
and opposed to the goal besides. Encrypted output cannot be audited by anyone, including
for the failures auditing exists to catch.

**Closed-weights models** — not a technical limit. Weights that were never published cannot
be put on chain by any means.

## Limitations

A single-node devnet shows verification is *feasible*, not that it is *distributed*. The
claim is that any party with commodity hardware can independently verify — not that many
parties do. Bootstrapping an independent validator set is a social and economic problem,
and it is harder than any engineering task described here.

The chain is not standard EVM. Contracts using the precompile will not run on Ethereum.

## Licence

PolyForm Noncommercial 1.0.0 — see [LICENSE](LICENSE). Copyright (c) 2026 Richard Chung.
