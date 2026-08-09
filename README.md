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
- **A small set of vectorised precompiles.** `matmul` and `swiglu`, not one and not a whole
  transformer block. Matmul alone was the original design and measurement rejected it: once
  matrix multiplication is native, a layer's SwiGLU is **1,247x** its matmul, so accelerating
  only matmul relocates the bottleneck rather than removing it. Each precompile stays a
  primitive, so a new architecture adds a sibling rather than forcing a revision.
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

That argument has now been measured rather than asserted. `determinism` hashes the kernel,
the fixed-point primitives and a whole four-layer forward pass, and the digest is identical
on an Apple M4, an Intel Xeon Platinum 8481C and an AMD EPYC 7B13 — two instruction sets,
three vendors, two operating systems, and twelve thread configurations including counts
that do not divide the work evenly:

```sh
cargo run --release -p tensor-model --bin determinism
# OVERALL  0146f1d1cee7ad530188c43169bafdfe260fdc644c018810f8eddc6e18edcc14
```

It needs neither revm nor solc nor a network, so anyone can reproduce it on any machine in
about a minute. What it shows is that the *arithmetic* is portable. It does not yet show
that two nodes agree, which additionally covers block execution and gas accounting — see
[`docs/results.md`](docs/results.md).

## Status

Phase 0 complete and the premise holds. See [`docs/results.md`](docs/results.md) for the
measurements and [`docs/design.md`](docs/design.md) for the plan.

| Phase | Deliverable | State |
|---|---|---|
| 0 | Deterministic kernel + bandwidth measurement | **done** |
| 1 | Precompile, callable from contract bytecode | **done** |
| 2 | Weights read from chain state, not calldata | **done** |
| ~~2.5~~ | ~~Activations in state too~~ | dropped — measured unnecessary |
| 3 | Model registry, session-keyed state, arch flags | **done** |
| 4 | Real weights, verified against float64 | **done** |
| 5 | A real Qwen3 layer, verified against float64 | **done** |
| 6 | Full 28-layer forward pass and logits | **done** |
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
on a laptop.

### Phase 1 headline

The precompile works, and measuring it exposed the constraint that shapes everything after.
Passing weights as calldata costs **31,549× more than multiplying by them** — 5,174,146 gas
of transport against 164 gas of arithmetic. Making the computation 16,000× cheaper buys
nothing while moving its operands costs that much more than performing it.

Sharper still: EIP-7623's calldata floor already exceeds the matmul's price by four orders
of magnitude, so a gas limit low enough to starve the arithmetic is rejected before
execution begins. The computation is not cheap relative to transport — it is unpriceable.

So weights cannot travel as arguments. They must already be on chain, with the caller
passing a reference and the precompile reading from state.

### Largest model verified

**Qwen3-4B**, on chain, every position matching float64 — 176M gas a token against 112M for
Qwen3-0.6B. Six point seven times the parameters for 1.57 times the gas, because arithmetic
was never what a token was paying for.

### Phase 6 headline

The whole of Qwen3-0.6B, in integers, reproduces the model: **top-1 and top-5 both agree
with float64**, logits match to 1.7e-03, and it runs at **21.6 tokens/sec** on a laptop with
every value reproducible bit for bit.

Depth was worth testing rather than assuming — one layer matched to 2e-05, twenty-eight and
a classifier to 1.7e-03, so error compounds with depth by about two orders of magnitude. It
stays well below what changes an answer, and now that is measured rather than hoped.

### Phase 2 headline

Weights now live in contract code and the precompile reads them through the journal. At
256 x 256 that is **93x cheaper and 125x less input** — 55,200 gas against 5,174,310, with
calldata falling from 132,360 bytes to 1,056. The saving grows with model size, because
input now scales with `cols` alone rather than `rows x cols`.

That looked like it moved the bottleneck rather than removing it: the remaining calldata is
the activation vector, which projected to ~147M gas per token at 32B against ~31M for the
arithmetic. The projection was wrong. It priced inner-loop calls as transaction calldata,
but a forward pass never re-enters the transaction between layers — the contract holds the
residual stream in memory and passes it from there, where arguments cost memory instead.

Measured by increment across repeated calls in one transaction: **217 gas of overhead per
call, and flat.** At 32 columns it is 217; at 2048 columns, sixty-four times the activation
width, it is still 217. So transport is about **0.3%** of a token at 32B, not 500%, and
activations stay in memory.

## Layout

```
crates/tensor-kernel/       deterministic integer matmul, plus the bandwidth benchmark
crates/tensor-model/        the transformer, and the cross-machine determinism report
crates/tensor-node/         a development chain carrying the precompiles, over JSON-RPC
crates/tensor-precompile/   the EVM precompile, and its end-to-end tests
contracts/src/              Solidity: the matmul library and the model registry
tools/                      safetensors -> int16 conversion, and reference vectors
testdata/                   real Qwen3-0.6B slices the end-to-end tests run against
docs/design.md              architecture, scaling analysis, non-goals, limitations
docs/results.md             every measurement, with the command that produces it
```

## Running a chain

```sh
cargo run --release -p tensor-node    # JSON-RPC on 127.0.0.1:8545
./demo.sh                             # upload a tensor, multiply by it, over RPC
```

The node seals a block per transaction and answers Ethereum's JSON-RPC, so `cast`, `forge`,
MetaMask and a block explorer work against it unchanged — only the precompile set differs
from any other EVM chain. It raises two of Ethereum's limits, and says so on startup: the
transaction gas cap, because a token exceeds EIP-7825's 2^24, and the contract code size,
because a layer of Qwen3-0.6B is 31.5MB against EIP-170's 24KB.

It is not a client. There is no peer-to-peer, no consensus and no sync, so nobody can join
it — a real deployment means these precompiles inside reth. What it does establish is that a
model runs under an ordinary Ethereum interface, and that what it did is a matter of record
afterwards rather than something that happened inside one process.

## Checking it works

```sh
./verify.sh          # tests, lint, formatting — needs nothing but the repository
./verify.sh --full   # the above, plus the real checkpoint if models/ is populated
```

The first tier proves the whole generation loop — seed, every layer, classifier, argmax —
against what float64 says the model emits. The second runs the same path against the
published Qwen3-0.6B weights.

## Building

```sh
cargo test --release                # everything, including the generation loop end to end
cargo run  --release --bin bench    # bandwidth measurement and per-model projections
```

`cargo test` covers the whole path — seed, every layer, classifier, argmax — against a
model small enough to live in the repository, and asserts the tokens float64 says it emits.
The fixture is tiny but not simplified: `head_dim` independent of `dim / n_heads` so the
projections are not square, grouped attention, per-layer norm weights well above unit
scale, and several positions so the causal mask and the key/value cache are exercised.
Every one of those has caused a bug here.

The real checkpoint runs as examples, since 1.2 GB cannot be committed:

```sh
python3 tools/export_model.py --model models/qwen3-0.6b.safetensors \
    --config models/config.json --out models/full
cargo run --release --example full_token -- models/full   # real Qwen3-0.6B, real tokens
cargo run --release --example onchain_layer -- models/layer0   # one block, gas attributed
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
