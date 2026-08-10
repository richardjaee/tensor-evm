# Design

## Thesis

Large language model inference can run entirely on chain, re-executed by every validator,
with no off-chain prover and no trusted operator — if the chain treats matrix
multiplication as a first-class operation rather than as something to be simulated in a
bytecode interpreter.

The claim this project sets out to test:

> A frontier-scale open-weights model can be served by a public chain where every node
> independently recomputes every token, at a cost any individual can afford to pay.

If that holds, verifiable inference does not require zero-knowledge proofs, trusted
execution environments, or an honest-majority committee. It requires ordinary consensus
applied to a VM that can do arithmetic at memory speed.

## Where this comes from

A companion project ([`on-chain-llm`](../../on-chain-llm)) implements a transformer in
Solidity and runs it on Base Sepolia. It works: weights live in contract storage, the
arithmetic executes in the EVM, and every validator re-executes the forward pass. Two
findings from that work motivate this one.

**The gas ceiling is not the limit.** Row-sharded matmuls, pair-aligned rotary shards, a
running argmax over the classifier, and packed activation lanes removed the per-transaction
ceiling entirely. What remains is `O(dim)`, not `O(params)`. Widening the model barely
moves it.

**The interpreter is the limit.** What replaced gas as the binding constraint is raw
execution speed. The EVM dispatches one 256-bit operation at a time, sequentially, reading
each weight through a Merkle-trie-backed store — to do arithmetic on 16-bit integers. The
gap to native code is roughly four orders of magnitude, and no amount of contract-level
optimisation closes it.

This project moves the inner loop into the chain's VM and keeps everything else.

## What stays the same

The parts of the companion project that carry over unchanged, because they were never the
problem:

- **Integer arithmetic throughout.** `int16` weights with per-row power-of-two scales, Q32
  activations. Chosen originally because the EVM has no floating point.
- **Resumable phase structure.** Even with a fast VM, long generations benefit from
  checkpointing between layers.
- **Weights as chain state.** Not a hash of weights held elsewhere. The actual parameters.
- **Model conversion and tokenisation.** The HuggingFace export path and byte-level BPE
  work are directly reusable.

## The determinism argument

This is the load-bearing property, and it is worth stating precisely because most attempts
at verifiable inference fail here.

A chain works only if every node computes identical bits. Floating point does not give you
that across heterogeneous hardware: reduction order varies with thread count, kernel
selection varies with vendor and driver, and fused multiply-add changes results depending
on whether the compiler emitted it. Two honest validators on different machines produce
different outputs and the chain forks.

Integer arithmetic has none of these problems. Addition is associative and multiplication
is exact, so results do not depend on evaluation order, thread count, instruction
selection, or hardware generation.

The kernel goes one step further and never combines an accumulator across threads at all.
Output rows are partitioned but never split, so each dot product is computed start to
finish by one thread. This makes the result independent of the partitioning itself, not
merely reproducible for a fixed partition. `thread_count_does_not_change_result` in
`crates/tensor-kernel` is the regression test for it; if that test ever fails, nodes with
different core counts would fork.

## Architecture

### Chain

A standalone L1, not a rollup.

The rollup path fails on data availability, not on gas. A rollup must post its data to
Ethereum, and the data here is tens to thousands of gigabytes of weights. Ethereum's blob
throughput is shared across every rollup on the network and priced by a fee market that
rises superlinearly under load. An L1 has no such requirement: weights are ordinary state,
distributed by peer-to-peer sync exactly as every chain already distributes state.

Built on [reth](https://github.com/paradigmxyz/reth), whose node builder is explicitly
designed to be extended and whose EVM (`revm`) is structured as a framework for variants.
Rust also gives direct access to SIMD and work-stealing parallelism without fighting the
language.

### The precompile

One precompile, at a fixed address, with a deliberately narrow signature:

```
matmul(weights_location, activations) -> result
```

Narrow because breadth is a trap. A precompile that ran a whole transformer block would
need revision for every architecture variant — attention biases, QK normalisation, GeGLU
against SwiGLU, sliding windows. A precompile that only multiplies matrices serves all of
them, and serves mixture-of-experts models without modification.

Everything else — the phase machine, KV cache, sampling, tokenisation, session management
— stays in Solidity, where it can be changed without a hard fork.

### Model registry

The companion project bakes a single model's configuration into a contract constructor.
That does not scale to many models. Inverted here:

```
modelId -> (weights_directory, config, arch_flags)
```

One runtime contract serves every registered model. Adding a model means uploading a
weight blob and registering a row — no new inference code, no redeploy.

`arch_flags` is a bitmask covering the ways model families differ: QKV projection biases,
RMS normalisation applied to queries and keys, activation function, sliding-window
attention, logit softcapping, rope scaling. These differences are small, and each one is
silent when wrong — a mismatched flag produces fluent nonsense rather than an error. Every
new family must be validated against reference logits at the smallest available size
before being trusted at scale.

### Session state

All generation state is keyed by session: position, layer, phase, cursor, KV cache,
scratch space, prompt, and output. The companion project holds these as globals, which
permits exactly one generation chain-wide and lets any address destroy another's work by
calling `start`. Session keying fixes both, and is a prerequisite for serving multiple
models concurrently.

### Weight upload

Weights are uploaded in ordinary transactions. On a chain that sets its own gas limit and
gas price, the constraint is block propagation rather than gas: at roughly 100 MB per
transaction, a 64 GB model is about 640 transactions, or twenty minutes.

Genesis allocation is available as a convenience for a chain's initial model, but is not
the mechanism, because a chain that can only gain models at genesis cannot gain models.

## Scaling

A forward pass reads every active weight exactly once and performs two operations with it.
Past a few hundred million parameters the arithmetic is free and the memory system is the
entire story, so the governing number is **bandwidth, not FLOPs**:

```
seconds per token ≈ active_parameters × 2 bytes / achieved GB/s
```

`crates/tensor-kernel`'s benchmark measures the right-hand side directly and projects the
left. Everything downstream is calibrated against that measurement rather than estimated.

### Mixture of experts

Above roughly 100B parameters, MoE stops being an optimisation and becomes the only
tractable design. A dense 1T model reads 2 TB per token; DeepSeek-V3 is 671B total but
activates ~37B, so it reads ~74 GB. That is the difference between minutes and seconds.

The consequence for this design is inverted from the naive reading. MoE is *worse* for
storage — every validator must hold all experts, since routing can select any of them —
and storage is the cheap axis. A 4 TB drive costs a few hundred dollars and Ethereum full
nodes already carry comparable state. MoE is *much better* for bandwidth, which is the
expensive axis. Net: strongly favourable.

Top-k routing also keeps the cost model intact. The quantity of work per token is fixed;
only *which* weights are read varies. Address resolution must therefore stay dynamic —
a static weight layout would foreclose MoE support later.

### Why weights are not sharded across validators

A tempting optimisation: let each node hold a slice of the model, so nobody needs the full
2 TB. It must be refused. The moment no single validator can execute a complete forward
pass, re-execution stops being the verification mechanism and is replaced by committee
attestation — which is the trust structure this project exists to avoid.

Every validator holds every weight. That is the cost of the guarantee, and it is
affordable.

### The real state problem is the KV cache

Weights are written once and read by everyone forever; amortised, they are cheap. The KV
cache is per-session, grows with every generated token, and reaches gigabytes at long
context. That is unbounded state growth driven by usage, which is what actually degrades
chains over time.

Two mitigations belong in the design early, because retrofitting them after sessions exist
is painful: latent attention, which compresses the cache by an order of magnitude, and
explicit session expiry so abandoned generations are reclaimed.

## Non-goals

**Zero-knowledge proofs.** A prover computing off chain and posting a proof is a different
architecture with different trust properties. It is the better answer above the size where
re-execution stops being affordable; this project's purpose is to find out where that is.

**Homomorphic encryption.** Measured in the companion project at roughly four orders of
magnitude above plaintext execution. Also fundamentally opposed to the goal: encrypted
outputs cannot be audited by anyone, including for the failures auditing exists to catch.

**Closed-weights models.** Not a technical limit. Weights that were never published cannot
be placed on chain by any means.

**Being an Ethereum L2.** Data availability costs make the rollup path unworkable at these
sizes, and shared security is not the property being sought.

## Honest limitations

- A single-node devnet demonstrates that verification is *feasible*, not that it is
  *distributed*. The claim is "any party with commodity hardware can independently verify
  this," which is different from "many parties do."
- Bootstrapping an independent validator set is a social and economic problem, and it is
  harder than any engineering task described here.
- The chain is not standard EVM. Contracts depending on the precompile will not run on
  Ethereum, and verifying nodes need this implementation rather than any client.
- Model licences govern redistribution, and chain state cannot be withdrawn. Only
  permissively licensed weights should be published.

## Phases, and where the plan was wrong

| Phase | Deliverable | State |
|---|---|---|
| 0 | Deterministic kernel + bandwidth measurement | done |
| 1 | Precompile callable from Solidity | done — in revm directly, not a reth fork |
| 2 | Weights read from chain state | done |
| 3 | Model registry, session-keyed state, arch flags | done |
| 4 | Real weights, verified against float64 | done |
| 5 | A whole model, on a chain, over JSON-RPC | done — Qwen3-0.6B, then Qwen3-30B-A3B |
| 6 | Mixture of experts | done — router precompile, verified on a trained router |
| 7 | Ordering: peer-to-peer, consensus, sync | a reth node runs the precompiles end to end; two peer, but a follower needs a consensus layer to adopt a head |

Two things about that table are worth keeping rather than tidying away.

**Phase 1 was planned as a reth fork and was not built that way.** Precompiles went into revm
directly and a development node was written around them. That was the right order — it
answered whether a model could run at all before spending weeks on a client — but it means
the thing built is a harness, not a client, and phase 7 is larger than the row suggests.

**The plan had no phase for the storage format, and that is where the bugs were.** Both
silent failures this project shipped were there: a directory read as if it were a chunk, and
a chunk length that wrapped at 65,536. Neither was in the arithmetic, which is what all the
early planning was about. A layer of a real model cannot be one chunk, so every real model
takes the path that no fixture exercised, and it took uploading 1.25 GB to find out.

**What the phases understated.** The measured ceiling turned out not to be gas or arithmetic
but three things nobody planned for: the state trie being rebuilt every block, memory spent
keeping transactions as well as state, and the float64 reference needing four times the
model to exist at once. Each was found by trying to run something real, and each took a day
of the schedule that the phases had allotted to inference.

## What would have to be true for this to be a chain

The measurements say a 30B mixture model fits, runs, and costs 3.79 billion gas a token that
anyone can read off a receipt. They do not say it is a chain. That needs:

- **Ordering.** Nodes here agree because they were handed the same transactions, not because
  they gossiped and settled. This is reth's work and it is the largest piece left.
- **A second implementation.** Three machines running one binary rules out hardware, threads
  and operating system. It does not rule out a bug in that binary.
- **Someone else running it.** The 30B chain was produced twice, on two CPU vendors, and
  agreed. Both runs were mine.
