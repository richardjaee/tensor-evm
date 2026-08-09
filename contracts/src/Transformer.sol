// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

import {Tensor} from "./Tensor.sol";

/// @notice A Qwen3 transformer block, executed on chain.
///
/// @dev The orchestration is contract code: this decides what to multiply and in what
///      order, reads weights from another contract's code, and calls out for the
///      arithmetic. What it never does is reshape data. Activations stay in the
///      precompiles' own packed form from the residual stream in to the residual stream
///      out, so one operation's result is already the next one's argument.
///
///      That is not a micro-optimisation. With the arithmetic already native, a block spent
///      99.8% of its gas converting between a Solidity array's 32-byte elements and the
///      wire format's 8 — nine million gas of marshalling around eighteen thousand of
///      computation.
///
///      Weights never travel at all. They live in the code of a deployed contract and what
///      crosses the call boundary is an address and an offset; passing them as arguments
///      costs about 31,500 times more than multiplying by them.
library Transformer {
    /// @dev How an uploaded blob announces its shape, matching `WeightStore`.
    uint8 internal constant TAG_CHUNK = 0x00;
    uint8 internal constant TAG_DIRECTORY = 0x01;

    error BadBlob();

    /// @notice Geometry. `headDim` is independent of `dim / nHeads` — in Qwen3-0.6B it is
    ///         128 against 64, so the query and output projections are not square, and code
    ///         assuming otherwise produces fluent nonsense.
    struct Config {
        uint32 dim;
        uint32 hidden;
        uint32 nHeads;
        uint32 nKv;
        uint32 headDim;
        uint32 seqLen;
        /// @dev Needed once a caller drives more than one block, which is what makes a
        ///      forward pass resumable rather than a single library call.
        uint32 nLayers;
        /// @dev Q32, from the checkpoint. Qwen3 specifies 1e-6 where Llama specifies 1e-5.
        int256 rmsEps;
    }

    /// @notice Where one layer's tensors live: which contract, and at what offsets.
    /// @dev The address is per layer rather than per model because a layer of Qwen3-0.6B is
    ///      31.5MB and twenty-eight of them will not share one deployment under any code
    ///      size limit worth setting. It also means a model can be uploaded incrementally,
    ///      which matters when uploading is itself thousands of transactions.
    struct Layer {
        address at;
        uint32 q;
        uint32 k;
        uint32 v;
        uint32 o;
        uint32 gate;
        uint32 up;
        uint32 down;
        /// @dev Norm weights live in the same blob as the tensors. They were calldata until
        ///      measurement showed twenty-eight layers' worth is 523KB a call — about 21M
        ///      gas of a 116M token, spent re-sending values that never change. They are
        ///      model weights and belong where the rest of the model is.
        uint32 inputNorm;
        uint32 qNorm;
        uint32 kNorm;
        uint32 postNorm;
    }

    /// @notice Rotary tables for this position only, sliced by the caller so that nothing
    ///         downstream needs a notion of position.
    struct RopeTable {
        bytes cos;
        bytes sin;
    }

    /// @notice Keys and values for every position so far, packed.
    /// @dev The state that grows with generation, and what will dominate at long context.
    struct Kv {
        bytes k;
        bytes v;
    }

    /// @notice Everything identifying one layer, bundled because the EVM's stack — not gas
    ///         — is what limits how a block can be written.
    struct Block {
        Config c;
        Layer l;
        /// @dev One buffer, reused for every call's input. Memory is never freed and is
        ///      priced quadratically, so a fresh allocation per call cost 662,544 gas of
        ///      expansion for buffers dead the moment they were read.
        Tensor.Scratch s;
        /// @dev Carried here rather than passed alongside, because a memory struct is a
        ///      reference and a separate parameter is a stack slot the caller does not
        ///      have. The stack, not gas, is what limits how a block can be written.
        Profile p;
    }

    /// @notice Optional gas attribution, recorded as a block runs.
    struct Profile {
        uint256[] gas;
        uint256 at;
    }

    function qWidth(Config memory c) internal pure returns (uint256) {
        return uint256(c.nHeads) * c.headDim;
    }

    function kvWidth(Config memory c) internal pure returns (uint256) {
        return uint256(c.nKv) * c.headDim;
    }

    /// @notice Bytes of reusable scratch a block of this shape needs
    /// @dev Here rather than at the call site so that sizing it does not put four separate
    ///      values on a stack that has no room for them.
    function scratchBytes(Config memory c, uint256 cacheDepth) internal pure returns (uint256) {
        return Tensor.scratchFor(c.dim, c.hidden, qWidth(c), cacheDepth);
    }

    /// @dev Its own function because assembling the argument bundle alongside everything
    ///      `_attention` already holds exhausts the stack.
    function _normRope(
        Block memory b,
        RopeTable memory rope,
        bytes memory vec,
        uint32 heads,
        uint32 offset
    ) private view returns (bytes memory) {
        return Tensor.normRope(
            b.s,
            heads,
            b.c.headDim,
            b.c.rmsEps,
            Tensor.NormRopeArgs({
                weight: _norm(b.l.at, offset, b.c.headDim),
                cos: rope.cos,
                sin: rope.sin,
                vec: vec
            })
        );
    }

    /// @dev A Q32 vector read straight out of contract code. Three gas a word against
    ///      sixteen a byte for calldata, and the values are identical every call.
    ///
    ///      This is the one place a contract reads weights without going through a
    ///      precompile, so it is the one place that has to understand how a blob is stored.
    ///      A blob is either a single chunk — a tag byte then the bytes — or a directory: a
    ///      tag byte then twenty bytes per chunk. Both must be handled here, because a real
    ///      layer has no choice about which it is. Qwen3-0.6B's is 31.5MB against a chunk
    ///      limit of 24,000 bytes, so every layer of every real model is a directory.
    ///
    ///      An earlier version handled only the single-chunk case, and the failure was
    ///      silent: reading a directory returned its address table interpreted as norm
    ///      weights, which is well-formed nonsense. It generated tokens, they were simply
    ///      the wrong ones.
    function _norm(address at, uint32 offset, uint256 n) private view returns (bytes memory out) {
        out = new bytes(n * 8);
        uint256 tag;
        assembly ("memory-safe") {
            // Scratch space at 0x00 is scratch precisely so one byte may be read into it
            // without disturbing the allocator.
            extcodecopy(at, 0x00, 0, 1)
            tag := byte(0, mload(0x00))
        }
        if (tag == TAG_CHUNK) {
            assembly ("memory-safe") {
                extcodecopy(at, add(out, 0x20), add(offset, 1), mul(n, 8))
            }
        } else if (tag == TAG_DIRECTORY) {
            // A separate function rather than a branch, because the walk needs more locals
            // than the compiler can reach from inside a caller this large.
            _spanFromDirectory(at, offset, out);
        } else {
            revert BadBlob();
        }
    }

    /// @dev Fills `out` from a directory, crossing chunk boundaries as needed.
    function _spanFromDirectory(address at, uint256 want, bytes memory out) private view {
        bytes memory dir = at.code;
        uint256 entries = (dir.length - 1) / 20;
        uint256 seen = 0; // payload bytes passed
        uint256 left = out.length;
        uint256 done = 0; // bytes written

        for (uint256 i = 0; i < entries && left > 0; ++i) {
            address chunk;
            uint256 size;
            assembly ("memory-safe") {
                chunk := shr(96, mload(add(add(dir, 0x20), add(1, mul(i, 20)))))
                size := sub(extcodesize(chunk), 1)
            }
            if (want >= seen + size) {
                seen += size;
                continue;
            }
            uint256 start = want - seen;
            uint256 take = size - start;
            if (take > left) take = left;
            assembly ("memory-safe") {
                extcodecopy(chunk, add(add(out, 0x20), done), add(start, 1), take)
            }
            want += take;
            left -= take;
            done += take;
            seen += size;
        }
        // Reading past the end must revert rather than return a short vector, which would
        // be normalised against zeros into plausible garbage.
        if (left != 0) revert BadBlob();
    }

    function _mark(Block memory b, uint256 before) private view {
        if (b.p.gas.length == 0) return;
        b.p.gas[b.p.at] = before - gasleft();
        b.p.at += 1;
    }

    /// @notice One transformer block, applied to the packed residual stream
    /// @param pos Position in the sequence; the causal mask and the cache follow from it
    function forwardLayer(
        Block memory b,
        Kv memory kv,
        RopeTable memory rope,
        uint256 pos,
        bytes memory x
    ) internal view returns (bytes memory) {
        _attention(b, kv, rope, pos, x);
        _feedForward(b, x);
        return x;
    }

    function _attention(
        Block memory b,
        Kv memory kv,
        RopeTable memory rope,
        uint256 pos,
        bytes memory x
    ) private view {
        Config memory c = b.c;

        uint256 g = gasleft();
        bytes memory normed = Tensor.rmsnorm(b.s, c.rmsEps, _norm(b.l.at, b.l.inputNorm, b.c.dim), x);
        _mark(b, g);

        g = gasleft();
        _extendCache(b, kv, rope, pos, normed);
        _mark(b, g);

        g = gasleft();
        bytes memory q = Tensor.matmul(b.s, uint32(qWidth(c)), c.dim, b.l.at, b.l.q, normed);
        _mark(b, g);

        // QK-norm before rotary, not after. Normalising a rotated vector rescales its
        // components unequally and destroys the angle the embedding encodes.
        g = gasleft();
        q = _normRope(b, rope, q, c.nHeads, b.l.qNorm);
        _mark(b, g);

        _attendAndProject(b, kv, q, pos, x);
    }

    /// @dev Split off because the stack cannot hold the query, the cache, the residual
    ///      stream, and the scratch buffer at once. Attention and its output projection
    ///      belong together anyway: nothing else ever consumes the attention result.
    function _attendAndProject(
        Block memory b,
        Kv memory kv,
        bytes memory q,
        uint256 pos,
        bytes memory x
    ) private view {
        uint256 g = gasleft();
        bytes memory attn = Tensor.attention(
            b.s,
            b.c.nHeads,
            b.c.nKv,
            b.c.headDim,
            uint32(pos + 1),
            Tensor.AttnArgs({q: q, k: kv.k, v: kv.v})
        );
        _mark(b, g);

        g = gasleft();
        Tensor.addInto(
            x, Tensor.matmul(b.s, b.c.dim, uint32(qWidth(b.c)), b.l.at, b.l.o, attn)
        );
        _mark(b, g);
    }

    /// @dev Keys and values, normalised, rotated, and appended to the cache.
    ///
    ///      The cache is exactly `pos + 1` positions deep when attention reads it, which is
    ///      what makes the mask causal: the precompile has no notion of position and
    ///      attends to everything it is given.
    function _extendCache(
        Block memory b,
        Kv memory kv,
        RopeTable memory rope,
        uint256 pos,
        bytes memory normed
    ) private view {
        Config memory c = b.c;
        uint32 kvw = uint32(kvWidth(c));

        bytes memory k = Tensor.matmul(b.s, kvw, c.dim, b.l.at, b.l.k, normed);
        k = _normRope(b, rope, k, b.c.nKv, b.l.kNorm);
        bytes memory v = Tensor.matmul(b.s, kvw, c.dim, b.l.at, b.l.v, normed);

        uint256 at = pos * uint256(kvw) * 8;
        bytes memory kc = kv.k;
        bytes memory vc = kv.v;
        assembly ("memory-safe") {
            mcopy(add(add(kc, 0x20), at), add(k, 0x20), mload(k))
            mcopy(add(add(vc, 0x20), at), add(v, 0x20), mload(v))
        }
    }

    function _feedForward(Block memory b, bytes memory x) private view {
        Config memory c = b.c;

        uint256 g = gasleft();
        bytes memory normed = Tensor.rmsnorm(b.s, c.rmsEps, _norm(b.l.at, b.l.postNorm, b.c.dim), x);
        _mark(b, g);

        g = gasleft();
        bytes memory gate = Tensor.matmul(b.s, c.hidden, c.dim, b.l.at, b.l.gate, normed);
        bytes memory up = Tensor.matmul(b.s, c.hidden, c.dim, b.l.at, b.l.up, normed);
        _mark(b, g);

        g = gasleft();
        bytes memory gated = Tensor.swiglu(b.s, gate, up);
        _mark(b, g);

        g = gasleft();
        Tensor.addInto(x, Tensor.matmul(b.s, c.dim, c.hidden, b.l.at, b.l.down, gated));
        _mark(b, g);
    }
}
