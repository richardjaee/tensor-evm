// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {FixedPoint} from "./FixedPoint.sol";
import {Ops} from "./Ops.sol";
import {TensorMatmul} from "./TensorMatmul.sol";

/// @notice A Qwen3 transformer block, executed on chain.
///
/// @dev This is the piece that makes the claim true. Everything before it demonstrated
///      that *a matmul* runs on chain; the forward pass itself lived in Rust, beside the
///      chain rather than on it. Here the orchestration is contract code: the layer decides
///      what to multiply, in what order, against weights held in chain state, and calls out
///      only for the arithmetic a precompile accelerates.
///
///      Weights never travel. They live in the code of a contract already deployed, and
///      what crosses the call boundary is an address and an offset — passing them as
///      arguments costs about 31,500 times more than multiplying by them.
///
///      Activations do travel, and that is fine. Arguments to a call within a transaction
///      come from memory, not transaction calldata, so the overhead is a flat 217 gas
///      regardless of width.
library Transformer {
    /// @notice Geometry. `headDim` is deliberately independent of `dim / nHeads` — in
    ///         Qwen3-0.6B it is 128 against 64, so the query and output projections are not
    ///         square, and code assuming otherwise produces fluent nonsense.
    struct Config {
        uint32 dim;
        uint32 hidden;
        uint32 nHeads;
        uint32 nKv;
        uint32 headDim;
        uint32 seqLen;
        /// @dev Q32, from the checkpoint. Qwen3 specifies 1e-6 where Llama specifies 1e-5.
        int256 rmsEps;
    }

    /// @notice Byte offsets of one layer's tensors within the weight contract's code.
    /// @dev Offsets rather than a derived layout, so a model whose tensors are laid out
    ///      differently needs no new code — only a different set of numbers.
    struct Layer {
        uint32 q;
        uint32 k;
        uint32 v;
        uint32 o;
        uint32 gate;
        uint32 up;
        uint32 down;
    }

    /// @notice Norm weights and rotary tables, which are Q32 vectors rather than quantised
    ///         matrices and so are passed directly rather than read from a blob.
    struct Norms {
        int256[] input;
        int256[] qNorm;
        int256[] kNorm;
        int256[] post;
    }

    /// @notice Everything identifying one layer: its geometry, where its tensors live, and
    ///         the norm weights that are vectors rather than matrices.
    /// @dev Bundled because the EVM's stack, not gas, is what limits how a transformer
    ///      block can be written. Passing these separately leaves `_attention` with eight
    ///      parameters and no room for the values it computes, and `--via-ir` would compile
    ///      it at a measured 2.2x the gas.
    struct Block {
        Config c;
        Layer l;
        Norms n;
        address weights;
    }

    /// @notice Precomputed rotary tables, `seqLen * headDim/2` entries each.
    /// @dev Bundled because the EVM's stack is the binding constraint on how a transformer
    ///      block can be written: passing `cos` and `sin` separately costs two slots that
    ///      `_attention` does not have.
    struct RopeTable {
        int256[] cos;
        int256[] sin;
    }

    /// @notice Keys and values for every position seen so far, for one layer.
    /// @dev Flat, indexed `[pos][kvWidth]`. This is the state that grows with generation
    ///      and dominates everything else at long context.
    struct Kv {
        int256[] k;
        int256[] v;
    }

    function qWidth(Config memory c) internal pure returns (uint256) {
        return uint256(c.nHeads) * c.headDim;
    }

    function kvWidth(Config memory c) internal pure returns (uint256) {
        return uint256(c.nKv) * c.headDim;
    }

    /// @notice One transformer block, applied to the residual stream
    /// @param b   Geometry, tensor offsets, norm weights, and the weight contract
    /// @param pos Position in the sequence; the causal mask and the cache follow from it
    /// @param x   Residual stream in, Q32
    /// @return    The residual stream after the block
    ///
    /// @dev Split into two halves not for tidiness but because the EVM's stack cannot hold
    ///      a whole block at once. Nine parameters and a dozen live values exhaust it, and
    ///      `--via-ir` would compile it at a measured 2.2x the gas.
    function forwardLayer(
        Block memory b,
        Kv memory kv,
        RopeTable memory rope,
        uint256 pos,
        int256[] memory x
    ) internal view returns (int256[] memory) {
        _attention(b, kv, rope, pos, x);
        _feedForward(b, x);
        return x;
    }

    /// @dev Writes its projection back into `x`, so the residual add happens here rather
    ///      than returning a vector the caller has to fold in.
    function _attention(
        Block memory b,
        Kv memory kv,
        RopeTable memory rope,
        uint256 pos,
        int256[] memory x
    ) private view {
        Config memory c = b.c;
        int256[] memory normed = Ops.rmsnorm(x, b.n.input, c.rmsEps);
        _extendCache(b, kv, rope, pos, normed);

        int256[] memory q =
            TensorMatmul.matmul(uint32(qWidth(c)), c.dim, b.weights, b.l.q, normed);
        // QK-norm before rotary, not after. Normalising a rotated vector rescales its
        // components unequally and destroys the angle the embedding encodes.
        _normAndRotate(q, c.nHeads, c.headDim, b.n.qNorm, rope, pos, c.rmsEps);

        int256[] memory proj =
            TensorMatmul.matmul(c.dim, uint32(qWidth(c)), b.weights, b.l.o, _attend(c, kv, q, pos));
        for (uint256 i = 0; i < c.dim; ++i) {
            x[i] += proj[i];
        }
    }

    /// @dev Keys and values, normalised, rotated, and appended to the cache. Its own
    ///      function because holding `k` and `v` alive alongside the rest of a block
    ///      exhausts the EVM's stack.
    function _extendCache(
        Block memory b,
        Kv memory kv,
        RopeTable memory rope,
        uint256 pos,
        int256[] memory normed
    ) private view {
        uint32 kvw = uint32(kvWidth(b.c));
        int256[] memory k = TensorMatmul.matmul(kvw, b.c.dim, b.weights, b.l.k, normed);
        _normAndRotate(k, b.c.nKv, b.c.headDim, b.n.kNorm, rope, pos, b.c.rmsEps);
        _store(kv, kvw, pos, k, TensorMatmul.matmul(kvw, b.c.dim, b.weights, b.l.v, normed));
    }

    function _feedForward(Block memory b, int256[] memory x) private view {
        int256[] memory normed = Ops.rmsnorm(x, b.n.post, b.c.rmsEps);
        int256[] memory gated = TensorMatmul.swiglu(
            TensorMatmul.matmul(b.c.hidden, b.c.dim, b.weights, b.l.gate, normed),
            TensorMatmul.matmul(b.c.hidden, b.c.dim, b.weights, b.l.up, normed)
        );
        int256[] memory proj =
            TensorMatmul.matmul(b.c.dim, b.c.hidden, b.weights, b.l.down, gated);
        for (uint256 i = 0; i < b.c.dim; ++i) {
            x[i] += proj[i];
        }
    }

    /// @dev The per-head slice is normalised into a temporary and copied back, since
    ///      RMSNorm reads every element before writing any.
    function _normAndRotate(
        int256[] memory vec,
        uint256 heads,
        uint256 hs,
        int256[] memory weight,
        RopeTable memory rope,
        uint256 pos,
        int256 eps
    ) private pure {
        uint256 half = hs / 2;
        int256[] memory head = new int256[](hs);
        for (uint256 h = 0; h < heads; ++h) {
            uint256 off = h * hs;
            for (uint256 j = 0; j < hs; ++j) {
                head[j] = vec[off + j];
            }
            int256[] memory nrm = Ops.rmsnorm(head, weight, eps);
            for (uint256 j = 0; j < hs; ++j) {
                vec[off + j] = nrm[j];
            }
            Ops.rope(vec, off, hs, rope.cos, rope.sin, pos * half);
        }
    }

    function _store(Kv memory kv, uint256 kvw, uint256 pos, int256[] memory k, int256[] memory v)
        private
        pure
    {
        uint256 at = pos * kvw;
        for (uint256 i = 0; i < kvw; ++i) {
            kv.k[at + i] = k[i];
            kv.v[at + i] = v[i];
        }
    }

    /// @dev Grouped-query attention over every position up to and including `pos`.
    function _attend(Config memory c, Kv memory kv, int256[] memory q, uint256 pos)
        private
        pure
        returns (int256[] memory attn)
    {
        uint256 hs = c.headDim;
        uint256 kvw = kvWidth(c);
        uint256 group = c.nHeads / c.nKv;
        int256 scale = Ops.invSqrt(hs);

        attn = new int256[](qWidth(c));
        int256[] memory scores = new int256[](pos + 1);

        for (uint256 h = 0; h < c.nHeads; ++h) {
            uint256 kvh = h / group;
            for (uint256 t = 0; t <= pos; ++t) {
                scores[t] = Ops.scaledDot(q, h * hs, kv.k, t * kvw + kvh * hs, hs, scale);
            }
            Ops.softmax(scores, pos + 1);

            for (uint256 t = 0; t <= pos; ++t) {
                uint256 base = t * kvw + kvh * hs;
                int256 w = scores[t];
                for (uint256 j = 0; j < hs; ++j) {
                    attn[h * hs + j] += (w * kv.v[base + j]) >> FixedPoint.F;
                }
            }
        }
    }
}
