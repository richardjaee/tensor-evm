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
    /// @param weights Contract whose code holds this layer's quantised tensors
    /// @param pos     Position in the sequence; the causal mask and the cache follow from it
    /// @return x      The residual stream after the block
    function forwardLayer(
        Config memory c,
        Layer memory l,
        Norms memory n,
        address weights,
        Kv memory kv,
        int256[] memory cos,
        int256[] memory sin,
        uint256 pos,
        int256[] memory x
    ) internal view returns (int256[] memory) {
        uint256 hs = c.headDim;
        uint256 qw = qWidth(c);
        uint256 kvw = kvWidth(c);

        // ---- attention ----
        int256[] memory normed = Ops.rmsnorm(x, n.input);
        int256[] memory q = TensorMatmul.matmul(uint32(qw), c.dim, weights, l.q, normed);
        int256[] memory k = TensorMatmul.matmul(uint32(kvw), c.dim, weights, l.k, normed);
        int256[] memory v = TensorMatmul.matmul(uint32(kvw), c.dim, weights, l.v, normed);

        // QK-norm before rotary, not after. Normalising a rotated vector rescales its
        // components unequally and destroys the angle the embedding encodes.
        _normAndRotate(q, c.nHeads, hs, n.qNorm, cos, sin, pos);
        _normAndRotate(k, c.nKv, hs, n.kNorm, cos, sin, pos);

        _store(kv, kvw, pos, k, v);

        int256[] memory attn = _attend(c, kv, q, pos);
        int256[] memory proj = TensorMatmul.matmul(c.dim, uint32(qw), weights, l.o, attn);
        for (uint256 i = 0; i < c.dim; ++i) {
            x[i] += proj[i];
        }

        // ---- feed-forward ----
        normed = Ops.rmsnorm(x, n.post);
        int256[] memory g = TensorMatmul.matmul(c.hidden, c.dim, weights, l.gate, normed);
        int256[] memory u = TensorMatmul.matmul(c.hidden, c.dim, weights, l.up, normed);
        proj = TensorMatmul.matmul(c.dim, c.hidden, weights, l.down, TensorMatmul.swiglu(g, u));
        for (uint256 i = 0; i < c.dim; ++i) {
            x[i] += proj[i];
        }
        return x;
    }

    /// @dev Split out because Solidity's stack cannot hold the whole block at once. The
    ///      per-head slice is normalised into a temporary and copied back, since RMSNorm
    ///      reads every element before writing any.
    function _normAndRotate(
        int256[] memory vec,
        uint256 heads,
        uint256 hs,
        int256[] memory weight,
        int256[] memory cos,
        int256[] memory sin,
        uint256 pos
    ) private pure {
        uint256 half = hs / 2;
        int256[] memory head = new int256[](hs);
        for (uint256 h = 0; h < heads; ++h) {
            uint256 off = h * hs;
            for (uint256 j = 0; j < hs; ++j) {
                head[j] = vec[off + j];
            }
            int256[] memory nrm = Ops.rmsnorm(head, weight);
            for (uint256 j = 0; j < hs; ++j) {
                vec[off + j] = nrm[j];
            }
            Ops.rope(vec, off, hs, cos, sin, pos * half);
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
