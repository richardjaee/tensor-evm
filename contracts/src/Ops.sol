// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {FixedPoint} from "./FixedPoint.sol";

/// @notice The transformer operations that are not matrix multiplication and not SwiGLU.
///
/// @dev These stay in the EVM for now, and measurement says they will not stay here long.
///      Once matmul and SwiGLU are precompiles, one Qwen3-0.6B token costs roughly 600k gas
///      of precompiled work against several million of this — `isqrt` alone is 3,905 gas
///      and a layer calls it twenty-six times. Whether that is acceptable is exactly what
///      this library exists to measure rather than estimate.
///
///      Every routine matches `crates/tensor-model/src/ops.rs` bit for bit. Not
///      approximately: a chain and its reference implementation disagreeing by one unit in
///      the last place is two different models, and the divergence would first appear as an
///      occasional different token rather than as an error.
library Ops {
    /// @notice RMSNorm: x / sqrt(mean(x^2) + eps) * weight, Q32 throughout
    /// @dev The sum of squares is Q64 and accumulates over `dim` terms; int256 leaves
    ///      ample headroom. The square root of a Q64 value is Q32, which is exactly the
    ///      scale the divisor needs, so no rescaling is required between the two.
    /// @param eps Q32, from the model's own configuration rather than a constant
    function rmsnorm(int256[] memory x, int256[] memory weight, int256 eps)
        internal
        pure
        returns (int256[] memory out)
    {
        uint256 n = x.length;
        require(weight.length == n, "weight width");

        int256 ss = 0;
        for (uint256 i = 0; i < n; ++i) {
            unchecked {
                ss += x[i] * x[i];
            }
        }
        int256 mean = ss / int256(n) + (eps << FixedPoint.F);
        int256 rms = int256(FixedPoint.isqrt(uint256(mean)));
        if (rms == 0) rms = 1;

        out = new int256[](n);
        for (uint256 i = 0; i < n; ++i) {
            // Shifted into Q64 before dividing, so the quotient lands in Q32 rather than
            // truncating to zero for the small values a normalised vector is made of.
            int256 normed = (x[i] << FixedPoint.F) / rms;
            out[i] = (normed * weight[i]) >> FixedPoint.F;
        }
    }

    /// @notice Softmax over the first `len` entries, in place
    /// @dev The maximum is subtracted first, which is not a numerical nicety here: expQ32
    ///      is only defined for non-positive arguments, so this is what makes each call
    ///      legal.
    function softmax(int256[] memory x, uint256 len) internal pure {
        if (len == 0) return;
        int256 max = x[0];
        for (uint256 i = 1; i < len; ++i) {
            if (x[i] > max) max = x[i];
        }

        uint256 sum = 0;
        uint256[] memory e = new uint256[](len);
        for (uint256 i = 0; i < len; ++i) {
            e[i] = FixedPoint.expQ32(x[i] - max);
            sum += e[i];
        }
        if (sum == 0) {
            // Every term underflowed, which needs a spread beyond what Q32 represents.
            // Uniform is the honest answer; zeros would silently drop the attention row.
            int256 uniform = FixedPoint.ONE / int256(len);
            for (uint256 i = 0; i < len; ++i) {
                x[i] = uniform;
            }
            return;
        }
        for (uint256 i = 0; i < len; ++i) {
            x[i] = int256((e[i] << FixedPoint.F) / sum);
        }
    }

    /// @notice Rotary embedding applied to one head, in place
    /// @dev Pairs are (j, j + hs/2), the layout HuggingFace uses. The companion project
    ///      permutes checkpoints into llama2.c's interleaved order instead; matching the
    ///      source convention avoids the permutation entirely.
    ///
    ///      `cos` and `sin` are precomputed. An integer sine would need its own
    ///      approximation and its own agreement problem between implementations, where a
    ///      table is data — generated once and hashed alongside the weights.
    function rope(int256[] memory v, uint256 off, uint256 hs, int256[] memory cos, int256[] memory sin, uint256 tabOff)
        internal
        pure
    {
        uint256 half = hs / 2;
        for (uint256 j = 0; j < half; ++j) {
            int256 a = v[off + j];
            int256 b = v[off + j + half];
            int256 c = cos[tabOff + j];
            int256 s = sin[tabOff + j];
            v[off + j] = (a * c - b * s) >> FixedPoint.F;
            v[off + j + half] = (a * s + b * c) >> FixedPoint.F;
        }
    }

    /// @notice Scaled dot product of two Q32 slices, returning Q32
    function scaledDot(
        int256[] memory a,
        uint256 aOff,
        int256[] memory b,
        uint256 bOff,
        uint256 n,
        int256 scale
    ) internal pure returns (int256) {
        int256 acc = 0;
        for (uint256 i = 0; i < n; ++i) {
            unchecked {
                acc += a[aOff + i] * b[bOff + i];
            }
        }
        return ((acc >> FixedPoint.F) * scale) >> FixedPoint.F;
    }

    /// @notice 1/sqrt(headDim) in Q32, by integer square root
    /// @dev Derived rather than supplied, so the constant has no float provenance and
    ///      cannot differ between a chain and its reference.
    function invSqrt(uint256 headDim) internal pure returns (int256) {
        uint256 root = FixedPoint.isqrt(headDim << 64); // sqrt(hs) in Q32
        return int256((uint256(FixedPoint.ONE) << 32) / root);
    }
}
