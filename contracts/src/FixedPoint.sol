// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

/// @title FixedPoint
/// @notice Integer-only primitives for transformer inference: floor square root,
///         exponential, and the SiLU activation, all in Q32 fixed point.
/// @dev Q32 (value * 2^32) in an int256 register leaves 224 bits of headroom, so
///      dot-product accumulation cannot overflow at any realistic dimension. No
///      floating point is used, and the exp constant table is derived by an exact
///      integer square-root chain rather than being hardcoded from a float source.
///
///      Algorithms follow Karpathy's llama2.c (MIT); the fixed-point formulation
///      and the constant derivation are original to this project. The Python
///      specification these must match bit for bit is tools/intref.py.
library FixedPoint {
    /// @notice 1.0 in Q32
    int256 internal constant ONE = 1 << 32;

    /// @notice Number of fractional bits
    uint256 internal constant F = 32;

    /// @notice floor(log2(e) * 2^32), converting exp() into a base-2 problem
    int256 internal constant LOG2E = 6196328018;

    /// @notice 1e-5 in Q32 — the RMSNorm epsilon used by llama2.c
    int256 internal constant EPS_LLAMA = 42950;

    /// @notice 1e-6 in Q32 — Qwen3's RMSNorm epsilon
    /// @dev Not a universal constant. Models specify it, and Ops.rmsnorm takes it as an
    ///      argument for that reason: a hardcoded value is where a chain and its reference
    ///      quietly become two different models.
    int256 internal constant EPS_QWEN3 = 4295;

    /// @notice Fractional bits consumed by the exp bit-product
    uint256 internal constant EXP_STEPS = 32;

    error NegativeSqrt();
    error PositiveExp();

    /// @notice Floor integer square root: the largest r with r*r <= a
    /// @dev Newton's method from a power-of-two seed at or above sqrt(a), which is
    ///      monotonically decreasing, so the first non-decreasing step is the answer.
    function isqrt(uint256 a) internal pure returns (uint256 r) {
        if (a < 2) return a;
        uint256 x;
        unchecked {
            // seed = 2^ceil(bits/2) >= sqrt(a)
            uint256 bits = 0;
            uint256 t = a;
            while (t > 0) {
                t >>= 1;
                bits++;
            }
            x = uint256(1) << ((bits + 1) >> 1);
            while (true) {
                uint256 y = (x + a / x) >> 1;
                if (y >= x) return x;
                x = y;
            }
        }
    }

    /// @dev 2^(-2^-(i+1)) in Q64, four uint64 lanes per word. Derived by the
    ///      integer square-root chain in tools/intref.py, so the table has no
    ///      float provenance; embedded rather than recomputed because rebuilding
    ///      it per call costs ~228k gas and it is called hundreds of times a layer.
    uint256 private constant EXP2_0 = 0xf5257d152486cc2ceac0c6e7dd24392ed744fccad69d6af4b504f333f9de6484;
    uint256 private constant EXP2_1 = 0xff4ecb59511ec8a4fe9e115c7b8f884bfd3e0c0cf486c174fa83b2db722a033a;
    uint256 private constant EXP2_2 = 0xfff4e91bff1b8c3dffe9d2b2f7db2755ffd3a751c0f7e10bffa756521c8daed1;
    uint256 private constant EXP2_3 = 0xffff4e8e25879bf9fffe9d1cc60ddab0fffd3a3b7814eb53fffa747ea0040663;
    uint256 private constant EXP2_4 = 0xfffff4e8debe025dffffe9d1bdf703aeffffd3a37dda0312ffffa7470363f450;
    uint256 private constant EXP2_5 = 0xffffff4e8de845adfffffe9d1bd10659fffffd3a37a3f8affffffa746f4fa14f;
    uint256 private constant EXP2_6 = 0xfffffff4e8de80c0ffffffe9d1bd01fbffffffd3a37a05e3ffffffa746f41376;
    uint256 private constant EXP2_7 = 0xffffffff4e8de808fffffffe9d1bd011fffffffd3a37a024fffffffa746f4050;

    /// @notice 2^(-2^-(i+1)) in Q64
    function exp2Const(uint256 i) internal pure returns (uint256) {
        uint256 word;
        uint256 w = i >> 2;
        if (w == 0) word = EXP2_0;
        else if (w == 1) word = EXP2_1;
        else if (w == 2) word = EXP2_2;
        else if (w == 3) word = EXP2_3;
        else if (w == 4) word = EXP2_4;
        else if (w == 5) word = EXP2_5;
        else if (w == 6) word = EXP2_6;
        else word = EXP2_7;
        return uint64(word >> (64 * (i & 3)));
    }

    /// @notice exp(x) for Q32 x <= 0; result Q32 in [0, 2^32]
    /// @dev exp(x) = 2^(x*log2e). The integer part of the exponent becomes a shift;
    ///      the fraction becomes a product of 2^(-2^-i) factors chosen by its bits.
    function expQ32(int256 x) internal pure returns (uint256) {
        if (x > 0) revert PositiveExp();
        if (x == 0) return uint256(ONE);
        uint256 e;
        unchecked {
            e = uint256((-x * LOG2E) >> F); // |x| * log2(e), Q32
        }
        uint256 intPart = e >> F;
        if (intPart >= 64) return 0; // underflows Q32
        uint256 frac = e - (intPart << F);

        uint256 acc = uint256(1) << 64; // 1.0 in Q64
        unchecked {
            for (uint256 i = 0; i < EXP_STEPS; ++i) {
                if ((frac >> (F - 1 - i)) & 1 == 1) {
                    acc = (acc * exp2Const(i)) >> 64;
                }
            }
            return (acc >> intPart) >> 32; // Q64 -> Q32 with the integer shift applied
        }
    }

    /// @notice SiLU: z * sigmoid(z), Q32 in and out
    /// @dev Branches on the sign so expQ32 only ever sees a non-positive argument
    ///      and the sigmoid is computed without cancellation.
    function silu(int256 z) internal pure returns (int256) {
        uint256 sig;
        if (z >= 0) {
            uint256 e = expQ32(-z);
            sig = (uint256(ONE) << F) / (uint256(ONE) + e);
        } else {
            uint256 e = expQ32(z);
            sig = (e << F) / (uint256(ONE) + e);
        }
        return (z * int256(sig)) >> F;
    }

    /// @notice Greedy selection: first maximum wins
    function argmax(int256[] memory v) internal pure returns (uint256 best) {
        int256 hi = v[0];
        for (uint256 i = 1; i < v.length; ++i) {
            if (v[i] > hi) {
                hi = v[i];
                best = i;
            }
        }
    }
}
