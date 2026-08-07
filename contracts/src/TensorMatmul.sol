// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

/// @notice Calls the chain's matmul precompile.
///
/// @dev The weights never travel through this library. They live in the code of a
///      contract already on the chain, and what crosses the call boundary is an address
///      and an offset — because passing them as arguments costs about 31,500 times more
///      than multiplying by them, and at any real width the calldata floor puts the
///      arithmetic's price out of reach entirely.
///
///      The activations do travel, and that is fine. Arguments to a call within a
///      transaction come from memory, not from transaction calldata, so the overhead is a
///      flat 217 gas regardless of width — measured identical at 32 and at 2048 columns.
library TensorMatmul {
    /// @dev Matmul reading weights from chain state. Not an Ethereum precompile; a chain
    ///      without it will treat this address as an empty account and silently return
    ///      nothing, which is why `matmul` checks the returned length rather than trusting
    ///      the success flag.
    address internal constant MATMUL = address(0x101);

    /// @dev SwiGLU, at 0x102. Present for the same reason matmul is: with matmul native, a
    ///      layer's 3,072 SiLU calls cost 19.1M gas against 15,360 for its matmuls, so
    ///      accelerating only matmul relocates the bottleneck rather than removing it.
    address internal constant SWIGLU = address(0x102);

    /// @dev Fused QK-norm and rotary, at 0x103. With matmul and SwiGLU already native, a
    ///      real layer spent 43% of its gas here — sixteen heads through a Solidity loop
    ///      that allocates an array apiece.
    address internal constant NORM_ROPE = address(0x103);

    /// @dev Grouped-query attention over a cache, at 0x104, and RMSNorm at 0x105. Both are
    ///      cheap arithmetic that the EVM charges for by the loop rather than by the work:
    ///      after matmul, SwiGLU, and QK-norm went native these were 19% and 14% of a layer.
    address internal constant ATTENTION = address(0x104);
    address internal constant RMSNORM = address(0x105);

    /// @dev Carries the shape, because a block issues eight of these and a bare failure
    ///      says nothing about which one.
    error PrecompileFailed(uint256 rows, uint256 cols, uint256 inputLen, uint256 retLen);

    /// @notice Arguments to `normRope`, bundled.
    /// @dev Field order is load-bearing: the encoder walks these four pointers in sequence,
    ///      and the precompile expects weight, cos, sin, vec in exactly this order.
    struct NormRope {
        int256[] weight;
        int256[] cos;
        int256[] sin;
        int256[] vec;
    }
    error UnexpectedOutputLength(uint256 got, uint256 want);

    /// @notice out[i] = sum_j W[i][j] * x[j], with W read from `weights` at `offset`
    /// @param rows      Output rows
    /// @param cols      Input width, and the length of `x`
    /// @param weights   Contract whose code holds the tensor
    /// @param offset    Byte offset of the tensor within that code
    /// @param x         Q32 activations
    /// @return out      `rows` Q32 values
    ///
    /// @dev Values are Q32 and travel to the precompile as `int64`, but the Solidity type
    ///      is `int256` because a memory array occupies a full word per element either way
    ///      — narrowing the type would save nothing and force a conversion at every
    ///      composition point. The wire format narrows on write and sign-extends on read.
    ///
    ///      `int64` on the wire rather than `int32`: Q32 in 32 bits spans only
    ///      [-0.5, 0.5), and a transformer's activations sit around unit scale, so the
    ///      narrower width would saturate on ordinary values everywhere and silently.
    function matmul(uint32 rows, uint32 cols, address weights, uint32 offset, int256[] memory x)
        internal
        view
        returns (int256[] memory out)
    {
        require(x.length == cols, "activations must be cols");

        // Layout: rows | cols | weights | offset | activations, big-endian throughout.
        // Built by hand rather than with abi.encodePacked so the activation loop writes
        // four bytes per element instead of allocating an intermediate per value.
        uint256 want = 32 + uint256(cols) * 8;
        // Over-allocated by a word, then trimmed. Each activation is written with `mstore`,
        // which stores thirty-two bytes, so writing an eight-byte value at a time zeroes the
        // twenty-four bytes after it. Ascending order means every clobber is repaired by the
        // next write except the last, whose overrun needs somewhere harmless to land — and
        // without the slack that somewhere is whatever the allocator hands out next.
        bytes memory input = new bytes(want + 32);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            // Strictly ascending, because every `mstore` writes thirty-two bytes and so
            // zeroes whatever follows the field being set. Writing `offset` before
            // `weights` puts the address on top of it and sends every matmul to offset
            // zero — which fails silently, since reading the wrong tensor is still a
            // well-formed matmul.
            mstore(p, or(shl(224, rows), shl(192, cols)))
            mstore(add(p, 8), shl(96, weights))
            mstore(add(p, 28), shl(224, offset))

            let dst := add(p, 32)
            let src := add(x, 0x20)
            let n := mload(x)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, mul(i, 8)), shl(192, mload(add(src, shl(5, i)))))
            }
            mstore(input, want)
        }

        (bool ok, bytes memory ret) = MATMUL.staticcall(input);
        if (!ok) revert PrecompileFailed(rows, cols, input.length, ret.length);
        // A chain lacking the precompile returns success with empty data. Checking the
        // length turns that into a revert instead of a silently wrong forward pass.
        if (ret.length != uint256(rows) * 8) {
            revert UnexpectedOutputLength(ret.length, uint256(rows) * 8);
        }

        out = new int256[](rows);
        assembly ("memory-safe") {
            let src := add(ret, 0x20)
            let dst := add(out, 0x20)
            for { let i := 0 } lt(i, rows) { i := add(i, 1) } {
                // Load the eight bytes into the high end of a word, then arithmetic-shift
                // down so the sign extends across the full slot.
                mstore(add(dst, shl(5, i)), sar(192, mload(add(src, mul(i, 8)))))
            }
        }
    }

    /// @notice silu(gate) * up, elementwise, through the precompile
    function swiglu(int256[] memory gate, int256[] memory up)
        internal
        view
        returns (int256[] memory out)
    {
        uint256 n = gate.length;
        require(up.length == n, "gate and up must match");

        uint256 want = 4 + n * 16;
        // Same over-allocate-and-trim as `matmul`, and here the hazard is sharper: the two
        // vectors are adjacent, so an interleaved loop would have the last `gate` write land
        // on the first three `up` values and zero them. Writing `gate` to completion first
        // and `up` second means `up` repairs whatever `gate` overran, and only `up`'s own
        // final overrun needs the slack.
        bytes memory input = new bytes(want + 32);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            mstore(p, shl(224, n))
            let dst := add(p, 4)
            let g := add(gate, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, mul(i, 8)), shl(192, mload(add(g, shl(5, i)))))
            }
            let dst2 := add(dst, mul(n, 8))
            let u := add(up, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst2, mul(i, 8)), shl(192, mload(add(u, shl(5, i)))))
            }
            mstore(input, want)
        }

        (bool ok, bytes memory ret) = SWIGLU.staticcall(input);
        if (!ok) revert PrecompileFailed(0, n, input.length, ret.length);
        if (ret.length != n * 8) revert UnexpectedOutputLength(ret.length, n * 8);

        out = new int256[](n);
        assembly ("memory-safe") {
            let src := add(ret, 0x20)
            let dst := add(out, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, shl(5, i)), sar(192, mload(add(src, mul(i, 8)))))
            }
        }
    }

    /// @notice RMS-normalise each head and rotate it, through the precompile
    /// @dev Fused because the two are always adjacent and always in this order:
    ///      normalising a rotated vector rescales its components unequally and destroys the
    ///      angle the embedding encodes.
    function normRope(uint32 heads, uint32 headDim, int256 eps, NormRope memory a)
        internal
        view
        returns (int256[] memory out)
    {
        uint256 half = headDim / 2;
        uint256 n = uint256(heads) * headDim;
        require(a.weight.length == headDim, "weight width");
        require(a.cos.length == half && a.sin.length == half, "table width");
        require(a.vec.length == n, "vector width");

        uint256 want = 16 + (uint256(headDim) + half + half + n) * 8;
        // Over-allocated and trimmed, as elsewhere: `mstore` writes thirty-two bytes, so
        // the last eight-byte value of the last vector overruns by twenty-four.
        bytes memory input = new bytes(want + 32);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            // Ascending, so no field lands on top of one already written.
            mstore(p, or(shl(224, heads), shl(192, headDim)))
            mstore(add(p, 8), shl(192, eps))
            let dst := add(p, 16)
            for { let s := 0 } lt(s, 4) { s := add(s, 1) } {
                // weight, cos, sin, vec in order; each argument is at a fixed stack slot
                // The struct's four array pointers sit in consecutive words, so the
                // fields are walked rather than named — naming them would put four more
                // live values on a stack that has none to spare.
                let src := mload(add(a, mul(s, 0x20)))
                let len := mload(src)
                let from := add(src, 0x20)
                for { let i := 0 } lt(i, len) { i := add(i, 1) } {
                    mstore(add(dst, mul(i, 8)), shl(192, mload(add(from, shl(5, i)))))
                }
                dst := add(dst, mul(len, 8))
            }
            mstore(input, want)
        }

        (bool ok, bytes memory ret) = NORM_ROPE.staticcall(input);
        if (!ok) revert PrecompileFailed(heads, headDim, input.length, ret.length);
        if (ret.length != n * 8) revert UnexpectedOutputLength(ret.length, n * 8);

        out = new int256[](n);
        assembly ("memory-safe") {
            let src := add(ret, 0x20)
            let dst := add(out, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, shl(5, i)), sar(192, mload(add(src, mul(i, 8)))))
            }
        }
    }

    /// @notice Grouped-query attention over the whole cache, through the precompile
    /// @param positions Cache depth including this one; the causal mask is the caller's
    function attention(
        uint32 heads,
        uint32 nKv,
        uint32 headDim,
        uint32 positions,
        int256[] memory q,
        int256[] memory k,
        int256[] memory v
    ) internal view returns (int256[] memory out) {
        uint256 n = uint256(heads) * headDim;
        uint256 cache = uint256(positions) * nKv * headDim;
        require(q.length == n, "q width");
        require(k.length == cache && v.length == cache, "cache width");

        uint256 want = 16 + (n + cache + cache) * 8;
        bytes memory input = new bytes(want + 32);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            mstore(p, or(or(shl(224, heads), shl(192, nKv)), or(shl(160, headDim), shl(128, positions))))
            let dst := add(p, 16)
            for { let s := 0 } lt(s, 3) { s := add(s, 1) } {
                let src := q
                switch s
                case 1 { src := k }
                case 2 { src := v }
                let len := mload(src)
                let from := add(src, 0x20)
                for { let i := 0 } lt(i, len) { i := add(i, 1) } {
                    mstore(add(dst, mul(i, 8)), shl(192, mload(add(from, shl(5, i)))))
                }
                dst := add(dst, mul(len, 8))
            }
            mstore(input, want)
        }

        (bool ok, bytes memory ret) = ATTENTION.staticcall(input);
        if (!ok) revert PrecompileFailed(heads, positions, input.length, ret.length);
        if (ret.length != n * 8) revert UnexpectedOutputLength(ret.length, n * 8);
        out = _unpack(ret, n);
    }

    /// @notice RMSNorm over one vector, through the precompile
    function rmsnorm(int256 eps, int256[] memory weight, int256[] memory x)
        internal
        view
        returns (int256[] memory out)
    {
        uint256 n = x.length;
        require(weight.length == n, "weight width");

        uint256 want = 12 + n * 16;
        bytes memory input = new bytes(want + 32);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            mstore(p, shl(224, n))
            mstore(add(p, 4), shl(192, eps))
            let dst := add(p, 12)
            let w := add(weight, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, mul(i, 8)), shl(192, mload(add(w, shl(5, i)))))
            }
            let dst2 := add(dst, mul(n, 8))
            let src := add(x, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst2, mul(i, 8)), shl(192, mload(add(src, shl(5, i)))))
            }
            mstore(input, want)
        }

        (bool ok, bytes memory ret) = RMSNORM.staticcall(input);
        if (!ok) revert PrecompileFailed(1, n, input.length, ret.length);
        if (ret.length != n * 8) revert UnexpectedOutputLength(ret.length, n * 8);
        out = _unpack(ret, n);
    }

    /// @dev Eight-byte big-endian values into a Solidity array, sign-extended. Shared,
    ///      because every one of these precompiles returns the same shape.
    function _unpack(bytes memory ret, uint256 n) private pure returns (int256[] memory out) {
        out = new int256[](n);
        assembly ("memory-safe") {
            let src := add(ret, 0x20)
            let dst := add(out, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, shl(5, i)), sar(192, mload(add(src, mul(i, 8)))))
            }
        }
    }
}
