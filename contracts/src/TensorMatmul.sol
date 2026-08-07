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

    error PrecompileFailed();
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
        bytes memory input = new bytes(32 + uint256(cols) * 8);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            mstore(p, or(shl(224, rows), shl(192, cols)))
            // address occupies [8..28); shifting it left by 32 bits places it there when
            // written at p+8 as a full word, so write offset first and overlay the address
            mstore(add(p, 28), shl(224, offset))
            mstore(add(p, 8), shl(96, weights))
        }
        // Written a word at a time through assembly rather than byte by byte: at these
        // widths the loop runs once per activation and a contract does this on every
        // projection of every layer.
        assembly ("memory-safe") {
            let dst := add(add(input, 0x20), 32)
            let src := add(x, 0x20)
            let n := mload(x)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, mul(i, 8)), shl(192, mload(add(src, shl(5, i)))))
            }
        }

        (bool ok, bytes memory ret) = MATMUL.staticcall(input);
        if (!ok) revert PrecompileFailed();
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

        bytes memory input = new bytes(4 + n * 16);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            mstore(p, shl(224, n))
            let g := add(gate, 0x20)
            let u := add(up, 0x20)
            let dst := add(p, 4)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, mul(i, 8)), shl(192, mload(add(g, shl(5, i)))))
                mstore(add(add(dst, mul(n, 8)), mul(i, 8)), shl(192, mload(add(u, shl(5, i)))))
            }
        }

        (bool ok, bytes memory ret) = SWIGLU.staticcall(input);
        if (!ok) revert PrecompileFailed();
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
}
