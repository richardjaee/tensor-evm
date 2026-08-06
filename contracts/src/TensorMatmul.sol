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

    error PrecompileFailed();
    error UnexpectedOutputLength(uint256 got, uint256 want);

    /// @notice out[i] = sum_j W[i][j] * x[j], with W read from `weights` at `offset`
    /// @param rows      Output rows
    /// @param cols      Input width, and the length of `x`
    /// @param weights   Contract whose code holds the tensor
    /// @param offset    Byte offset of the tensor within that code
    /// @param x         Q32 activations
    /// @return out      `rows` Q32 values
    function matmul(uint32 rows, uint32 cols, address weights, uint32 offset, int32[] memory x)
        internal
        view
        returns (int32[] memory out)
    {
        require(x.length == cols, "activations must be cols");

        // Layout: rows | cols | weights | offset | activations, big-endian throughout.
        // Built by hand rather than with abi.encodePacked so the activation loop writes
        // four bytes per element instead of allocating an intermediate per value.
        bytes memory input = new bytes(32 + uint256(cols) * 4);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            mstore(p, or(shl(224, rows), shl(192, cols)))
            // address occupies [8..28); shifting it left by 32 bits places it there when
            // written at p+8 as a full word, so write offset first and overlay the address
            mstore(add(p, 28), shl(224, offset))
            mstore(add(p, 8), shl(96, weights))
        }
        for (uint256 i = 0; i < cols; ++i) {
            uint32 v = uint32(x[i]);
            uint256 at = 32 + i * 4;
            input[at] = bytes1(uint8(v >> 24));
            input[at + 1] = bytes1(uint8(v >> 16));
            input[at + 2] = bytes1(uint8(v >> 8));
            input[at + 3] = bytes1(uint8(v));
        }

        (bool ok, bytes memory ret) = MATMUL.staticcall(input);
        if (!ok) revert PrecompileFailed();
        // A chain lacking the precompile returns success with empty data. Checking the
        // length turns that into a revert instead of a silently wrong forward pass.
        if (ret.length != uint256(rows) * 4) {
            revert UnexpectedOutputLength(ret.length, uint256(rows) * 4);
        }

        out = new int32[](rows);
        for (uint256 i = 0; i < rows; ++i) {
            uint256 at = i * 4;
            out[i] = int32(
                uint32(
                    (uint32(uint8(ret[at])) << 24) | (uint32(uint8(ret[at + 1])) << 16)
                        | (uint32(uint8(ret[at + 2])) << 8) | uint32(uint8(ret[at + 3]))
                )
            );
        }
    }
}
