// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

import {Tensor} from "./Tensor.sol";

/// @notice Generation state that survives between transactions.
///
/// @dev A token costs more gas than one transaction may spend, so the forward pass has to
///      be resumable — and resuming means writing the residual stream and the key/value
///      cache to storage, then reading them back.
///
///      That is not a detail bolted onto the design; at these sizes it is likely to be the
///      design's largest cost. A Q32 value is eight bytes and a storage slot is
///      thirty-two, so four values pack into a slot and a 1024-wide residual stream is 256
///      slots. Writing a slot that was previously zero costs 22,100 gas, so one checkpoint
///      of the residual stream alone is about 5.7M — against roughly 690k for the layer
///      that produced it.
///
///      The cache is worse and unavoidable. Each layer contributes `kvWidth` values per
///      position, they cannot be recomputed without redoing every earlier position, and
///      they must outlive the transaction because the next token reads them. What that
///      actually costs is measured rather than asserted; see `docs/results.md`.
library Session {
    /// @dev Q32 values per 32-byte slot.
    uint256 internal constant PACK = 4;

    /// @notice Write a packed buffer to storage under `base`
    /// @dev The buffer is already four values to the word, which is exactly a slot, so
    ///      this is a word-for-word copy rather than a repack.
    function store(mapping(uint256 => uint256) storage dst, uint256 base, bytes memory src)
        internal
    {
        uint256 slots = (src.length + 31) / 32;
        for (uint256 i = 0; i < slots; ++i) {
            uint256 word;
            assembly ("memory-safe") {
                word := mload(add(add(src, 0x20), mul(i, 32)))
            }
            dst[base + i] = word;
        }
    }

    /// @notice Read `n` Q32 values from storage into a packed buffer
    function load(mapping(uint256 => uint256) storage src, uint256 base, uint256 n)
        internal
        view
        returns (bytes memory out)
    {
        out = Tensor.alloc(n);
        uint256 slots = (n * 8 + 31) / 32;
        for (uint256 i = 0; i < slots; ++i) {
            uint256 word = src[base + i];
            assembly ("memory-safe") {
                mstore(add(add(out, 0x20), mul(i, 32)), word)
            }
        }
    }

    /// @notice Slots a vector of `n` Q32 values occupies
    function slotsFor(uint256 n) internal pure returns (uint256) {
        return (n + PACK - 1) / PACK;
    }
}
