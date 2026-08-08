// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

/// @notice The key/value cache, held as contract code rather than storage slots.
///
/// @dev Storage was measured at 22,100 gas for a previously-zero slot, which puts one
///      token's cache at 316M — sixteen times the computation that produced it. Code is
///      priced differently: 200 gas a byte to deploy, and a slot holds thirty-two bytes,
///      so the same data costs 6,400 gas per slot-equivalent instead of 22,100.
///
///      Reading is where the gap becomes absurd. `SLOAD` is 2,100 gas for a cold slot;
///      `EXTCODECOPY` is 2,600 for the account and then 3 gas per word for as much as you
///      like. A cache that costs 30M to read from storage costs about 45k to read from
///      code.
///
///      The catch is that code is immutable, so a position's slice is written once and
///      never revised. That happens to be exactly how a key/value cache behaves — an entry
///      is produced at one position and read unchanged forever after — so the constraint
///      costs nothing here. It would be fatal for the residual stream, which is rewritten
///      every layer, and that stays in storage.
///
///      This is the same pattern the companion project uses for weights, applied to the
///      state instead.
library KvStore {
    error DeployFailed();

    /// @notice Write `data` as the code of a fresh contract
    /// @dev The init code is the standard SSTORE2 prologue: copy the payload after itself
    ///      into memory and return it, so the payload becomes the deployed code. A leading
    ///      `00` byte is prepended so the code can never be executed as a function —
    ///      `STOP` is the first instruction anything reaching it would run.
    ///      The prologue is twelve bytes and `CODECOPY` reads from offset twelve, which is
    ///      where the trailing `00` sits. Those two numbers have to agree: the usual
    ///      SSTORE2 prologue uses `PUSH4` for the length and copies from fourteen, and
    ///      shortening it to `PUSH2` without moving the offset deploys code that begins two
    ///      bytes early. That failure is invisible until something reads a blob back, which
    ///      at position zero never happens — the cache is written after it is read.
    function write(bytes memory data) internal returns (address ptr) {
        bytes memory init = abi.encodePacked(
            hex"61", uint16(data.length + 1), hex"80600c6000396000f300", data
        );
        assembly ("memory-safe") {
            ptr := create(0, add(init, 0x20), mload(init))
        }
        if (ptr == address(0)) revert DeployFailed();
    }

    /// @notice Read `len` bytes from `ptr`, skipping the leading STOP
    function read(address ptr, uint256 len) internal view returns (bytes memory out) {
        out = new bytes(len);
        assembly ("memory-safe") {
            extcodecopy(ptr, add(out, 0x20), 1, len)
        }
    }

    /// @notice Read a slice, for a cache whose layers are laid out end to end
    function readAt(address ptr, uint256 offset, uint256 len)
        internal
        view
        returns (bytes memory out)
    {
        out = new bytes(len);
        assembly ("memory-safe") {
            extcodecopy(ptr, add(out, 0x20), add(offset, 1), len)
        }
    }

    /// @notice Bytes `write` can hold
    /// @dev EIP-170 caps deployed code at 24,576 bytes. A chain running this raises the
    ///      limit — the weight blobs need it far more than the cache does — but the cap is
    ///      named here so a caller on a stock chain fails at the boundary rather than
    ///      somewhere downstream.
    uint256 internal constant MAX = 24_575;
}
