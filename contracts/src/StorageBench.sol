// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

import {KvStore} from "./KvStore.sol";
import {Session} from "./Session.sol";

/// @notice Prices the two ways of keeping a key/value cache on chain, side by side.
///
/// @dev Storage slots and contract code hold the same bytes and are charged on completely
///      different schedules, and the gap decides whether a token costs 350M gas or 90M. The
///      comparison is measured here rather than derived from the yellow paper, because the
///      derivation has been wrong in the optimistic direction twice already.
contract StorageBench {
    mapping(uint256 => uint256) private _slots;
    address private _ptr;

    /// @notice Write `data` to storage slots and report the gas
    function benchStorage(bytes calldata data) external returns (uint256 used) {
        bytes memory d = data;
        uint256 g = gasleft();
        Session.store(_slots, 0, d);
        used = g - gasleft();
    }

    /// @notice Read it back and report the gas
    function benchStorageRead(uint256 n) external view returns (uint256 used) {
        uint256 g = gasleft();
        Session.load(_slots, 0, n);
        used = g - gasleft();
    }

    /// @notice Write `data` as contract code and report the gas
    function benchCode(bytes calldata data) external returns (uint256 used) {
        bytes memory d = data;
        uint256 g = gasleft();
        _ptr = KvStore.write(d);
        used = g - gasleft();
    }

    /// @notice Read it back and report the gas
    function benchCodeRead(uint256 len) external view returns (uint256 used) {
        uint256 g = gasleft();
        KvStore.read(_ptr, len);
        used = g - gasleft();
    }
}
