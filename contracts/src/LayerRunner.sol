// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

import {Tensor} from "./Tensor.sol";
import {Transformer} from "./Transformer.sol";

/// @notice External entry point for one transformer block, so a layer can be called and
///         measured on its own.
///
/// @dev The ABI speaks `int256[]` and the block speaks packed bytes, so this converts at
///      the boundary — once per call rather than once per operation. A real generation loop
///      would convert once per token and keep everything packed in between; converting per
///      operation is what made a layer 99.8% marshalling.
contract LayerRunner {
    /// @notice What a block needs that is not in the weight blob.
    /// @dev The norm weights used to be here. They moved into the blob beside the tensors
    ///      once measurement showed a token was re-sending 523KB of unchanging values every
    ///      call, so a layer is now identified by one address and a set of offsets.
    struct Inputs {
        int256[] cos;
        int256[] sin;
        int256[] x;
    }

    function runLayer(
        Transformer.Config calldata c,
        Transformer.Layer calldata l,
        uint256 pos,
        Inputs calldata i
    ) external view returns (int256[] memory) {
        return Tensor.unpack(_run(c, l, pos, i, new uint256[](0)));
    }

    /// @notice The same block, returning where its gas went
    /// @dev Ten sections, in the order a block performs them.
    function profileLayer(
        Transformer.Config calldata c,
        Transformer.Layer calldata l,
        uint256 pos,
        Inputs calldata i
    ) external view returns (uint256[] memory) {
        // Eleven slots: ten sections, then the memory the layer ended up touching.
        // Memory is priced quadratically and never freed, so total allocation is a cost in
        // its own right — and one that no per-section timing would attribute anywhere.
        // `prof` rather than `gas`, which shadows a Yul builtin.
        uint256[] memory prof = new uint256[](11);
        _run(c, l, pos, i, prof);
        assembly ("memory-safe") {
            // The free memory pointer rather than `msize`, which cannot be used alongside
            // the Yul optimiser because optimisation can change what it reports. For
            // allocation totals the two agree to within a word.
            mstore(add(prof, mul(32, 11)), mload(0x40))
        }
        return prof;
    }

    function _run(
        Transformer.Config calldata c,
        Transformer.Layer calldata l,
        uint256 pos,
        Inputs calldata i,
        uint256[] memory gas
    ) private view returns (bytes memory) {
        // The cache holds exactly the positions this call writes. A generation loop carries
        // it between calls; measuring one block does not need to, and pretending otherwise
        // would hide the cache's cost rather than reveal it.
        uint256 depth = (pos + 1) * Transformer.kvWidth(c);
        Transformer.Kv memory kv =
            Transformer.Kv({k: Tensor.alloc(depth), v: Tensor.alloc(depth)});

        // Sized before the block is assembled, so the expression does not have to live on
        // the stack alongside everything else the constructor needs.
        uint256 cap = Transformer.scratchBytes(c, depth);
        Tensor.Scratch memory scratch = Tensor.Scratch({buf: new bytes(cap), cap: cap});

        return Transformer.forwardLayer(
            Transformer.Block({
                c: c,
                l: l,
                s: scratch,
                p: Transformer.Profile({gas: gas, at: 0})
            }),
            kv,
            Transformer.RopeTable({cos: Tensor.pack(_copy(i.cos)), sin: Tensor.pack(_copy(i.sin))}),
            pos,
            Tensor.pack(_copy(i.x))
        );
    }

    function _copy(int256[] calldata a) private pure returns (int256[] memory out) {
        out = new int256[](a.length);
        for (uint256 j = 0; j < a.length; ++j) {
            out[j] = a[j];
        }
    }
}
