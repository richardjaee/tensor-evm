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
    struct Inputs {
        int256[] inputNorm;
        int256[] qNorm;
        int256[] kNorm;
        int256[] postNorm;
        int256[] cos;
        int256[] sin;
        int256[] x;
    }

    function runLayer(
        Transformer.Config calldata c,
        Transformer.Layer calldata l,
        address weights,
        uint256 pos,
        Inputs calldata i
    ) external view returns (int256[] memory) {
        return Tensor.unpack(_run(c, l, weights, pos, i, new uint256[](0)));
    }

    /// @notice The same block, returning where its gas went
    /// @dev Ten sections, in the order a block performs them.
    function profileLayer(
        Transformer.Config calldata c,
        Transformer.Layer calldata l,
        address weights,
        uint256 pos,
        Inputs calldata i
    ) external view returns (uint256[] memory) {
        uint256[] memory gas = new uint256[](10);
        _run(c, l, weights, pos, i, gas);
        return gas;
    }

    function _run(
        Transformer.Config calldata c,
        Transformer.Layer calldata l,
        address weights,
        uint256 pos,
        Inputs calldata i,
        uint256[] memory gas
    ) private view returns (bytes memory) {
        Transformer.Norms memory n = Transformer.Norms({
            input: Tensor.pack(_copy(i.inputNorm)),
            qNorm: Tensor.pack(_copy(i.qNorm)),
            kNorm: Tensor.pack(_copy(i.kNorm)),
            post: Tensor.pack(_copy(i.postNorm))
        });

        // The cache holds exactly the positions this call writes. A generation loop carries
        // it between calls; measuring one block does not need to, and pretending otherwise
        // would hide the cache's cost rather than reveal it.
        uint256 depth = (pos + 1) * Transformer.kvWidth(c);
        Transformer.Kv memory kv =
            Transformer.Kv({k: Tensor.alloc(depth), v: Tensor.alloc(depth)});

        return Transformer.forwardLayer(
            Transformer.Block({
                c: c,
                l: l,
                n: n,
                weights: weights,
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
