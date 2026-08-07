// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.20;

import {TensorMatmul} from "./TensorMatmul.sol";
import {Transformer} from "./Transformer.sol";

/// @notice External entry point for one transformer block, so a layer can be called and
///         measured on its own.
///
/// @dev A library cannot be called from outside a contract, and the point of this phase is
///      to find out what a layer costs when a *transaction* drives it — including the
///      memory a real forward pass expands into, which a library benchmark would not
///      account for.
contract LayerRunner {
    /// @notice Everything a block needs that is not a quantised matrix.
    /// @dev Bundled rather than passed as eleven parameters, which exhausts the stack.
    ///      `--via-ir` would also compile it; the companion project measured that path at
    ///      2.2x the gas, so restructuring is the cheaper answer.
    struct Inputs {
        int256[] inputNorm;
        int256[] qNorm;
        int256[] kNorm;
        int256[] postNorm;
        int256[] cos;
        int256[] sin;
        int256[] x;
    }

    /// @notice One matmul, for isolating which call in a block fails
    /// @dev Diagnostic. A block issues seven matmuls and a SwiGLU, and a revert from any of
    ///      them surfaces identically at the top; this narrows it to one.
    function probeMatmul(
        uint32 rows,
        uint32 cols,
        address weights,
        uint32 offset,
        int256[] calldata x
    ) external view returns (int256[] memory) {
        int256[] memory a = new int256[](x.length);
        for (uint256 i = 0; i < x.length; ++i) {
            a[i] = x[i];
        }
        return TensorMatmul.matmul(rows, cols, weights, offset, a);
    }

    /// @notice One SwiGLU, for the same reason
    function probeSwiglu(int256[] calldata g, int256[] calldata u)
        external
        view
        returns (int256[] memory)
    {
        int256[] memory a = new int256[](g.length);
        int256[] memory b = new int256[](u.length);
        for (uint256 i = 0; i < g.length; ++i) {
            a[i] = g[i];
            b[i] = u[i];
        }
        return TensorMatmul.swiglu(a, b);
    }

    /// @notice Run one block and return the resulting residual stream
    /// @param weights Contract whose code holds this layer's quantised tensors
    function runLayer(
        Transformer.Config calldata c,
        Transformer.Layer calldata l,
        address weights,
        uint256 pos,
        Inputs calldata i
    ) external view returns (int256[] memory) {
        Transformer.Norms memory n = Transformer.Norms({
            input: i.inputNorm,
            qNorm: i.qNorm,
            kNorm: i.kNorm,
            post: i.postNorm
        });

        // One position at a time, so the cache holds exactly what this call writes. A
        // generation loop carries it between calls; measuring a single block does not need
        // to, and pretending otherwise would hide the cache's cost rather than reveal it.
        uint256 kvw = Transformer.kvWidth(c);
        Transformer.Kv memory kv =
            Transformer.Kv({k: new int256[]((pos + 1) * kvw), v: new int256[]((pos + 1) * kvw)});

        return Transformer.forwardLayer(
            Transformer.Block({c: c, l: l, n: n, weights: weights}),
            kv,
            Transformer.RopeTable({cos: i.cos, sin: i.sin}),
            pos,
            i.x
        );
    }
}
