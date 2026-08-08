// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

import {Session} from "./Session.sol";
import {Tensor} from "./Tensor.sol";
import {Transformer} from "./Transformer.sol";

/// @notice A resumable forward pass: a token spans as many transactions as it needs.
///
/// @dev One token of Qwen3-0.6B is about 76M gas of computation, and Ethereum caps a
///      transaction at 2^24. So generation cannot be a single call, and the state between
///      calls — the residual stream and the key/value cache — has to live in storage.
///
///      How much of a token to attempt per transaction is the caller's choice rather than
///      a constant. Fewer layers per call means more checkpoints and more gas, and buys
///      finer-grained auditing: every intermediate residual stream becomes a value on
///      chain that anyone can inspect and re-derive, rather than something that existed
///      only inside one transaction's memory. That is a real trade and belongs to whoever
///      is paying for it.
contract Generator {
    struct Meta {
        address owner;
        uint32 pos;
        uint16 layer;
        bool active;
        address weights;
    }

    mapping(bytes32 => Meta) public meta;
    /// @dev Residual stream, four Q32 values a slot.
    mapping(bytes32 => mapping(uint256 => uint256)) private _resid;
    /// @dev Keys and values, indexed `(layer * seqLen + pos) * slots + i`, with values
    ///      offset by a constant so both share one map without a second mapping's overhead.
    mapping(bytes32 => mapping(uint256 => uint256)) private _kv;

    uint256 private constant KV_VALUES = 1 << 128;

    /// @notice Everything about the model a step needs, bundled.
    /// @dev Passed per call rather than stored: these are tens of kilobytes and storing
    ///      them would cost more than the computation. Bundled because six calldata
    ///      parameters leave `step` no stack for the values it computes.
    struct Model {
        Transformer.Config c;
        Transformer.Layer[] layers;
        Transformer.Norms norms;
        Transformer.RopeTable rope;
    }

    event Started(bytes32 indexed id, address indexed owner);
    event Advanced(bytes32 indexed id, uint32 pos, uint16 fromLayer, uint16 toLayer, uint256 gas);
    event Emitted(bytes32 indexed id, uint32 pos);

    error NotOwner();
    error Inactive();
    error Exists();

    /// @notice Begin a generation, seeding the residual stream with a token's embedding
    /// @dev The embedding is supplied rather than looked up: it is a row of a table, not a
    ///      matrix multiply, and the row is the only part of a 311MB table a step reads.
    function start(bytes32 id, address weights, bytes calldata embedding) external {
        if (meta[id].owner != address(0)) revert Exists();
        meta[id] = Meta({
            owner: msg.sender,
            pos: 0,
            layer: 0,
            active: true,
            weights: weights
        });
        Session.store(_resid[id], 0, embedding);
        emit Started(id, msg.sender);
    }

    /// @notice Advance a generation by up to `maxLayers` layers
    /// @param maxLayers How much of a token to attempt here. The granularity knob: smaller
    ///        values cost more in checkpoints and expose more intermediate state on chain.
    /// @return stopped The layer reached, so a caller can size its next call
    function step(bytes32 id, Model calldata w, uint16 maxLayers)
        external
        returns (uint16 stopped)
    {
        Meta memory m = meta[id];
        if (m.owner != msg.sender) revert NotOwner();
        if (!m.active) revert Inactive();

        uint256 g = gasleft();
        bytes memory x = Session.load(_resid[id], 0, w.c.dim);

        uint16 last = m.layer + maxLayers;
        if (last > w.c.nLayers) last = uint16(w.c.nLayers);

        _runLayers(id, w, m, last, x);

        Session.store(_resid[id], 0, x);
        meta[id].layer = last;
        emit Advanced(id, m.pos, m.layer, last, g - gasleft());
        return last;
    }

    /// @dev Its own function so the loop's working values do not share a stack frame with
    ///      the bookkeeping around it.
    function _runLayers(
        bytes32 id,
        Model calldata w,
        Meta memory m,
        uint16 last,
        bytes memory x
    ) private {
        uint256 kvw = Transformer.kvWidth(w.c);
        uint256 cap = Transformer.scratchBytes(w.c, uint256(m.pos + 1) * kvw);
        Tensor.Scratch memory scratch = Tensor.Scratch({buf: new bytes(cap), cap: cap});

        for (uint16 l = m.layer; l < last; ++l) {
            Transformer.Kv memory kv = _cacheFor(id, w.c, l, m.pos, kvw);
            Transformer.forwardLayer(
                Transformer.Block({
                    c: w.c,
                    l: w.layers[l],
                    n: w.norms,
                    weights: m.weights,
                    s: scratch,
                    p: Transformer.Profile({gas: new uint256[](0), at: 0})
                }),
                kv,
                w.rope,
                m.pos,
                x
            );
            _writeCache(id, w.c, l, m.pos, kvw, kv);
        }
    }

    /// @dev Every position up to and including this one, for one layer. Read whole because
    ///      attention needs the whole thing, and it is the argument that grows.
    function _cacheFor(
        bytes32 id,
        Transformer.Config calldata c,
        uint16 layer,
        uint32 pos,
        uint256 kvw
    ) private view returns (Transformer.Kv memory kv) {
        uint256 depth = (uint256(pos) + 1) * kvw;
        uint256 base = (uint256(layer) * c.seqLen) * Session.slotsFor(kvw);
        kv = Transformer.Kv({
            k: Session.load(_kv[id], base, depth),
            v: Session.load(_kv[id], KV_VALUES + base, depth)
        });
    }

    /// @dev Only this position's slice is written back. Earlier positions are unchanged, and
    ///      rewriting them would pay the storage cost of the whole cache every layer.
    function _writeCache(
        bytes32 id,
        Transformer.Config calldata c,
        uint16 layer,
        uint32 pos,
        uint256 kvw,
        Transformer.Kv memory kv
    ) private {
        uint256 slots = Session.slotsFor(kvw);
        uint256 base = (uint256(layer) * c.seqLen) * slots + uint256(pos) * slots;
        bytes memory k = _tail(kv.k, pos, kvw);
        bytes memory v = _tail(kv.v, pos, kvw);
        Session.store(_kv[id], base, k);
        Session.store(_kv[id], KV_VALUES + base, v);
    }

    function _tail(bytes memory a, uint32 pos, uint256 kvw) private pure returns (bytes memory out) {
        out = Tensor.alloc(kvw);
        uint256 from = uint256(pos) * kvw * 8;
        assembly ("memory-safe") {
            mcopy(add(out, 0x20), add(add(a, 0x20), from), mul(kvw, 8))
        }
    }

    /// @notice The residual stream as it stands, for inspection between steps
    /// @dev The point of checkpointing finely: this is a value anyone can read and
    ///      re-derive, not something that existed only inside a transaction.
    function residual(bytes32 id, uint32 dim) external view returns (int256[] memory) {
        return Tensor.unpack(Session.load(_resid[id], 0, dim));
    }
}
