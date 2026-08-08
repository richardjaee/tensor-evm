// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

import {KvStore} from "./KvStore.sol";
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
    }

    mapping(bytes32 => Meta) public meta;
    /// @dev Residual stream, four Q32 values a slot.
    mapping(bytes32 => mapping(uint256 => uint256)) private _resid;
    /// @dev Keys and values live in contract code, one deployment per layer per position,
    ///      addressed by `layer * seqLen + pos`. Only the pointer is a storage slot.
    ///
    ///      Measured: 8KB of cache costs 5,716,966 gas as storage slots and 1,697,337 as
    ///      code, and reading it back costs 600,539 against 7,378. Code is immutable, which
    ///      would be fatal for the residual stream — rewritten every layer — and costs
    ///      nothing here, because a cache entry is produced once at one position and read
    ///      unchanged forever after.
    ///
    ///      Per layer rather than per position, because a step may cover only some layers
    ///      and code cannot be appended to. The granularity has to match what a
    ///      transaction can finish.
    mapping(bytes32 => mapping(uint256 => address)) private _kvPtr;
    /// @dev What the generation has produced. The point of the whole exercise, and cheap:
    ///      one slot per token against millions for the state that produced it.
    mapping(bytes32 => uint32[]) private tokens;

    /// @notice Everything about the model a step needs, bundled.
    /// @dev Passed per call rather than stored: these are tens of kilobytes and storing
    ///      them would cost more than the computation. Bundled because six calldata
    ///      parameters leave `step` no stack for the values it computes.
    struct Model {
        Transformer.Config c;
        Transformer.Layer[] layers;
        /// @dev One set per layer. Norm weights are not shared — every block has its own
        ///      input and post-attention norms, and its own per-head query and key norms —
        ///      and reusing one set across twenty-eight layers would produce a fluent
        ///      model that is not the one that was published.
        Transformer.Norms[] norms;
        Transformer.RopeTable rope;
    }

    event Started(bytes32 indexed id, address indexed owner);
    event Advanced(bytes32 indexed id, uint32 pos, uint16 fromLayer, uint16 toLayer, uint256 gas);
    event Emitted(bytes32 indexed id, uint32 pos, uint32 token, int256 logit);

    error NotOwner();
    error Inactive();
    error Exists();

    /// @notice Begin a generation, seeding the residual stream with a token's embedding
    /// @dev The embedding is supplied rather than looked up: it is a row of a table, not a
    ///      matrix multiply, and the row is the only part of a 311MB table a step reads.
    function start(bytes32 id, bytes calldata embedding) external {
        if (meta[id].owner != address(0)) revert Exists();
        meta[id] = Meta({owner: msg.sender, pos: 0, layer: 0, active: true});
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
                    n: w.norms[l],
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
        uint256 width = kvw * 8;
        kv = Transformer.Kv({
            k: Tensor.alloc((uint256(pos) + 1) * kvw),
            v: Tensor.alloc((uint256(pos) + 1) * kvw)
        });
        // One deployment per earlier position, read whole. The account access dominates
        // the copy — 2,600 against 3 gas a word — so this grows with context in account
        // touches rather than in bytes.
        for (uint256 t = 0; t <= pos; ++t) {
            address ptr = _kvPtr[id][uint256(layer) * c.seqLen + t];
            if (ptr == address(0)) continue; // this position is being written now
            bytes memory k = kv.k;
            bytes memory v = kv.v;
            uint256 at = t * width;
            assembly ("memory-safe") {
                extcodecopy(ptr, add(add(k, 0x20), at), 1, width)
                extcodecopy(ptr, add(add(v, 0x20), at), add(1, width), width)
            }
        }
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
        // This position's slice only. Keys and values go into one deployment so a read
        // costs one account access rather than two.
        uint256 width = kvw * 8;
        bytes memory blob = new bytes(width * 2);
        uint256 from = uint256(pos) * width;
        bytes memory k = kv.k;
        bytes memory v = kv.v;
        assembly ("memory-safe") {
            mcopy(add(blob, 0x20), add(add(k, 0x20), from), width)
            mcopy(add(add(blob, 0x20), width), add(add(v, 0x20), from), width)
        }
        _kvPtr[id][uint256(layer) * c.seqLen + pos] = KvStore.write(blob);
    }

    /// @notice Finish a position: normalise, classify, and record the token
    /// @dev Separate from `step` because it is the only part that touches the vocabulary,
    ///      and because a caller pacing itself against a gas limit needs to know whether
    ///      the next call is a layer or the classifier.
    function emitToken(
        bytes32 id,
        Transformer.Config calldata c,
        bytes calldata finalNorm,
        address classifier,
        uint32 offset,
        uint32 vocab
    ) external returns (uint32 token, int256 logit) {
        Meta memory m = meta[id];
        if (m.owner != msg.sender) revert NotOwner();
        if (!m.active) revert Inactive();
        require(m.layer == c.nLayers, "layers incomplete");

        bytes memory x = Session.load(_resid[id], 0, c.dim);
        Tensor.Scratch memory s = Tensor.Scratch({buf: "", cap: 0});
        bytes memory normed = Tensor.rmsnorm(s, c.rmsEps, finalNorm, x);
        (token, logit) = Tensor.classify(vocab, c.dim, classifier, offset, normed);

        tokens[id].push(token);
        // Ready for the next position: the cache keeps everything, the layer counter
        // restarts, and the residual stream will be overwritten by the next embedding.
        meta[id].pos = m.pos + 1;
        meta[id].layer = 0;
        emit Emitted(id, m.pos, token, logit);
    }

    /// @notice Seed the next position with its embedding
    /// @dev The embedding is a table lookup rather than a multiply, so the row is supplied.
    ///      Only this row of a 311MB table is read, which is why it is not on chain.
    function feed(bytes32 id, bytes calldata embedding) external {
        Meta memory m = meta[id];
        if (m.owner != msg.sender) revert NotOwner();
        if (!m.active) revert Inactive();
        require(m.layer == 0, "position in progress");
        Session.store(_resid[id], 0, embedding);
    }

    /// @notice Tokens produced so far
    function generated(bytes32 id) external view returns (uint32[] memory) {
        return tokens[id];
    }

    /// @notice The residual stream as it stands, for inspection between steps
    /// @dev The point of checkpointing finely: this is a value anyone can read and
    ///      re-derive, not something that existed only inside a transaction.
    function residual(bytes32 id, uint32 dim) external view returns (int256[] memory) {
        return Tensor.unpack(Session.load(_resid[id], 0, dim));
    }
}
