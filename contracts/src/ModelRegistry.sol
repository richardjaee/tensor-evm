// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

import {Tensor} from "./Tensor.sol";

/// @notice Models available on this chain, and the generations running against them.
///
/// @dev Two things here are corrections to the companion project rather than new ideas.
///
///      That project bakes one model's configuration into a constructor, so every model is
///      a separate deployment carrying its own copy of the inference code. This inverts it:
///      one runtime, many registered models, and adding a model is a transaction rather
///      than a deploy.
///
///      It also holds generation state in globals — one generation chain-wide, and any
///      address may call `start` and destroy another's work. State here is keyed by
///      session and owned, so concurrency and access control fall out of the same change.
contract ModelRegistry {
    /// @dev Ways model families differ. Each is small, and each is silent when wrong: a
    ///      mismatched flag yields fluent nonsense rather than an error, so every new
    ///      family must be checked against reference logits at the smallest size before
    ///      being trusted at scale.
    uint32 internal constant FLAG_QKV_BIAS = 1 << 0; // Qwen2 and kin
    uint32 internal constant FLAG_QK_NORM = 1 << 1; // Qwen3, Gemma2
    uint32 internal constant FLAG_GEGLU = 1 << 2; // GeGLU rather than SwiGLU
    uint32 internal constant FLAG_SLIDING_WINDOW = 1 << 3; // Mistral
    uint32 internal constant FLAG_LOGIT_SOFTCAP = 1 << 4; // Gemma2
    uint32 internal constant FLAG_TIED_EMBEDDINGS = 1 << 5; // classifier shares tok_emb

    struct Config {
        uint16 dim;
        uint16 hidden;
        uint16 nLayers;
        uint16 nHeads;
        uint16 nKv;
        uint16 seqLen;
        uint32 vocab;
        uint32 ropeTheta;
        uint32 archFlags;
    }

    struct Model {
        address weights;
        /// @dev Hash of the weight blob, fixed at registration.
        ///
        ///      This is what makes an audit mean anything. Re-executing a computation
        ///      proves the arithmetic; it says nothing about *which* parameters were used
        ///      unless they are identified. Committing to the hash turns "the model that
        ///      was audited" into a one-call check, and closes the silent-downgrade
        ///      problem where a provider swaps in a cheaper variant undetectably.
        bytes32 weightsHash;
        Config config;
        address registrar;
        bool exists;
    }

    struct Session {
        address owner;
        uint256 modelId;
        uint32 pos;
        uint16 layer;
        uint8 phase;
        bool active;
    }

    mapping(uint256 => Model) private _models;
    mapping(bytes32 => Session) private _sessions;
    uint256 public modelCount;

    event ModelRegistered(uint256 indexed modelId, address weights, bytes32 weightsHash);
    event SessionStarted(bytes32 indexed sessionId, uint256 indexed modelId, address indexed owner);
    event SessionAdvanced(bytes32 indexed sessionId, uint32 pos, uint16 layer, uint8 phase);
    event SessionFinished(bytes32 indexed sessionId, uint32 tokens);

    error ModelExists();
    error NoSuchModel();
    error EmptyWeights();
    error SessionExists();
    error NoSuchSession();
    error NotSessionOwner();
    error SessionInactive();

    /// @notice Register a model. Immutable once written.
    ///
    /// @dev Immutability is the point rather than a simplification. If a registered model
    ///      could be repointed, every hash previously published against it would become a
    ///      claim about something that no longer exists, and the provenance guarantee would
    ///      be worth nothing. A new model takes a new id.
    function registerModel(uint256 modelId, address weights, Config calldata config) external {
        if (_models[modelId].exists) revert ModelExists();
        if (weights == address(0) || weights.code.length == 0) revert EmptyWeights();

        bytes32 h;
        assembly ("memory-safe") {
            h := extcodehash(weights)
        }

        _models[modelId] = Model({
            weights: weights,
            weightsHash: h,
            config: config,
            registrar: msg.sender,
            exists: true
        });
        modelCount++;
        emit ModelRegistered(modelId, weights, h);
    }

    function model(uint256 modelId) external view returns (Model memory m) {
        m = _models[modelId];
        if (!m.exists) revert NoSuchModel();
    }

    /// @notice Confirm a model's weights are still the ones registered.
    ///
    /// @dev Cheap enough to call before relying on an output. Weights held as contract code
    ///      cannot in fact be altered in place, so this guards against the registry
    ///      pointing somewhere the caller did not expect rather than against mutation.
    function verifyWeights(uint256 modelId) external view returns (bool) {
        Model storage m = _models[modelId];
        if (!m.exists) revert NoSuchModel();
        address w = m.weights;
        bytes32 h;
        assembly ("memory-safe") {
            h := extcodehash(w)
        }
        return h == m.weightsHash;
    }

    /// @notice Begin a generation. The caller owns it.
    function startSession(bytes32 sessionId, uint256 modelId) external {
        if (!_models[modelId].exists) revert NoSuchModel();
        if (_sessions[sessionId].owner != address(0)) revert SessionExists();

        _sessions[sessionId] =
            Session({owner: msg.sender, modelId: modelId, pos: 0, layer: 0, phase: 0, active: true});
        emit SessionStarted(sessionId, modelId, msg.sender);
    }

    function session(bytes32 sessionId) external view returns (Session memory s) {
        s = _sessions[sessionId];
        if (s.owner == address(0)) revert NoSuchSession();
    }

    /// @notice Run one projection for a session, reading weights from the registered model.
    ///
    /// @dev The gate that ownership exists for: only the session's owner may advance it. In
    ///      the companion project any address could call `start` and delete an in-flight
    ///      generation, because the state was global and unguarded.
    function project(
        bytes32 sessionId,
        uint32 rows,
        uint32 cols,
        uint32 offset,
        int64[] calldata x
    ) external view returns (int64[] memory) {
        Session storage s = _sessions[sessionId];
        if (s.owner == address(0)) revert NoSuchSession();
        if (s.owner != msg.sender) revert NotSessionOwner();
        if (!s.active) revert SessionInactive();

        return _narrow(
            Tensor.unpack(
                Tensor.matmul(
                    _noScratch(), rows, cols, _models[s.modelId].weights, offset, Tensor.pack(_widen(x))
                )
            )
        );
    }

    /// @notice Matmul against a registered model without a session, for reads and probing.
    function projectStateless(uint256 modelId, uint32 rows, uint32 cols, uint32 offset, int64[] calldata x)
        external
        view
        returns (int64[] memory)
    {
        if (!_models[modelId].exists) revert NoSuchModel();
        return _narrow(
            Tensor.unpack(
                Tensor.matmul(
                    _noScratch(), rows, cols, _models[modelId].weights, offset, Tensor.pack(_widen(x))
                )
            )
        );
    }

    /// @dev A zero-capacity scratch, which makes `Tensor` allocate per call. These entry
    ///      points perform one matmul apiece, so there is nothing to reuse a buffer across
    ///      — reuse pays inside a block, where a dozen calls follow one another.
    function _noScratch() private pure returns (Tensor.Scratch memory) {
        return Tensor.Scratch({buf: "", cap: 0});
    }

    /// @dev The ABI speaks int64 because that is the value's actual width; Solidity memory
    ///      gives every element a full word regardless, so internal code uses int256 and
    ///      converts only at this boundary.
    function _widen(int64[] calldata x) private pure returns (int256[] memory out) {
        out = new int256[](x.length);
        for (uint256 i = 0; i < x.length; ++i) {
            out[i] = x[i];
        }
    }

    function _narrow(int256[] memory x) private pure returns (int64[] memory out) {
        out = new int64[](x.length);
        for (uint256 i = 0; i < x.length; ++i) {
            out[i] = int64(x[i]);
        }
    }

    function advance(bytes32 sessionId, uint32 pos, uint16 layer, uint8 phase) external {
        Session storage s = _sessions[sessionId];
        if (s.owner == address(0)) revert NoSuchSession();
        if (s.owner != msg.sender) revert NotSessionOwner();
        if (!s.active) revert SessionInactive();

        s.pos = pos;
        s.layer = layer;
        s.phase = phase;
        emit SessionAdvanced(sessionId, pos, layer, phase);
    }

    function finish(bytes32 sessionId) external {
        Session storage s = _sessions[sessionId];
        if (s.owner == address(0)) revert NoSuchSession();
        if (s.owner != msg.sender) revert NotSessionOwner();
        s.active = false;
        emit SessionFinished(sessionId, s.pos);
    }
}
