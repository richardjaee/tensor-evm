// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

/// @notice Uploading a model, one transaction at a time.
///
/// @dev Every measurement so far handed the EVM accounts that already held their weights.
///      That is fine for measuring inference and says nothing about how a model gets on
///      chain, which is its own problem: a layer of Qwen3-0.6B is 31.5MB, and no
///      transaction carries 31.5MB of calldata on any chain worth running.
///
///      So weights arrive in chunks, each becoming its own contract, and a directory lists
///      them. A tensor is then a byte range spanning several chunks, which the precompile
///      stitches back together — the cost of chunking being that the reader must know the
///      layout rather than the writer.
///
///      Chunk and directory are distinguished by the first byte of the deployed code rather
///      than by a magic number in the payload. A blob's first byte is a quantisation scale
///      and can be anything, so any marker inside the data could collide with real weights;
///      a byte the deployer controls cannot.
library WeightStore {
    /// @dev First byte of a chunk's code. Also `STOP`, so nothing that reaches this code
    ///      executes anything.
    uint8 internal constant TAG_CHUNK = 0x00;
    /// @dev First byte of a directory's code.
    uint8 internal constant TAG_DIRECTORY = 0x01;

    /// @notice Largest payload a chunk may hold under EIP-170.
    /// @dev A chain running this raises the limit, but chunking is required regardless —
    ///      not by the code size cap but by what a transaction can carry — so the stock
    ///      value is what the uploader targets.
    uint256 internal constant MAX_CHUNK = 24_000;

    error DeployFailed();
    error ChunkTooLarge(uint256 size);

    /// @notice Deploy `data` as a chunk and return its address
    function storeChunk(bytes memory data) internal returns (address ptr) {
        if (data.length > MAX_CHUNK) revert ChunkTooLarge(data.length);
        ptr = _deploy(TAG_CHUNK, data);
    }

    /// @notice Deploy a directory listing `chunks` in order
    /// @dev Twenty bytes each, concatenated. A tensor's byte range is resolved against them
    ///      by position, so order is the layout and cannot be changed after the fact —
    ///      which is the same immutability that makes weights worth committing to.
    function storeDirectory(address[] memory chunks) internal returns (address ptr) {
        bytes memory body = new bytes(chunks.length * 20);
        for (uint256 i = 0; i < chunks.length; ++i) {
            bytes20 a = bytes20(chunks[i]);
            assembly ("memory-safe") {
                mstore(add(add(body, 0x20), mul(i, 20)), a)
            }
        }
        ptr = _deploy(TAG_DIRECTORY, body);
    }

    /// @dev The standard SSTORE2 prologue with a chosen leading byte. Length field and
    ///      `CODECOPY` offset must agree: the prologue below is twelve bytes and the copy
    ///      starts at twelve, where the tag sits. Shortening one without moving the other
    ///      deploys code that begins early, which nothing detects at write time — a bug
    ///      this project has already shipped once.
    function _deploy(uint8 tag, bytes memory data) private returns (address ptr) {
        bytes memory init =
            abi.encodePacked(hex"61", uint16(data.length + 1), hex"80600c6000396000f3", tag, data);
        assembly ("memory-safe") {
            ptr := create(0, add(init, 0x20), mload(init))
        }
        if (ptr == address(0)) revert DeployFailed();
    }
}

/// @notice Permissionless entry point for uploading weights.
///
/// @dev No ownership and no registry. A chunk is content, not a claim: whoever pays for the
///      gas gets an address back, and what makes a model trustworthy is that a registry
///      commits to its hash afterwards rather than that only one party could write it.
contract Uploader {
    event ChunkStored(address indexed ptr, uint256 size);
    event DirectoryStored(address indexed ptr, uint256 chunks);

    /// @notice Store one chunk of a weight blob
    function store(bytes calldata data) external returns (address ptr) {
        ptr = WeightStore.storeChunk(data);
        emit ChunkStored(ptr, data.length);
    }

    /// @notice Store several chunks in one transaction, for callers batching uploads
    function storeMany(bytes[] calldata parts) external returns (address[] memory ptrs) {
        ptrs = new address[](parts.length);
        for (uint256 i = 0; i < parts.length; ++i) {
            ptrs[i] = WeightStore.storeChunk(parts[i]);
            emit ChunkStored(ptrs[i], parts[i].length);
        }
    }

    /// @notice Publish the directory that turns a pile of chunks into a tensor blob
    function finish(address[] calldata chunks) external returns (address ptr) {
        address[] memory c = new address[](chunks.length);
        for (uint256 i = 0; i < chunks.length; ++i) {
            c[i] = chunks[i];
        }
        ptr = WeightStore.storeDirectory(c);
        emit DirectoryStored(ptr, chunks.length);
    }
}
