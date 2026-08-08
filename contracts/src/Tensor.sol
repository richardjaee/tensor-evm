// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pragma solidity ^0.8.24;

/// @notice The tensor precompiles, called on packed activations.
///
/// @dev Everything here speaks `bytes` holding big-endian `int64` in Q32 — the precompiles'
///      own wire format — rather than `int256[]`.
///
///      That choice is the whole point of this library. Measuring a layer whose arithmetic
///      was already native gave 18,712 gas of precompile against 9,264,360 gas of block:
///      99.8% of the work was Solidity handing precompiles their arguments. The cause was a
///      representation mismatch, not a missing precompile. A Solidity memory array gives
///      every element a full 32-byte word, the wire format is 8 bytes, so every call packed
///      on the way in and unpacked on the way out, one element at a time, forever.
///
///      Keeping activations packed end to end means a layer's data is never reshaped.
///      Assembling a call becomes a header plus `mcopy`, which moves 32 bytes per step
///      instead of 8 and does no shifting — and a precompile's return value is already in
///      the right form to be the next one's argument.
library Tensor {
    address internal constant MATMUL = address(0x101);
    address internal constant SWIGLU = address(0x102);
    address internal constant NORM_ROPE = address(0x103);
    address internal constant ATTENTION = address(0x104);
    address internal constant RMSNORM = address(0x105);

    /// @notice Operand bundles.
    /// @dev Not tidiness: the EVM's stack is what limits how a transformer block can be
    ///      written, and seven loose arguments leave a caller no room for the values it
    ///      computes. Field order is load-bearing — the builder walks these pointers in
    ///      sequence and the precompiles expect exactly this order.
    struct NormRopeArgs {
        bytes weight;
        bytes cos;
        bytes sin;
        bytes vec;
    }

    struct AttnArgs {
        bytes q;
        bytes k;
        bytes v;
    }

    /// @notice A buffer reused for every call's input.
    ///
    /// @dev Memory in the EVM is never freed and is priced quadratically, so what matters
    ///      is not how much is live but how much has ever been allocated. A layer that
    ///      allocated a fresh input buffer per call touched 565KB and paid 662,544 gas for
    ///      it — sixty percent of the block, for buffers that were dead the instant the
    ///      call returned.
    ///
    ///      Allocated once at the largest size any call needs, then resized downward per
    ///      use. The alias is deliberate and the constraint that comes with it is real: the
    ///      value returned by `_build` is only valid until the next `_build`, which holds
    ///      because `_call` consumes it immediately.
    struct Scratch {
        bytes buf;
        /// @dev Capacity, tracked separately because resizing writes the length field and
        ///      would otherwise shrink the buffer permanently — each call would then be
        ///      bounded by whatever the previous one happened to need.
        uint256 cap;
    }

    /// @notice Bytes of scratch a layer of this shape needs.
    /// @dev The largest input any single call assembles, which is SwiGLU's: a four-byte
    ///      header and two vectors of `hidden` values.
    function scratchFor(uint256 dim, uint256 hidden, uint256 qw, uint256 cacheDepth)
        internal
        pure
        returns (uint256 n)
    {
        n = 4 + hidden * 2 * W; // swiglu
        uint256 attn = 16 + (qw + cacheDepth * 2) * W;
        if (attn > n) n = attn;
        uint256 mm = 32 + (hidden > dim ? hidden : dim) * W;
        if (mm > n) n = mm;
    }

    error PrecompileFailed(address at, uint256 inputLen);
    error UnexpectedOutputLength(address at, uint256 got, uint256 want);

    /// @dev Q32 values are 8 bytes on the wire.
    uint256 private constant W = 8;

    /// @notice Number of Q32 values in a packed buffer
    function len(bytes memory a) internal pure returns (uint256) {
        return a.length / W;
    }

    /// @notice Read one Q32 value, sign-extended
    function at(bytes memory a, uint256 i) internal pure returns (int256 v) {
        assembly ("memory-safe") {
            // Load the eight bytes into the top of a word, then shift down arithmetically
            // so the sign extends across the full slot.
            v := sar(192, mload(add(add(a, 0x20), mul(i, 8))))
        }
    }

    /// @notice Allocate a packed buffer of `n` values
    function alloc(uint256 n) internal pure returns (bytes memory) {
        return new bytes(n * W);
    }

    /// @notice Pack an `int256[]` once, at the edge of the system
    /// @dev Called where a caller's ABI meets this library, not between operations.
    function pack(int256[] memory v) internal pure returns (bytes memory out) {
        uint256 n = v.length;
        out = new bytes(n * W + 32);
        assembly ("memory-safe") {
            let dst := add(out, 0x20)
            let src := add(v, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, mul(i, 8)), shl(192, mload(add(src, shl(5, i)))))
            }
            // Over-allocated by a word and trimmed: `mstore` writes 32 bytes, so the last
            // value's write overruns by 24 and needs somewhere harmless to land.
            mstore(out, mul(n, 8))
        }
    }

    function unpack(bytes memory a) internal pure returns (int256[] memory out) {
        uint256 n = a.length / W;
        out = new int256[](n);
        assembly ("memory-safe") {
            let src := add(a, 0x20)
            let dst := add(out, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                mstore(add(dst, shl(5, i)), sar(192, mload(add(src, mul(i, 8)))))
            }
        }
    }

    /// @notice `dst[i] += src[i]`, elementwise, in place — the residual connection
    /// @dev Still per element, because there is no vector add over packed bytes. At `dim`
    ///      values twice a layer this is the one loop worth leaving in the EVM.
    function addInto(bytes memory dst, bytes memory src) internal pure {
        uint256 n = dst.length / W;
        require(src.length == dst.length, "residual widths must match");
        assembly ("memory-safe") {
            let d := add(dst, 0x20)
            let s := add(src, 0x20)
            for { let i := 0 } lt(i, n) { i := add(i, 1) } {
                let p := add(d, mul(i, 8))
                let sum := add(sar(192, mload(p)), sar(192, mload(add(s, mul(i, 8)))))
                // Write the eight low bytes without disturbing the neighbours: read the
                // surrounding word, clear the field, and put the value back.
                let word := mload(p)
                mstore(p, or(shl(192, and(sum, 0xffffffffffffffff)), shr(64, shl(64, word))))
            }
        }
    }

    /// @dev Calls a precompile and checks the reply is the length expected. A chain without
    ///      these precompiles returns success with empty data, so checking the length turns
    ///      that into a revert rather than a forward pass quietly full of zeros.
    function _call(address to, bytes memory input, uint256 want)
        private
        view
        returns (bytes memory out)
    {
        bool ok;
        (ok, out) = to.staticcall(input);
        if (!ok) revert PrecompileFailed(to, input.length);
        if (out.length != want) revert UnexpectedOutputLength(to, out.length, want);
    }

    /// @dev Header bytes followed by packed payloads, assembled with `mcopy`. This is what
    ///      replaced the per-element packing loops: 32 bytes a step and no shifting.
    function _build(Scratch memory s, uint256 headLen, bytes32 head, bytes[] memory parts)
        private
        pure
        returns (bytes memory input)
    {
        uint256 total = headLen;
        for (uint256 i = 0; i < parts.length; ++i) {
            total += parts[i].length;
        }
        input = s.buf;
        if (s.cap == 0) {
            input = new bytes(total);
        } else {
            require(total <= s.cap, "scratch too small");
            assembly ("memory-safe") {
                mstore(input, total)
            }
        }
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            mstore(p, head)
            let dst := add(p, headLen)
            for { let i := 0 } lt(i, mload(parts)) { i := add(i, 1) } {
                let part := mload(add(add(parts, 0x20), shl(5, i)))
                let n := mload(part)
                mcopy(dst, add(part, 0x20), n)
                dst := add(dst, n)
            }
        }
    }

    function one(bytes memory a) private pure returns (bytes[] memory p) {
        p = new bytes[](1);
        p[0] = a;
    }

    function two(bytes memory a, bytes memory b) private pure returns (bytes[] memory p) {
        p = new bytes[](2);
        p[0] = a;
        p[1] = b;
    }

    /// @notice out[i] = sum_j W[i][j] * x[j], with W read from chain state
    function matmul(
        Scratch memory s,
        uint32 rows,
        uint32 cols,
        address weights,
        uint32 offset,
        bytes memory x
    )
        internal
        view
        returns (bytes memory)
    {
        require(x.length == uint256(cols) * W, "activation width");
        bytes32 head = bytes32(
            (uint256(rows) << 224) | (uint256(cols) << 192) | (uint256(uint160(weights)) << 32)
                | uint256(offset)
        );
        return _call(MATMUL, _build(s, 32, head, one(x)), uint256(rows) * W);
    }

    /// @notice silu(gate) * up, elementwise
    function swiglu(Scratch memory s, bytes memory gate, bytes memory up)
        internal
        view
        returns (bytes memory)
    {
        uint256 n = gate.length / W;
        require(up.length == gate.length, "gate and up must match");
        return _call(SWIGLU, _build(s, 4, bytes32(n << 224), two(gate, up)), n * W);
    }

    /// @notice RMS-normalise each head and rotate it
    function normRope(
        Scratch memory s,
        uint32 heads,
        uint32 headDim,
        int256 eps,
        NormRopeArgs memory a
    )
        internal
        view
        returns (bytes memory)
    {
        bytes[] memory parts = new bytes[](4);
        parts[0] = a.weight;
        parts[1] = a.cos;
        parts[2] = a.sin;
        parts[3] = a.vec;
        bytes32 head = bytes32(
            (uint256(heads) << 224) | (uint256(headDim) << 192)
                | (uint256(uint64(int64(eps))) << 128)
        );
        return _call(NORM_ROPE, _build(s, 16, head, parts), uint256(heads) * headDim * W);
    }

    /// @notice Grouped-query attention over the whole cache
    function attention(
        Scratch memory s,
        uint32 heads,
        uint32 nKv,
        uint32 headDim,
        uint32 positions,
        AttnArgs memory a
    ) internal view returns (bytes memory) {
        bytes[] memory parts = new bytes[](3);
        parts[0] = a.q;
        parts[1] = a.k;
        parts[2] = a.v;
        bytes32 head = bytes32(
            (uint256(heads) << 224) | (uint256(nKv) << 192) | (uint256(headDim) << 160)
                | (uint256(positions) << 128)
        );
        return _call(ATTENTION, _build(s, 16, head, parts), uint256(heads) * headDim * W);
    }

    /// @notice RMSNorm over one vector
    function rmsnorm(Scratch memory s, int256 eps, bytes memory weight, bytes memory x)
        internal
        view
        returns (bytes memory)
    {
        uint256 n = x.length / W;
        require(weight.length == x.length, "weight width");
        bytes32 head = bytes32((n << 224) | (uint256(uint64(int64(eps))) << 160));
        return _call(RMSNORM, _build(s, 12, head, two(weight, x)), n * W);
    }
}
