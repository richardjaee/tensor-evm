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
    /// @dev Classifier fused with its argmax. Fused because the intermediate is enormous
    ///      and nobody wants it: a 151,936-entry logit vector is 1.2MB, and the caller's
    ///      next act is always to discard all but one.
    address internal constant CLASSIFY = address(0x106);

    /// @dev The mixture-of-experts router. In the VM rather than here because selection
    ///      decides which weights a token reads: two nodes that break a tie differently do
    ///      not round differently, they run different parameters and emit different text.
    address internal constant ROUTER = address(0x107);

    /// @dev The whole feed-forward in one call. Three separate ones spend far more carrying
    ///      `hidden`-wide intermediates across call boundaries than the arithmetic in them
    ///      costs — a layer profiled at 2.8% precompile and the rest argument marshalling.
    address internal constant FFN = address(0x108);

    /// @dev Attention and the projection that always follows it. Nothing else consumes an
    ///      attention result, so it never needs to leave the VM — and it is twice the
    ///      residual stream wide, which made `o_proj` the most expensive line in a layer.
    address internal constant ATTN_PROJ = address(0x109);

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
        // Forty-four rather than sixteen: attention is now fused with its output
        // projection, and the wider header carries the weights address and offset. Sizing
        // this for the old call was a "scratch too small" revert on every layer.
        uint256 attn = 44 + (qw + cacheDepth * 2) * W;
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

    /// @notice `dst[i] += (src[i] * weight) >> 32`, elementwise, in place
    ///
    /// @dev How a mixture of experts is summed: each expert's output enters the residual
    ///      stream scaled by its routing weight, which is Q32 and sums to one across the
    ///      chosen experts.
    ///
    ///      Note that this wraps on overflow rather than saturating, matching `addInto`
    ///      above. The Rust reference saturates. Neither has reached a magnitude where the
    ///      two differ — routing weights are at most one, so a scaled expert output is
    ///      smaller than the output itself — but it is a divergence rather than a choice,
    ///      and it should be resolved in one direction rather than left to a magnitude
    ///      nobody has hit yet.
    function addScaled(bytes memory dst, bytes memory src, int256 weight) internal pure {
        require(src.length == dst.length, "residual widths must match");
        uint256 n = dst.length / W;
        for (uint256 i = 0; i < n; ++i) {
            int256 sum;
            assembly ("memory-safe") {
                let p := add(add(dst, 0x20), mul(i, 8))
                sum := sar(192, mload(p))
            }
            int256 v;
            assembly ("memory-safe") {
                v := sar(192, mload(add(add(src, 0x20), mul(i, 8))))
            }
            // Arithmetic shift, so a negative expert output stays negative.
            sum += (v * weight) >> 32;
            assembly ("memory-safe") {
                let p := add(add(dst, 0x20), mul(i, 8))
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
        return _build(s, headLen, head, bytes32(0), parts);
    }

    /// @dev As above, for a header that does not fit in one word.
    ///
    ///      Attention fused with its projection needs forty-four bytes of header, and
    ///      without this it could not use the scratch buffer at all — it allocated a fresh
    ///      one and copied the query and both caches into it, which cost more than the round
    ///      trip the fusion saved. Measured 2.4% worse before this existed.
    function _build(
        Scratch memory s,
        uint256 headLen,
        bytes32 head,
        bytes32 tail,
        bytes[] memory parts
    ) private pure returns (bytes memory input) {
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
            // A second word, for headers past thirty-two bytes. Written before the parts
            // and after the first word, so a caller passing zero here is unaffected.
            if gt(headLen, 32) { mstore(add(p, 32), tail) }
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

    /// @notice The classifier and its argmax, in one call
    /// @return token The winning row
    /// @return logit Its value, so a caller can judge how close the decision was
    function classify(uint32 rows, uint32 cols, address weights, uint32 offset, bytes memory x)
        internal
        view
        returns (uint32 token, int256 logit)
    {
        require(x.length == uint256(cols) * W, "activation width");
        bytes memory input = new bytes(32 + x.length);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            // Ascending, so no field lands on top of one already written.
            mstore(p, or(shl(224, rows), shl(192, cols)))
            mstore(add(p, 8), shl(96, weights))
            mstore(add(p, 28), shl(224, offset))
            mcopy(add(p, 32), add(x, 0x20), mload(x))
        }

        bytes memory ret = _call(CLASSIFY, input, 12);
        assembly ("memory-safe") {
            let word := mload(add(ret, 0x20))
            token := shr(224, word)
            // The logit occupies bytes 4..12; shift it to the top of a word, then down
            // arithmetically so the sign extends.
            logit := sar(192, shl(32, word))
        }
    }

    /// @notice Attention and its output projection, in one call
    function attnProj(
        Scratch memory s,
        AttnShape memory shape,
        address weights,
        uint32 offset,
        AttnArgs memory a
    ) internal view returns (bytes memory) {
        bytes[] memory parts = new bytes[](3);
        parts[0] = a.q;
        parts[1] = a.k;
        parts[2] = a.v;
        bytes32 head = bytes32(
            (shape.heads << 224) | (shape.nKv << 192) | (shape.headDim << 160)
                | (shape.positions << 128) | (shape.dim << 96)
                | (uint256(uint160(weights)) >> 64)
        );
        // The address straddles the word boundary: its top twelve bytes end the first word,
        // its bottom eight begin the second, with the offset behind them.
        bytes32 tail = bytes32(((uint256(uint160(weights)) & 0xffffffffffffffff) << 192) | (uint256(offset) << 160));
        return _call(ATTN_PROJ, _build(s, 44, head, tail, parts), shape.dim * W);
    }

    /// @dev The five widths an attention call needs, as a struct because the stack cannot
    ///      hold them alongside the three vectors.
    struct AttnShape {
        uint256 heads;
        uint256 nKv;
        uint256 headDim;
        uint256 positions;
        uint256 dim;
    }

    function mload_dim(AttnShape memory shape) private pure returns (uint256 d) {
        d = shape.dim;
    }

    /// @notice Gate, up, SwiGLU and down, in one call
    /// @dev The caller passes `dim` activations and receives `dim` back. Everything of width
    ///      `hidden` exists only inside the VM, where it costs nothing to hold.
    function ffn(
        Scratch memory s,
        uint32 hidden,
        uint32 dim,
        address weights,
        uint32 gateOff,
        uint32 upOff,
        uint32 downOff,
        bytes memory x
    ) internal view returns (bytes memory) {
        require(x.length == uint256(dim) * W, "activation width");
        bytes memory input = new bytes(40 + x.length);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            // Ascending, so no field lands on top of one already written.
            mstore(p, or(shl(224, hidden), shl(192, dim)))
            mstore(add(p, 8), shl(96, weights))
            mstore(add(p, 28), or(shl(224, gateOff), shl(192, upOff)))
            mstore(add(p, 36), shl(224, downOff))
            mcopy(add(p, 40), add(x, 0x20), mload(x))
        }
        return _call(FFN, input, uint256(dim) * W);
    }

    /// @notice Score every expert and return the `k` chosen, best first
    /// @return experts Their indices
    /// @return routeWeights Their routing weights, Q32, summing to one across the chosen
    ///
    /// @dev The two arrays are returned in parallel rather than as a struct array because a
    ///      caller uses them separately: the indices address weights, the weights scale
    ///      outputs, and pairing them would mean unpacking at every use.
    function route(
        uint32 nExperts,
        uint32 cols,
        address weights,
        uint32 offset,
        uint32 k,
        bytes memory x
    ) internal view returns (uint32[] memory experts, int256[] memory routeWeights) {
        require(x.length == uint256(cols) * W, "activation width");
        bytes memory input = new bytes(36 + x.length);
        assembly ("memory-safe") {
            let p := add(input, 0x20)
            // Ascending, so no field lands on top of one already written.
            mstore(p, or(shl(224, nExperts), shl(192, cols)))
            mstore(add(p, 8), shl(96, weights))
            mstore(add(p, 28), or(shl(224, offset), shl(192, k)))
            mcopy(add(p, 36), add(x, 0x20), mload(x))
        }

        // Twelve bytes an expert: a `uint32` index then an `int64` weight.
        bytes memory ret = _call(ROUTER, input, uint256(k) * 12);
        experts = new uint32[](k);
        routeWeights = new int256[](k);
        for (uint256 i = 0; i < k; ++i) {
            assembly ("memory-safe") {
                let word := mload(add(add(ret, 0x20), mul(i, 12)))
                mstore(add(add(experts, 0x20), mul(i, 0x20)), shr(224, word))
                mstore(add(add(routeWeights, 0x20), mul(i, 0x20)), sar(192, shl(32, word)))
            }
        }
    }
}
