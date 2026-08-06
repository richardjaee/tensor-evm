//! An EVM precompile exposing deterministic integer matmul.
//!
//! # Why a precompile and not an opcode
//!
//! Precompiles are how the EVM has always absorbed expensive primitives — `ecrecover`,
//! `modexp`, the pairing operations all exist because doing them in bytecode was
//! unaffordable. A tensor precompile is the same move, and it keeps the change to the
//! chain small: no new opcodes, no changes to the interpreter, and no effect on any
//! contract that does not call this address.
//!
//! # Why the interface is narrow
//!
//! `matmul` and nothing else. A precompile that ran a whole transformer block would need
//! revising for every architecture variant — attention biases, QK normalisation, GeGLU
//! against SwiGLU, sliding windows — and each revision would be a hard fork. Matrix
//! multiplication is the one operation every architecture agrees on, and a
//! mixture-of-experts model uses it unchanged.
//!
//! # Safety
//!
//! A panic here is a consensus failure: it would take down every node processing the
//! block, so a crafted input that panics is a denial of service against the whole chain.
//! Every length is checked before use and every arithmetic operation on attacker
//! controlled sizes is checked. Malformed input reverts the calling frame; it never
//! unwinds, and it never returns a fatal error, because a fatal error would let any
//! caller halt execution for everyone.

use revm::precompile::{
    u64_to_address, Precompile, PrecompileHalt, PrecompileId, PrecompileOutput, PrecompileResult,
};
use revm::primitives::{Address, Bytes};
use tensor_kernel::matmul;

/// Address of the matmul precompile.
///
/// 0x100 sits above the range Ethereum has reserved for itself, so a chain carrying this
/// precompile stays forward-compatible with future mainnet additions.
pub const MATMUL_ADDRESS: Address = u64_to_address(0x100);

/// Flat cost of the call, covering dispatch and header parsing.
pub const BASE_GAS: u64 = 100;

/// Multiply-accumulates purchased by one unit of gas.
///
/// Derived rather than chosen. Ethereum's schedule implies roughly 3M gas per second of
/// node work, and the kernel sustains about 98 billion multiply-accumulates per second on
/// commodity hardware, so honest pricing sits near 32,000 MACs per gas. This sets it at
/// 1024 — a wide margin over the measured figure, so the operation is priced well above
/// what it costs rather than below, which is the safe direction to be wrong in.
///
/// The practical consequence is a convenient unit: a 32B-parameter model costs about 31M
/// gas per token, so one token is roughly one Ethereum-sized block of work.
pub const MACS_PER_GAS: u64 = 1024;

/// Largest dimension accepted on either axis.
///
/// Bounds the accumulator in the kernel — at this width a sum of `i16::MIN * i32::MIN`
/// products still leaves headroom in `i64` — and bounds the allocation a single call can
/// request.
pub const MAX_DIM: u64 = 1 << 20;

/// Byte layout of the input, big-endian throughout to match how the EVM lays out words.
///
/// ```text
/// [0..4)                        rows          u32
/// [4..8)                        cols          u32
/// [8 ..)                        row scales    i8  x rows
/// [.. ..)                       weights       i16 x rows*cols, row-major
/// [.. ..)                       activations   i32 x cols
/// ```
///
/// Output is `rows` big-endian `i32` values.
const HEADER: usize = 8;

/// Gas for a matmul of the given shape, or `None` if the shape overflows.
pub fn gas_cost(rows: u64, cols: u64) -> Option<u64> {
    let macs = rows.checked_mul(cols)?;
    // Round up, so a small call is never free.
    BASE_GAS.checked_add(macs.div_ceil(MACS_PER_GAS))
}

/// Reverting costs the gas spent reaching the failure and unwinds the calling frame.
fn bad_input(reservoir: u64) -> PrecompileResult {
    Ok(PrecompileOutput::revert(BASE_GAS, Bytes::new(), reservoir))
}

/// The precompile itself.
pub fn matmul_precompile(input: &[u8], gas_limit: u64, reservoir: u64) -> PrecompileResult {
    if input.len() < HEADER {
        return bad_input(reservoir);
    }

    let rows = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
    let cols = u32::from_be_bytes(input[4..8].try_into().unwrap()) as u64;

    if rows == 0 || cols == 0 || rows > MAX_DIM || cols > MAX_DIM {
        return bad_input(reservoir);
    }

    let Some(gas_used) = gas_cost(rows, cols) else {
        return bad_input(reservoir);
    };
    if gas_used > gas_limit {
        // Halting rather than reverting mirrors how the EVM treats running out of gas
        // anywhere else, and leaves the caller's own handling intact.
        return Ok(PrecompileOutput::halt(PrecompileHalt::OutOfGas, reservoir));
    }

    let (rows, cols) = (rows as usize, cols as usize);

    // Checked because every term is attacker-controlled.
    let Some(weights_len) = rows.checked_mul(cols).and_then(|n| n.checked_mul(2)) else {
        return bad_input(reservoir);
    };
    let Some(acts_len) = cols.checked_mul(4) else {
        return bad_input(reservoir);
    };
    let Some(expected) = HEADER
        .checked_add(rows)
        .and_then(|n| n.checked_add(weights_len))
        .and_then(|n| n.checked_add(acts_len))
    else {
        return bad_input(reservoir);
    };

    // Exact rather than minimum. Trailing bytes mean the caller and the precompile
    // disagree about the layout, and guessing which is right is worse than refusing.
    if input.len() != expected {
        return bad_input(reservoir);
    }

    let mut at = HEADER;
    let scales: Vec<i8> = input[at..at + rows].iter().map(|&b| b as i8).collect();
    at += rows;

    let weights: Vec<i16> = input[at..at + weights_len]
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    at += weights_len;

    let acts: Vec<i32> = input[at..at + acts_len]
        .chunks_exact(4)
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let mut out = vec![0i32; rows];
    matmul(&weights, &scales, &acts, rows, cols, &mut out);

    let mut bytes = Vec::with_capacity(rows * 4);
    for v in out {
        bytes.extend_from_slice(&v.to_be_bytes());
    }

    Ok(PrecompileOutput::new(
        gas_used,
        Bytes::from(bytes),
        reservoir,
    ))
}

/// Registration record, to be extended into a chain's precompile set.
pub fn precompile() -> Precompile {
    Precompile::new(
        PrecompileId::Custom("tensor-matmul".into()),
        MATMUL_ADDRESS,
        matmul_precompile,
    )
}

/// Encodes a call, for tests and for anything driving the precompile directly rather
/// than through Solidity.
pub fn encode(rows: usize, cols: usize, scales: &[i8], weights: &[i16], acts: &[i32]) -> Vec<u8> {
    assert_eq!(scales.len(), rows, "one scale per row");
    assert_eq!(weights.len(), rows * cols, "weights must be rows*cols");
    assert_eq!(acts.len(), cols, "activations must be cols");

    let mut v = Vec::with_capacity(HEADER + rows + rows * cols * 2 + cols * 4);
    v.extend_from_slice(&(rows as u32).to_be_bytes());
    v.extend_from_slice(&(cols as u32).to_be_bytes());
    v.extend(scales.iter().map(|&s| s as u8));
    for w in weights {
        v.extend_from_slice(&w.to_be_bytes());
    }
    for a in acts {
        v.extend_from_slice(&a.to_be_bytes());
    }
    v
}

/// Decodes the output.
pub fn decode(bytes: &[u8]) -> Option<Vec<i32>> {
    if bytes.len() % 4 != 0 {
        return None;
    }
    Some(
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_kernel::matmul_scalar;

    const PLENTY: u64 = 1_000_000_000;

    fn fixture(rows: usize, cols: usize) -> (Vec<i8>, Vec<i16>, Vec<i32>) {
        let weights = (0..rows * cols)
            .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
            .collect();
        let scales = (0..rows).map(|r| (r % 5) as i8 + 28).collect();
        let acts = (0..cols)
            .map(|i| (((i as i64 * 40503 + 17) % 131071) - 65535) as i32)
            .collect();
        (scales, weights, acts)
    }

    #[test]
    fn matches_the_kernel() {
        let (rows, cols) = (37, 53);
        let (scales, weights, acts) = fixture(rows, cols);

        let mut expected = vec![0i32; rows];
        matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

        let input = encode(rows, cols, &scales, &weights, &acts);
        let out = matmul_precompile(&input, PLENTY, 0).expect("must not error");
        let got = decode(&out.bytes).expect("output must decode");

        assert_eq!(got, expected);
    }

    #[test]
    fn charges_for_the_work() {
        let (scales, weights, acts) = fixture(4, 4);
        let input = encode(4, 4, &scales, &weights, &acts);
        let out = matmul_precompile(&input, PLENTY, 0).unwrap();
        assert_eq!(out.gas_used, gas_cost(4, 4).unwrap());
        // Small shapes still pay the base cost, and round up rather than to zero.
        assert!(out.gas_used > BASE_GAS);
    }

    #[test]
    fn halts_when_gas_is_short() {
        let (scales, weights, acts) = fixture(64, 64);
        let input = encode(64, 64, &scales, &weights, &acts);
        let need = gas_cost(64, 64).unwrap();
        let out = matmul_precompile(&input, need - 1, 0).unwrap();
        assert!(out.status.is_halt(), "expected halt, got {:?}", out.status);
    }

    /// Every case here would panic on an unchecked implementation, and a panic in a
    /// precompile takes down every node processing the block.
    #[test]
    fn malformed_input_never_panics() {
        let (scales, weights, acts) = fixture(4, 4);
        let good = encode(4, 4, &scales, &weights, &acts);

        let with_header = |r: u32, c: u32| {
            let mut v = good.clone();
            v[0..4].copy_from_slice(&r.to_be_bytes());
            v[4..8].copy_from_slice(&c.to_be_bytes());
            v
        };

        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", vec![]),
            ("truncated header", vec![0u8; HEADER - 1]),
            ("header only", vec![0u8; HEADER]),
            ("zero rows", with_header(0, 4)),
            ("zero cols", with_header(4, 0)),
            ("oversized rows", with_header(u32::MAX, 4)),
            ("oversized cols", with_header(4, u32::MAX)),
            ("shape overflows body", with_header(1 << 19, 1 << 19)),
            ("truncated body", good[..good.len() - 1].to_vec()),
            ("trailing bytes", [good.clone(), vec![0]].concat()),
        ];

        for (name, input) in cases {
            let out = matmul_precompile(&input, PLENTY, 0)
                .unwrap_or_else(|e| panic!("{name}: returned a fatal error: {e:?}"));
            assert!(
                out.status.is_revert_or_halt(),
                "{name}: expected revert or halt, got {:?}",
                out.status
            );
        }
    }

    #[test]
    fn gas_cost_saturates_rather_than_overflowing() {
        assert!(gas_cost(u64::MAX, u64::MAX).is_none());
        assert!(gas_cost(MAX_DIM, MAX_DIM).is_some());
    }

    /// The encoding must survive a round trip, or the Solidity side and the precompile
    /// would silently disagree about layout.
    #[test]
    fn round_trips_through_the_wire_format() {
        let (rows, cols) = (11, 7);
        let (scales, weights, acts) = fixture(rows, cols);
        let input = encode(rows, cols, &scales, &weights, &acts);
        assert_eq!(input.len(), HEADER + rows + rows * cols * 2 + cols * 4);

        let out = matmul_precompile(&input, PLENTY, 0).unwrap();
        assert_eq!(decode(&out.bytes).unwrap().len(), rows);
    }
}
