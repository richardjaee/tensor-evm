//! The classifier precompile: a matmul over the vocabulary, and the argmax of it.
//!
//! # Why these are fused
//!
//! Because the intermediate is enormous and nobody wants it. Qwen3's vocabulary is 151,936
//! entries, so the logits are 1.2MB — returning them would cost more in memory expansion
//! than the matmul costs in arithmetic, and the caller's next act is always to discard all
//! but one. Fusing keeps a vocabulary-sized value from ever existing: the running maximum
//! is two numbers rather than 151,936.
//!
//! This is the same trick the companion project uses to make a 32k vocabulary cost two
//! storage slots instead of thirty-two thousand, applied one level down.
//!
//! # Greedy only
//!
//! No sampling, no temperature. Sampling needs randomness, and randomness on a chain is
//! either a trusted input or a source of disagreement between nodes — both of which are
//! exactly what this design exists to avoid. A caller who wants sampling can have the top-k
//! and do it wherever their entropy comes from.

use revm::precompile::{
    u64_to_address, Precompile, PrecompileHalt, PrecompileId, PrecompileOutput, PrecompileResult,
};
use revm::primitives::{Address, Bytes};

use crate::{BASE_GAS, MACS_PER_GAS, MAX_DIM};

/// Address of the classifier.
pub const CLASSIFY_ADDRESS: Address = u64_to_address(0x106);

/// ```text
/// [0..4)    rows    u32, the vocabulary
/// [4..8)    cols    u32, the model width
/// [8..28)   weights address whose code holds the classifier
/// [28..32)  offset  u32, byte offset within that code
/// [32 ..)   x       i64 x cols
/// ```
///
/// Output is 12 bytes: the winning row as `u32`, then its logit as `i64`.
const HEADER: usize = 32;

pub fn gas_cost(rows: u64, cols: u64) -> Option<u64> {
    let macs = rows.checked_mul(cols)?;
    BASE_GAS.checked_add(macs.div_ceil(MACS_PER_GAS))
}

fn bad_input(reservoir: u64) -> PrecompileResult {
    Ok(PrecompileOutput::revert(BASE_GAS, Bytes::new(), reservoir))
}

/// Runs the classifier against weights held in chain state and returns only the winner.
///
/// `weights` is the tensor blob, in the same layout the matmul precompile reads: `rows`
/// scale bytes followed by row-major big-endian `i16`.
pub fn classify(
    weights: &[u8],
    offset: usize,
    rows: usize,
    cols: usize,
    x: &[i64],
) -> Option<(u32, i64)> {
    let weights_len = rows.checked_mul(cols)?.checked_mul(2)?;
    let end = offset.checked_add(rows)?.checked_add(weights_len)?;
    if end > weights.len() {
        return None;
    }

    let scales = &weights[offset..offset + rows];
    let values = &weights[offset + rows..end];

    let mut best_row = 0u32;
    let mut best = i64::MIN;

    for r in 0..rows {
        let row = &values[r * cols * 2..(r + 1) * cols * 2];
        let mut acc: i128 = 0;
        for (j, c) in row.chunks_exact(2).enumerate() {
            acc += (i16::from_be_bytes([c[0], c[1]]) as i128) * (x[j] as i128);
        }
        // Requantise as the matmul does: the accumulator carries `scale + 32` fractional
        // bits and a Q32 result wants 32.
        let shift = scales[r] as i8 as i32;
        let v = if shift >= 0 {
            acc >> shift.min(127)
        } else {
            acc.saturating_mul(1i128 << (-shift).min(126))
        };
        let v = v.clamp(i64::MIN as i128, i64::MAX as i128) as i64;

        // Strictly greater, so the first maximum wins — matching the reference argmax. A
        // `>=` here would pick the last, which is a different model on any tie.
        if v > best {
            best = v;
            best_row = r as u32;
        }
    }
    Some((best_row, best))
}

pub fn classify_precompile(input: &[u8], gas_limit: u64, reservoir: u64) -> PrecompileResult {
    if input.len() < HEADER {
        return bad_input(reservoir);
    }
    let rows = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
    let cols = u32::from_be_bytes(input[4..8].try_into().unwrap()) as u64;

    if rows == 0 || cols == 0 || cols > MAX_DIM {
        return bad_input(reservoir);
    }
    // The vocabulary is larger than any activation width, so it gets its own bound.
    if rows > (1 << 24) {
        return bad_input(reservoir);
    }

    let Some(gas_used) = gas_cost(rows, cols) else {
        return bad_input(reservoir);
    };
    if gas_used > gas_limit {
        return Ok(PrecompileOutput::halt(PrecompileHalt::OutOfGas, reservoir));
    }

    let cols_u = cols as usize;
    let Some(acts_len) = cols_u.checked_mul(8) else {
        return bad_input(reservoir);
    };
    if input.len() != HEADER + acts_len {
        return bad_input(reservoir);
    }

    // The weights live in chain state, which only the provider can reach. This entry point
    // exists so the address is a member of the precompile set — membership is what makes
    // the interpreter treat it as one — and the provider intercepts it before this runs.
    bad_input(reservoir)
}

pub fn precompile() -> Precompile {
    Precompile::new(
        PrecompileId::Custom("tensor-classify".into()),
        CLASSIFY_ADDRESS,
        classify_precompile,
    )
}

/// Encodes a call.
pub fn encode(rows: u32, cols: u32, weights: Address, offset: u32, x: &[i64]) -> Vec<u8> {
    assert_eq!(x.len(), cols as usize, "activation width");
    let mut v = Vec::with_capacity(HEADER + x.len() * 8);
    v.extend_from_slice(&rows.to_be_bytes());
    v.extend_from_slice(&cols.to_be_bytes());
    v.extend_from_slice(weights.as_slice());
    v.extend_from_slice(&offset.to_be_bytes());
    for a in x {
        v.extend_from_slice(&a.to_be_bytes());
    }
    v
}

/// Decodes the reply: winning row, then its logit.
pub fn decode(bytes: &[u8]) -> Option<(u32, i64)> {
    if bytes.len() != 12 {
        return None;
    }
    Some((
        u32::from_be_bytes(bytes[0..4].try_into().ok()?),
        i64::from_be_bytes(bytes[4..12].try_into().ok()?),
    ))
}

/// Packs a reply.
pub fn pack(row: u32, logit: i64) -> Bytes {
    let mut b = Vec::with_capacity(12);
    b.extend_from_slice(&row.to_be_bytes());
    b.extend_from_slice(&logit.to_be_bytes());
    Bytes::from(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_kernel::matmul;

    fn blob(rows: usize, cols: usize) -> (Vec<u8>, Vec<i8>, Vec<i16>) {
        let scales: Vec<i8> = (0..rows).map(|r| (r % 5) as i8 + 20).collect();
        let w: Vec<i16> = (0..rows * cols)
            .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
            .collect();
        let mut b = scales.iter().map(|&s| s as u8).collect::<Vec<u8>>();
        for v in &w {
            b.extend_from_slice(&v.to_be_bytes());
        }
        (b, scales, w)
    }

    /// The fused path must pick exactly what a full matmul followed by an argmax would.
    #[test]
    fn matches_matmul_then_argmax() {
        let (rows, cols) = (97usize, 33usize);
        let (b, scales, w) = blob(rows, cols);
        let x: Vec<i64> = (0..cols)
            .map(|i| ((i as i64 * 40503 + 17) % 131071) - 65535)
            .collect();

        let mut logits = vec![0i64; rows];
        matmul(&w, &scales, &x, rows, cols, &mut logits);
        let want = tensor_kernel::fixed::argmax(&logits);

        let (row, logit) = classify(&b, 0, rows, cols, &x).expect("in range");
        assert_eq!(row as usize, want, "different token");
        assert_eq!(logit, logits[want], "different logit");
    }

    /// Ties go to the first row, matching the reference. Picking the last would be a
    /// different model on every tie and would never surface as an error.
    #[test]
    fn ties_go_to_the_first_row() {
        let (rows, cols) = (4usize, 2usize);
        // Every row identical, so every logit is identical.
        let scales = vec![20i8; rows];
        let w = vec![7i16; rows * cols];
        let mut b: Vec<u8> = scales.iter().map(|&s| s as u8).collect();
        for v in &w {
            b.extend_from_slice(&v.to_be_bytes());
        }
        let x = vec![1i64 << 30; cols];

        let (row, _) = classify(&b, 0, rows, cols, &x).expect("in range");
        assert_eq!(row, 0);
    }

    /// An offset lets the classifier share a blob with the rest of a model.
    #[test]
    fn honours_an_offset() {
        let (rows, cols) = (16usize, 8usize);
        let (b, _, _) = blob(rows, cols);
        let mut padded = vec![0xabu8; 999];
        padded.extend_from_slice(&b);
        let x: Vec<i64> = (0..cols).map(|i| (i as i64 - 4) << 28).collect();

        assert_eq!(
            classify(&padded, 999, rows, cols, &x),
            classify(&b, 0, rows, cols, &x)
        );
    }

    /// A blob too short for the shape must be refused rather than read past.
    #[test]
    fn refuses_a_blob_that_cannot_hold_the_shape() {
        let (rows, cols) = (8usize, 4usize);
        let (b, _, _) = blob(rows, cols);
        let x = vec![0i64; cols];
        assert!(classify(&b, 0, rows * 2, cols, &x).is_none());
        assert!(classify(&b, b.len(), rows, cols, &x).is_none());
    }

    #[test]
    fn reply_round_trips() {
        let b = pack(151_935, -42);
        assert_eq!(decode(&b), Some((151_935, -42)));
        assert_eq!(decode(&b[..11]), None);
    }
}
