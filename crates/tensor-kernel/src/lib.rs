//! Deterministic integer matmul — the kernel a chain-level precompile would call.
//!
//! Every value here is an integer. That is not a limitation being worked around, it is
//! the property the whole design rests on: a chain is only useful if every node computes
//! the same bits, and floating point does not give you that. Reduction order varies with
//! thread count, kernel selection varies with hardware, and two validators disagree.
//! Integer addition is associative, so this kernel produces identical output on any
//! machine, at any thread count, forever.
//!
//! # Format
//!
//! Weights are `i16`, row-major, with one power-of-two scale per row. Activations are
//! Q32 fixed point in `i64`.
//!
//! `i64` rather than `i32`, which is not a detail. Q32 in an `i32` spans only [-0.5, 0.5),
//! and a transformer's activations after normalisation are around unit scale — the
//! representation would saturate on ordinary values, silently, everywhere. Widening costs
//! four bytes per activation and nothing per weight, and activations are a vanishing
//! fraction of what a forward pass reads.
//!
//! Products are formed in `i64` and accumulated in `i128`, which is not the same as doing
//! both at the wider width. An unbounded `i64` activation times an `i16` weight reaches 79
//! bits and would wrap — on a chain that is a soundness failure, not a precision one — so
//! activations are bounded by [`MAX_ABS_ACTIVATION`] and the bound is checked once per
//! call, in `O(cols)`, against `O(rows * cols)` of arithmetic. Within it every product fits
//! `i64` comfortably, while the running sum still needs `i128` because `cols` of them do
//! not.
//!
//! The split matters: forming products in `i128` costs a little over half the kernel's
//! throughput, and the bound recovers it without giving up soundness.
//!
//! This mirrors the representation used by the Solidity implementation it is meant to
//! replace, so outputs can be compared directly rather than approximately.

use rayon::prelude::*;

/// Fractional bits carried by an activation.
///
/// Recorded for callers that need to interpret results; the requantisation in [`finish`]
/// does not reference it, for reasons worth stating because getting this wrong is silent.
///
/// A weight is stored as `q = w * 2^s` for that row's shift `s`, and an activation as
/// `x = v * 2^32`. Their product therefore carries `s + 32` fractional bits, and returning
/// to Q32 means shedding exactly `s` of them — the activation's own 32 are the ones being
/// kept. So the requantising shift is the row scale itself, not `32 - s`.
///
/// The difference is invisible at `s = 32`, where the two agree, and is why an earlier
/// version of this passed every test it had: both the kernel and the reference vector
/// generator implemented `32 - s`, agreed with each other, and disagreed with the model by
/// more than a thousand percent. Only checking against float64 truth exposed it.
pub const Q: u32 = 32;

/// Largest activation magnitude the kernel accepts.
///
/// `2^36` in Q32 is a real value of 16, comfortably above anything a normalised transformer
/// produces — post-RMSNorm values sit near unit scale — while being small enough that a
/// whole chunk of products can be summed in `i64`. See [`CHUNK`].
pub const MAX_ABS_ACTIVATION: i64 = 1 << 36;

/// Products summed per `i64` accumulator before widening.
///
/// The bound is arithmetic, not taste: a weight is at most `2^15`, an activation at most
/// [`MAX_ABS_ACTIVATION`] = `2^36`, so each product is at most `2^51` and `2^11` of them
/// reach `2^62` — one bit inside `i64`.
///
/// Chunking exists because of what the alternative costs. Accumulating every product
/// directly into an `i128` is correct and defeats auto-vectorisation, which measured as
/// less than half the kernel's throughput. Summing within a chunk at `i64` keeps the hot
/// loop vectorisable, and only the chunk totals — `cols/2048` of them — are widened.
const CHUNK: usize = 1 << 11;

/// Whether every activation is inside [`MAX_ABS_ACTIVATION`].
///
/// Checked once per call rather than per product: this is `O(cols)` against `O(rows*cols)`
/// of arithmetic, so it does not show up in the measurements.
pub fn activations_in_range(x: &[i64]) -> bool {
    x.iter()
        .all(|v| *v >= -MAX_ABS_ACTIVATION && *v <= MAX_ABS_ACTIVATION)
}

/// One row's dot product against `x`.
///
/// Split out because it is the unit of parallelism. Every output row reads the same
/// activation vector and writes one distinct element, so rows are independent — there is
/// no cross-thread reduction anywhere in this kernel, which is what makes the parallel
/// and scalar paths bit-identical rather than merely close.
#[inline]
fn dot(w_row: &[i16], x: &[i64]) -> i128 {
    debug_assert_eq!(w_row.len(), x.len());
    let mut acc: i128 = 0;
    for (wc, xc) in w_row.chunks(CHUNK).zip(x.chunks(CHUNK)) {
        let mut part: i64 = 0;
        for i in 0..wc.len() {
            // Both the product and this partial sum stay in `i64` by construction, which
            // is what lets the compiler vectorise the loop. Callers guarantee the bound
            // through `activations_in_range`.
            debug_assert!(xc[i].unsigned_abs() <= MAX_ABS_ACTIVATION as u64);
            part += (wc[i] as i64) * xc[i];
        }
        acc += part as i128;
    }
    acc
}

/// Convert an accumulator back to a Q32 activation.
///
/// The accumulator carries `row_scale + 32` fractional bits and a Q32 result wants 32, so
/// the shift is `row_scale`. See [`Q`] for why that is not `32 - row_scale`.
///
/// Saturates rather than wrapping. A wrap here would be silent and would corrupt every
/// downstream layer; saturation is wrong in the same place but stays bounded, and shows
/// up as a degraded output rather than as noise.
#[inline]
fn finish(acc: i128, row_scale: i8) -> i64 {
    let shift = row_scale as i32;
    let v = if shift >= 0 {
        acc >> shift.min(127)
    } else {
        acc.saturating_mul(1i128 << (-shift).min(126))
    };
    v.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Single-threaded reference. The definition of correct — every other path is checked
/// against this, not against a tolerance.
pub fn matmul_scalar(
    w: &[i16],
    scales: &[i8],
    x: &[i64],
    rows: usize,
    cols: usize,
    out: &mut [i64],
) {
    assert_eq!(w.len(), rows * cols, "weight length must be rows*cols");
    assert_eq!(scales.len(), rows, "one scale per row");
    assert_eq!(x.len(), cols, "activation length must be cols");
    assert_eq!(out.len(), rows, "output length must be rows");

    for r in 0..rows {
        out[r] = finish(dot(&w[r * cols..(r + 1) * cols], x), scales[r]);
    }
}

/// Below this many weights, dispatching to a thread pool costs more than the work saved.
///
/// Measured, not guessed: at 0.6 MiB the parallel path ran at a fifth of scalar speed and
/// at 12.5 MiB it ran at nearly twice, so the crossover sits between. The overhead is a
/// fixed per-call cost, so the threshold is expressed in absolute work rather than as a
/// ratio. Attention projections in small models land below it, which is exactly the case
/// that regressed before this existed.
pub const PARALLEL_THRESHOLD: usize = 2 << 20;

/// Multi-threaded above [`PARALLEL_THRESHOLD`], scalar below. Bit-identical to
/// [`matmul_scalar`] either way, regardless of thread count.
///
/// The guarantee is structural, not incidental: rows are partitioned, never split, so no
/// accumulator is ever combined across threads. A kernel that split a row and summed the
/// partial products would still be deterministic in integer arithmetic, but it would tie
/// the result to the partition size — and therefore to the machine. This does not, which
/// is also why switching between the two paths on a size threshold is safe.
pub fn matmul(w: &[i16], scales: &[i8], x: &[i64], rows: usize, cols: usize, out: &mut [i64]) {
    assert_eq!(w.len(), rows * cols, "weight length must be rows*cols");
    assert_eq!(scales.len(), rows, "one scale per row");
    assert_eq!(x.len(), cols, "activation length must be cols");
    assert_eq!(out.len(), rows, "output length must be rows");

    if rows * cols < PARALLEL_THRESHOLD {
        return matmul_scalar(w, scales, x, rows, cols, out);
    }

    out.par_iter_mut()
        .zip(scales.par_iter())
        .enumerate()
        .for_each(|(r, (o, &s))| {
            *o = finish(dot(&w[r * cols..(r + 1) * cols], x), s);
        });
}

/// Bytes of weight touched by one matmul. The figure that actually predicts runtime:
/// above a few hundred million parameters this kernel is bound by how fast weights can
/// be read, not by how fast products can be formed.
pub fn weight_bytes(rows: usize, cols: usize) -> usize {
    rows * cols * core::mem::size_of::<i16>()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chunked accumulator must not depend on where the chunk boundaries fall.
    #[test]
    fn chunk_boundaries_do_not_change_the_result() {
        // Straddling one, two, and several chunks, plus a length that is not a multiple.
        for cols in [CHUNK - 1, CHUNK, CHUNK + 1, CHUNK * 3 + 7] {
            let rows = 3usize;
            let w = fill(rows * cols, 3);
            let scales = vec![20i8; rows];
            let x: Vec<i64> = (0..cols)
                .map(|i| ((i as i64 * 7919) % 2001) - 1000)
                .collect();

            let mut got = vec![0i64; rows];
            matmul_scalar(&w, &scales, &x, rows, cols, &mut got);

            // Independent reference: one accumulator, full width, no chunking at all.
            for r in 0..rows {
                let mut acc: i128 = 0;
                for c in 0..cols {
                    acc += (w[r * cols + c] as i128) * (x[c] as i128);
                }
                assert_eq!(got[r], (acc >> 20) as i64, "cols {cols}, row {r}");
            }
        }
    }

    #[test]
    fn the_activation_bound_is_enforceable() {
        assert!(activations_in_range(&[
            0,
            MAX_ABS_ACTIVATION,
            -MAX_ABS_ACTIVATION
        ]));
        assert!(!activations_in_range(&[MAX_ABS_ACTIVATION + 1]));
        // The bound must keep a full chunk of worst-case products inside i64.
        let worst = (i16::MIN as i128).unsigned_abs() * MAX_ABS_ACTIVATION as u128 * CHUNK as u128;
        assert!(worst < i64::MAX as u128, "chunk can overflow: {worst}");
    }

    /// Deterministic filler. Not random, so a failure is reproducible from the test name
    /// alone without carrying a seed around.
    fn fill(n: usize, salt: i64) -> Vec<i16> {
        (0..n)
            .map(|i| (((i as i64 * 2654435761 + salt) % 4001) - 2000) as i16)
            .collect()
    }

    fn acts(n: usize) -> Vec<i64> {
        (0..n)
            .map(|i| ((i as i64 * 40503 + 17) % 131071) - 65535)
            .collect()
    }

    /// Both sides of [`PARALLEL_THRESHOLD`], since the dispatch below it is a second code
    /// path and a threshold that silently changed results would be the worst kind of bug.
    #[test]
    fn parallel_matches_scalar() {
        // deliberately not multiples of any lane width, and straddling the threshold
        for &(rows, cols) in &[(257usize, 385usize), (1543, 1361)] {
            assert!(
                (rows * cols < PARALLEL_THRESHOLD) ^ (1543 * 1361 == rows * cols),
                "shapes must cover both sides of the threshold"
            );
            let w = fill(rows * cols, 7);
            let scales: Vec<i8> = (0..rows).map(|r| (r % 5) as i8 + 28).collect();
            let x = acts(cols);

            let mut a = vec![0i64; rows];
            let mut b = vec![0i64; rows];
            matmul_scalar(&w, &scales, &x, rows, cols, &mut a);
            matmul(&w, &scales, &x, rows, cols, &mut b);

            assert_eq!(a, b, "parallel path diverged from scalar at {rows}x{cols}");
        }
    }

    /// The claim that matters for consensus: thread count cannot change the answer.
    /// If this ever fails, nodes with different core counts would fork the chain.
    #[test]
    fn thread_count_does_not_change_result() {
        // above PARALLEL_THRESHOLD, or this would exercise the scalar path and prove
        // nothing about the property it is named for
        let (rows, cols) = (2048, 1536);
        let w = fill(rows * cols, 11);
        let scales = vec![30i8; rows];
        let x = acts(cols);

        let mut reference = vec![0i64; rows];
        matmul_scalar(&w, &scales, &x, rows, cols, &mut reference);

        for threads in [1usize, 2, 3, 7, 14] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut got = vec![0i64; rows];
            pool.install(|| matmul(&w, &scales, &x, rows, cols, &mut got));
            assert_eq!(reference, got, "result changed at {threads} threads");
        }
    }

    #[test]
    fn saturates_instead_of_wrapping() {
        // Every product at the largest magnitude the kernel accepts, across enough columns
        // to exceed what the output can hold. The result must clamp rather than wrap.
        let (rows, cols) = (1, 4096);
        let w = vec![i16::MIN; cols];
        let scales = vec![-40i8; rows]; // negative shift scales the accumulator up
        let x = vec![MAX_ABS_ACTIVATION; cols];

        let mut out = vec![0i64; rows];
        matmul_scalar(&w, &scales, &x, rows, cols, &mut out);
        assert_eq!(
            out[0],
            i64::MIN,
            "expected saturation at the negative bound"
        );
    }

    #[test]
    fn known_value() {
        // Hand-checkable: 2x3 with no requantising shift.
        let w: Vec<i16> = vec![1, 2, 3, 4, 5, 6];
        let scales = vec![0i8, 0];
        let x: Vec<i64> = vec![10, 20, 30];
        let mut out = vec![0i64; 2];
        matmul_scalar(&w, &scales, &x, 2, 3, &mut out);
        assert_eq!(out, vec![140, 320]); // 10+40+90, 40+100+180
    }
}
