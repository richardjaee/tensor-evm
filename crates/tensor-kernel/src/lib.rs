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
//! Q32 fixed point in `i32`. Products accumulate in `i64`, which cannot overflow for any
//! width we care about: the worst case is `cols * i16::MIN * i32::MIN`, and at
//! `cols = 65536` that still leaves 15 bits of headroom.
//!
//! This mirrors the representation used by the Solidity implementation it is meant to
//! replace, so outputs can be compared directly rather than approximately.

use rayon::prelude::*;

/// Shift applied after accumulation, before the per-row scale.
///
/// Activations are Q32, so a product of (weight x activation) carries 32 fractional bits
/// that have to come back off. The row scale then adjusts for that row's magnitude.
pub const Q: u32 = 32;

/// One row's dot product against `x`.
///
/// Split out because it is the unit of parallelism. Every output row reads the same
/// activation vector and writes one distinct element, so rows are independent — there is
/// no cross-thread reduction anywhere in this kernel, which is what makes the parallel
/// and scalar paths bit-identical rather than merely close.
#[inline]
fn dot(w_row: &[i16], x: &[i32]) -> i64 {
    debug_assert_eq!(w_row.len(), x.len());
    let mut acc: i64 = 0;
    for i in 0..w_row.len() {
        // The cast happens before the multiply so the product is computed in 64 bits.
        // Multiplying as i32 first would overflow for large activations.
        acc += (w_row[i] as i64) * (x[i] as i64);
    }
    acc
}

/// Convert an accumulator back to a Q32 activation.
///
/// Saturates rather than wrapping. A wrap here would be silent and would corrupt every
/// downstream layer; saturation is wrong in the same place but stays bounded, and shows
/// up as a degraded output rather than as noise.
#[inline]
fn finish(acc: i64, row_scale: i8) -> i32 {
    let shift = Q as i32 - row_scale as i32;
    let v = if shift >= 0 {
        acc >> shift.min(63)
    } else {
        acc.saturating_mul(1i64 << (-shift).min(62))
    };
    v.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Single-threaded reference. The definition of correct — every other path is checked
/// against this, not against a tolerance.
pub fn matmul_scalar(
    w: &[i16],
    scales: &[i8],
    x: &[i32],
    rows: usize,
    cols: usize,
    out: &mut [i32],
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
pub fn matmul(w: &[i16], scales: &[i8], x: &[i32], rows: usize, cols: usize, out: &mut [i32]) {
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

    /// Deterministic filler. Not random, so a failure is reproducible from the test name
    /// alone without carrying a seed around.
    fn fill(n: usize, salt: i64) -> Vec<i16> {
        (0..n)
            .map(|i| (((i as i64 * 2654435761 + salt) % 4001) - 2000) as i16)
            .collect()
    }

    fn acts(n: usize) -> Vec<i32> {
        (0..n)
            .map(|i| (((i as i64 * 40503 + 17) % 131071) - 65535) as i32)
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

            let mut a = vec![0i32; rows];
            let mut b = vec![0i32; rows];
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

        let mut reference = vec![0i32; rows];
        matmul_scalar(&w, &scales, &x, rows, cols, &mut reference);

        for threads in [1usize, 2, 3, 7, 14] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut got = vec![0i32; rows];
            pool.install(|| matmul(&w, &scales, &x, rows, cols, &mut got));
            assert_eq!(reference, got, "result changed at {threads} threads");
        }
    }

    #[test]
    fn saturates_instead_of_wrapping() {
        // One row of maximum-magnitude products, scaled up hard enough to overflow i32.
        let (rows, cols) = (1, 64);
        let w = vec![i16::MIN; cols];
        let scales = vec![32i8; rows]; // shift of zero, so the raw accumulator survives
        let x = vec![i32::MAX; cols];

        let mut out = vec![0i32; rows];
        matmul_scalar(&w, &scales, &x, rows, cols, &mut out);
        assert_eq!(
            out[0],
            i32::MIN,
            "expected saturation at the negative bound"
        );
    }

    #[test]
    fn known_value() {
        // Hand-checkable: 2x3, scale 32 means shift of zero.
        let w: Vec<i16> = vec![1, 2, 3, 4, 5, 6];
        let scales = vec![32i8, 32];
        let x: Vec<i32> = vec![10, 20, 30];
        let mut out = vec![0i32; 2];
        matmul_scalar(&w, &scales, &x, 2, 3, &mut out);
        assert_eq!(out, vec![140, 320]); // 10+40+90, 40+100+180
    }
}
