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
//! Products and sums are both formed in `i128`, and no bound is placed on activations.
//!
//! An earlier version formed products in `i64` and only widened the running sum, which is
//! faster because it vectorises, and required activations to stay under a bound so the
//! product could not wrap. That bound was wrong three times. Qwen3's `k_norm` scales keys
//! by 96.5; its `post_norm` reaches 192; and its residual stream — the thing every matmul
//! consumes — peaks near 7,000 by the third layer, because the model genuinely has massive
//! activations. At that magnitude a single `i16 * i64` product already exceeds `i64`, so
//! there is no bound that both admits the real model and permits the optimisation.
//!
//! `i128` throughout costs roughly a quarter of the throughput and removes the whole class
//! of failure. It is worth it: the kernel's gas is 18,712 of a layer's 688,198, so its
//! speed barely moves the number this project is measured by, while a wrapped accumulator
//! would be a consensus failure rather than a slow one.
//!
//! This mirrors the representation used by the Solidity implementation it is meant to
//! replace, so outputs can be compared directly rather than approximately.

pub mod fixed;

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
    for i in 0..w_row.len() {
        // Widened before the multiply. The residual stream of a real model reaches
        // magnitudes where an `i64` product wraps, and a wrapped accumulator on a chain is
        // a consensus failure rather than a rounding error.
        acc += (w_row[i] as i128) * (x[i] as i128);
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
/// Measured by sweep, and re-measured after activations widened to `i64` — which moved the
/// crossover, because the kernel itself got slower and the dispatch cost did not. At 2^18 a
/// 576-wide projection regresses to 0.7x scalar; at 2^21 a 1024x1024 projection is denied
/// parallelism it would gain 1.8x from. 2^19 is the first value that costs nothing on the
/// small shape and recovers the larger one.
///
/// This mattered more than it looks. Qwen3-0.6B's `k_proj` is exactly 1024x1024, and under
/// the old threshold every key projection of every layer ran single-threaded.
pub const PARALLEL_THRESHOLD: usize = 1 << 19;

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

/// SiLU applied across a vector, in place.
///
/// Vectorised because the cost that matters is per call, not per element. In the EVM a
/// single SiLU is 6,235 gas, so a layer's 3,072 of them come to 19M — against 15,360 for
/// the same layer's matmuls once those are a precompile. Exposing this one element at a
/// time would keep the calling overhead and leave the bottleneck where it was.
pub fn silu_vec(x: &mut [i64]) {
    for v in x.iter_mut() {
        *v = fixed::silu(*v);
    }
}

/// `silu(gate) * up`, the SwiGLU pairing, written into `gate`.
pub fn swiglu_vec(gate: &mut [i64], up: &[i64]) {
    assert_eq!(gate.len(), up.len(), "gate and up must match");
    for i in 0..gate.len() {
        let g = fixed::silu(gate[i]);
        gate[i] = ((g as i128 * up[i] as i128) >> fixed::F) as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Accumulation must not depend on width. This guarded chunk boundaries when the
    /// kernel batched into `i64`; it still earns its place as a check that the reference
    /// and the kernel agree across awkward lengths.
    #[test]
    fn accumulation_is_width_independent() {
        for cols in [1usize, 2, 63, 64, 65, 2047, 2048, 6151] {
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

    /// The magnitudes a real model actually reaches. Qwen3's residual stream peaks near
    /// 7,000 by its third layer, which is where every bound this kernel used to impose came
    /// apart — so the property worth testing is that no bound is left to exceed.
    #[test]
    fn survives_the_magnitudes_a_real_model_reaches() {
        let cols = 1024usize;
        let w = vec![i16::MAX; cols];
        let big = (7000.0 * (1u64 << 32) as f64) as i64;
        let x = vec![big; cols];
        let mut out = vec![0i64; 1];
        matmul_scalar(&w, &[40], &x, 1, cols, &mut out);

        // Computed independently at full width, so this checks the specification rather
        // than the implementation restating itself.
        let want = ((i16::MAX as i128) * (big as i128) * (cols as i128)) >> 40;
        assert_eq!(
            out[0] as i128,
            want.clamp(i64::MIN as i128, i64::MAX as i128)
        );
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
        let x = vec![i64::MAX / 4; cols];

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
