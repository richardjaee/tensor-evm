//! Q32 fixed-point primitives: square root, exponential, SiLU.
//!
//! Everything a transformer needs beyond matrix multiplication, in integers. No floating
//! point appears anywhere, for the same reason it appears nowhere else in this project: a
//! chain is only useful if every node computes identical bits, and float does not give you
//! that across machines.
//!
//! The algorithms follow the companion project's Solidity library, which in turn follows
//! Karpathy's llama2.c. Matching it matters — a chain running this must agree with the
//! implementation it replaces, or the two are different models rather than one model in two
//! places.

/// 1.0 in Q32.
pub const ONE: i64 = 1 << 32;

/// Fractional bits.
pub const F: u32 = 32;

/// `floor(log2(e) * 2^32)`, which turns `exp` into a base-2 problem.
pub const LOG2E: i64 = 6_196_328_018;

/// RMSNorm's epsilon for the Llama family, 1e-5 in Q32.
///
/// Not a universal constant. Qwen3 specifies 1e-6, and models are free to specify anything;
/// `rmsnorm` takes it as an argument for that reason. Kept here as a named default for
/// callers that genuinely want the Llama value.
pub const RMS_EPS_LLAMA: i64 = 42_950;

/// RMSNorm's epsilon for Qwen3, 1e-6 in Q32.
pub const RMS_EPS_QWEN3: i64 = 4_295;

/// Bits of the fractional exponent consumed by the bit-product in [`exp_q32`].
const EXP_STEPS: usize = 32;

/// Floor integer square root: the largest `r` with `r*r <= a`.
///
/// Newton's method from a power-of-two seed at or above the true root. The iteration is
/// monotonically decreasing from such a seed, so the first non-decreasing step is the
/// answer — no epsilon, no iteration cap, and exact by construction.
pub fn isqrt(a: u128) -> u128 {
    if a < 2 {
        return a;
    }
    let bits = 128 - a.leading_zeros();
    let mut x = 1u128 << ((bits + 1) >> 1);
    loop {
        let y = (x + a / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// `2^(-2^-(i+1))` in Q64, for `i` in `0..EXP_STEPS`.
///
/// Derived rather than tabulated. Each constant is the square root of the one before it,
/// starting from `2^-0.5 = sqrt(2^127)`, so the whole table comes out of [`isqrt`] and has
/// no float provenance at all. The Solidity side embeds the same values because recomputing
/// them costs gas; here there is no reason not to derive them, and deriving them means the
/// table cannot silently disagree with the square root it was supposed to come from.
fn exp2_table() -> [u128; EXP_STEPS] {
    let mut t = [0u128; EXP_STEPS];
    // 2^(-1/2) in Q64 == sqrt(2^127), exactly.
    t[0] = isqrt(1u128 << 127);
    for i in 1..EXP_STEPS {
        // sqrt of a Q64 value, kept in Q64: sqrt(v * 2^64) = sqrt(v) * 2^32, so shifting
        // left by 64 before the root lands the result back in Q64.
        t[i] = isqrt(t[i - 1] << 64);
    }
    t
}

fn exp2_const(i: usize) -> u128 {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[u128; EXP_STEPS]> = OnceLock::new();
    TABLE.get_or_init(exp2_table)[i]
}

/// `exp(x)` for Q32 `x <= 0`, returning Q32 in `[0, 2^32]`.
///
/// `exp(x) = 2^(x * log2 e)`. The integer part of that exponent becomes a shift; the
/// fractional part becomes a product of `2^(-2^-i)` factors selected by its bits. Restricted
/// to non-positive arguments because that is all softmax and the sigmoid need, and because
/// it keeps the result bounded by 1.0 so no intermediate can overflow.
///
/// # Panics
///
/// If `x > 0`. Callers arrange the sign themselves — see [`silu`].
pub fn exp_q32(x: i64) -> u64 {
    assert!(x <= 0, "exp_q32 is defined for non-positive arguments only");
    if x == 0 {
        return ONE as u64;
    }

    let e = (((-x) as i128 * LOG2E as i128) >> F) as u128; // |x| * log2 e, Q32
    let int_part = (e >> F) as u32;
    if int_part >= 64 {
        return 0; // underflows Q32
    }
    let frac = e - ((int_part as u128) << F);

    let mut acc = 1u128 << 64; // 1.0 in Q64
    for i in 0..EXP_STEPS {
        if (frac >> (F as usize - 1 - i)) & 1 == 1 {
            acc = (acc * exp2_const(i)) >> 64;
        }
    }
    ((acc >> int_part) >> 32) as u64
}

/// SiLU: `z * sigmoid(z)`, Q32 in and out.
///
/// Branches on the sign so [`exp_q32`] only ever sees a non-positive argument and the
/// sigmoid is formed without cancellation — computing `1/(1+e^-z)` directly for large
/// negative `z` would subtract nearly equal quantities and lose most of the precision.
pub fn silu(z: i64) -> i64 {
    let sig: i64 = if z >= 0 {
        let e = exp_q32(-z) as i128;
        (((ONE as i128) << F) / (ONE as i128 + e)) as i64
    } else {
        let e = exp_q32(z) as i128;
        ((e << F) / (ONE as i128 + e)) as i64
    };
    (((z as i128) * (sig as i128)) >> F) as i64
}

/// The `k` largest values, best first, as `(index, value)`.
///
/// Ties go to the lower index, and the sort is by value then index rather than by value
/// alone. That is not a detail: a mixture-of-experts router picks which parameters a token
/// reads, so two nodes that break a tie differently do not merely round differently — they
/// read different weights and produce different tokens. Selection sort rather than
/// `sort_unstable_by` because `k` is small, and because "unstable" is precisely the property
/// that must not appear anywhere in consensus.
pub fn top_k(v: &[i64], k: usize, out: &mut Vec<(usize, i64)>) {
    out.clear();
    let k = k.min(v.len());
    for _ in 0..k {
        let mut best: Option<(usize, i64)> = None;
        for (i, &x) in v.iter().enumerate() {
            if out.iter().any(|&(j, _)| j == i) {
                continue;
            }
            match best {
                Some((_, bv)) if x <= bv => {}
                _ => best = Some((i, x)),
            }
        }
        if let Some(b) = best {
            out.push(b);
        }
    }
}

/// First maximum wins, matching the companion project's greedy selection.
pub fn argmax(v: &[i64]) -> usize {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q32 values compared as reals, since the point is agreement with mathematics rather
    /// than with another integer implementation.
    fn q(v: i64) -> f64 {
        v as f64 / ONE as f64
    }

    #[test]
    fn isqrt_is_exact_on_perfect_squares() {
        for n in [
            0u128,
            1,
            2,
            3,
            4,
            9,
            16,
            1_000_000,
            1u128 << 64,
            1u128 << 100,
        ] {
            let r = isqrt(n);
            assert!(r * r <= n, "isqrt({n}) = {r} overshot");
            assert!(
                (r + 1).checked_mul(r + 1).is_none_or(|s| s > n),
                "isqrt({n}) = {r} undershot"
            );
        }
    }

    /// The derived table must actually be the square-root chain it claims to be.
    #[test]
    fn exp2_table_is_a_square_root_chain() {
        let t = exp2_table();
        for i in 1..EXP_STEPS {
            assert_eq!(
                t[i],
                isqrt(t[i - 1] << 64),
                "entry {i} is not the root of {}",
                i - 1
            );
        }
        // First entry is 2^-0.5; check against the value the Solidity library embeds.
        assert_eq!(t[0], 0xb504_f333_f9de_6484);
    }

    /// Accuracy is judged against Q32's own resolution rather than a fixed relative
    /// tolerance. `exp(-10)` is about 4.5e-5, which Q32 represents in roughly seventeen
    /// bits, so demanding a relative error of 1e-6 would be asking for precision the
    /// format does not have. The meaningful question is whether the result is within a
    /// couple of representable steps of the true value, and it is.
    #[test]
    fn exp_matches_the_real_exponential() {
        const ULP: f64 = 1.0 / (1u64 << 32) as f64;
        for &x in &[0.0f64, -0.1, -0.5, -1.0, -2.0, -5.0, -10.0, -20.0] {
            let got = exp_q32((x * ONE as f64) as i64) as f64 / ONE as f64;
            let want = x.exp();
            let abs = (got - want).abs();
            // Either a couple of ULPs, or a relative error small enough that the value is
            // large enough for relative error to mean anything.
            let ok = abs <= 4.0 * ULP || abs / want < 1e-7;
            assert!(
                ok,
                "exp({x}): got {got}, want {want}, abs {abs:.3e} ({:.1} ULP)",
                abs / ULP
            );
        }
    }

    #[test]
    fn exp_underflows_to_zero_rather_than_wrapping() {
        assert_eq!(exp_q32(-100 * ONE), 0);
    }

    #[test]
    fn silu_matches_its_definition() {
        for &z in &[-8.0f64, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 8.0] {
            let got = q(silu((z * ONE as f64) as i64));
            let want = z / (1.0 + (-z).exp());
            assert!(
                (got - want).abs() < 1e-6,
                "silu({z}): got {got}, want {want}"
            );
        }
    }

    /// SiLU's defining shape: it is not monotonic, and its minimum sits near -1.278.
    /// A sign error or a mixed-up branch would break this while still looking plausible
    /// at isolated points.
    #[test]
    fn silu_has_its_characteristic_dip() {
        let at = |z: f64| q(silu((z * ONE as f64) as i64));
        assert!(at(-1.278) < at(-3.0), "left of the dip should be higher");
        assert!(at(-1.278) < at(0.0), "right of the dip should be higher");
        assert!(at(-1.278) < -0.27 && at(-1.278) > -0.29);
    }

    #[test]
    fn argmax_takes_the_first_maximum() {
        assert_eq!(argmax(&[1, 5, 5, 2]), 1);
        assert_eq!(argmax(&[-3, -1, -2]), 1);
    }
}

#[cfg(test)]
mod top_k_tests {
    use super::*;

    #[test]
    fn returns_the_largest_in_order() {
        let v = [3i64, 9, 1, 7, 5];
        let mut out = Vec::new();
        top_k(&v, 3, &mut out);
        assert_eq!(out, vec![(1, 9), (3, 7), (4, 5)]);
    }

    /// The property consensus depends on. Equal scores must resolve to the lower index on
    /// every machine, because the winner decides which expert's weights a token reads.
    #[test]
    fn ties_go_to_the_lower_index() {
        let v = [5i64, 5, 5, 5];
        let mut out = Vec::new();
        top_k(&v, 2, &mut out);
        assert_eq!(out, vec![(0, 5), (1, 5)]);
    }

    #[test]
    fn handles_negatives_and_a_k_past_the_end() {
        let v = [-9i64, -1, -5];
        let mut out = Vec::new();
        top_k(&v, 10, &mut out);
        assert_eq!(out, vec![(1, -1), (2, -5), (0, -9)]);
    }

    /// Selecting one must agree with `argmax`, or a router and a classifier would disagree
    /// about what "largest" means.
    #[test]
    fn agrees_with_argmax_at_k_of_one() {
        let mut out = Vec::new();
        for seed in 0..64i64 {
            let v: Vec<i64> = (0..37).map(|i| ((i * 37 + seed * 11) % 23) - 11).collect();
            top_k(&v, 1, &mut out);
            assert_eq!(out[0].0, argmax(&v), "seed {seed}");
        }
    }
}
