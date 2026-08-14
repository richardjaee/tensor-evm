//! The transformer operations that are not matrix multiplication.
//!
//! Normalisation, softmax, rotary embedding, and the SwiGLU pairing. All in Q32 integers,
//! all deterministic, and each one small enough to check against float64 directly rather
//! than against another integer implementation — which is the mistake that cost this
//! project a thousand-percent error before real weights were introduced.

use tensor_kernel::fixed::{exp_q32, isqrt, silu, F, ONE};

/// RMSNorm: `x / sqrt(mean(x^2) + eps) * weight`.
///
/// `eps` is Q32 and comes from the model rather than from a constant. Llama specifies 1e-5
/// and Qwen3 specifies 1e-6, and while the difference is negligible against a mean square
/// near unity, a fixed value is a place where a chain and its reference can quietly hold
/// different models. Two implementations that disagree in the last place are not one model
/// in two places.
///
/// The sum of squares accumulates in `i128`. Each term is Q64 and there are `dim` of them,
/// so at dim 8192 the worst case needs 77 bits — comfortably inside 128 but well outside
/// the 64 that would look sufficient at a glance.
pub fn rmsnorm(x: &[i64], weight: &[i64], eps: i64, out: &mut [i64]) {
    assert_eq!(x.len(), weight.len(), "weight must match the input width");
    assert_eq!(x.len(), out.len(), "output must match the input width");

    let mut ss: i128 = 0;
    for &v in x {
        ss += (v as i128) * (v as i128); // Q64
    }
    let mean = ss / x.len() as i128 + ((eps as i128) << F); // Q64

    // sqrt of a Q64 value is Q32, which is exactly the scale the divisor needs.
    let rms = isqrt(mean as u128) as i128;
    let rms = if rms == 0 { 1 } else { rms };

    for i in 0..x.len() {
        // Shifting into Q64 before dividing keeps the quotient in Q32 rather than
        // truncating it to zero for the small values that dominate a normalised vector.
        let normed = ((x[i] as i128) << F) / rms; // Q32
        out[i] = ((normed * weight[i] as i128) >> F) as i64;
    }
}

/// Softmax in place, over Q32 logits.
///
/// The maximum is subtracted first, which is not merely a numerical nicety here: the
/// exponential is only defined for non-positive arguments, so this is what makes every
/// call legal.
pub fn softmax(x: &mut [i64]) {
    if x.is_empty() {
        return;
    }
    let max = *x.iter().max().expect("non-empty");

    let mut sum: i128 = 0;
    let mut e = vec![0u64; x.len()];
    for i in 0..x.len() {
        e[i] = exp_q32(x[i] - max);
        sum += e[i] as i128;
    }
    if sum == 0 {
        // Every term underflowed, which can only happen if the spread exceeds what Q32
        // can represent. Uniform is the honest answer; leaving zeros would silently drop
        // the whole attention row.
        let uniform = ONE / x.len() as i64;
        x.iter_mut().for_each(|v| *v = uniform);
        return;
    }
    for i in 0..x.len() {
        x[i] = (((e[i] as i128) << F) / sum) as i64;
    }
}

/// Rotary position embedding, applied in place to one head's vector.
///
/// `cos` and `sin` arrive precomputed in Q32, indexed by `pos * (hs/2)`. They are tabulated
/// rather than derived because integer sine would need its own approximation, its own error
/// budget, and its own agreement problem between implementations — where a table is simply
/// data, generated once and hashed alongside the weights.
///
/// Pairs are `(j, j + hs/2)`, the layout HuggingFace uses. The companion project converts
/// checkpoints into llama2.c's interleaved `(2j, 2j+1)` order instead; matching the source
/// convention here avoids the permutation entirely.
pub fn rope(v: &mut [i64], cos: &[i64], sin: &[i64]) {
    let hs = v.len();
    let half = hs / 2;
    assert_eq!(cos.len(), half, "cos table must be hs/2 wide");
    assert_eq!(sin.len(), half, "sin table must be hs/2 wide");

    for j in 0..half {
        let a = v[j] as i128;
        let b = v[j + half] as i128;
        let c = cos[j] as i128;
        let s = sin[j] as i128;
        v[j] = ((a * c - b * s) >> F) as i64;
        v[j + half] = ((a * s + b * c) >> F) as i64;
    }
}

/// SwiGLU: `silu(gate) * up`, elementwise.
pub fn swiglu(gate: &[i64], up: &[i64], out: &mut [i64]) {
    assert_eq!(gate.len(), up.len(), "gate and up must match");
    assert_eq!(gate.len(), out.len(), "output must match");
    for i in 0..gate.len() {
        let g = silu(gate[i]);
        out[i] = ((g as i128 * up[i] as i128) >> F) as i64;
    }
}

/// Elementwise addition, for the residual stream.
pub fn add_into(dst: &mut [i64], src: &[i64]) {
    assert_eq!(dst.len(), src.len(), "residual widths must match");
    for i in 0..dst.len() {
        dst[i] = dst[i].saturating_add(src[i]);
    }
}

/// Scaled dot product of two Q32 vectors, returning Q32.
///
/// `scale` is Q32 and multiplies the result — for attention it carries `1/sqrt(head_dim)`.
/// [`scaled_dot`] against a cache held at Q16 in `i32`.
///
/// The widening is exact and happens per element rather than by materialising a converted
/// slice, so narrowing the cache costs no allocation on the path that reads it every
/// position of every head.
pub fn scaled_dot_kv(a: &[i64], b: &[i32], scale: i64) -> i64 {
    assert_eq!(a.len(), b.len(), "widths must match");
    let mut acc: i128 = 0;
    for i in 0..a.len() {
        acc += (a[i] as i128) * (crate::qwen3::widen_kv(b[i]) as i128); // Q64
    }
    (((acc >> F) * scale as i128) >> F) as i64
}

pub fn scaled_dot(a: &[i64], b: &[i64], scale: i64) -> i64 {
    assert_eq!(a.len(), b.len(), "widths must match");
    let mut acc: i128 = 0;
    for i in 0..a.len() {
        acc += (a[i] as i128) * (b[i] as i128); // Q64
    }
    (((acc >> F) * scale as i128) >> F) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_q(v: &[f64]) -> Vec<i64> {
        v.iter().map(|x| (x * ONE as f64) as i64).collect()
    }
    fn from_q(v: &[i64]) -> Vec<f64> {
        v.iter().map(|x| *x as f64 / ONE as f64).collect()
    }
    fn close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() < tol,
                "{what}[{i}]: got {g}, want {w}, diff {}",
                (g - w).abs()
            );
        }
    }

    #[test]
    fn rmsnorm_matches_float64() {
        let x = [0.3f64, -1.2, 0.75, 2.0, -0.4, 0.9, -1.5, 0.05];
        let w = [1.0f64, 0.5, 1.5, 0.8, 1.2, 1.0, 0.9, 1.1];

        let eps = 1e-6f64;
        let ms: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps;
        let want: Vec<f64> = x.iter().zip(&w).map(|(v, k)| v / ms.sqrt() * k).collect();

        let mut out = vec![0i64; x.len()];
        rmsnorm(&to_q(&x), &to_q(&w), (eps * ONE as f64) as i64, &mut out);
        close(&from_q(&out), &want, 1e-6, "rmsnorm");
    }

    #[test]
    fn softmax_matches_float64_and_sums_to_one() {
        let x = [1.0f64, -0.5, 3.0, 0.25, -2.0];
        let max = x.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
        let s: f64 = e.iter().sum();
        let want: Vec<f64> = e.iter().map(|v| v / s).collect();

        let mut got = to_q(&x);
        softmax(&mut got);
        close(&from_q(&got), &want, 1e-6, "softmax");

        let total: f64 = from_q(&got).iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "softmax summed to {total}");
    }

    /// A row whose spread exceeds Q32's exponential range must not silently become zeros,
    /// which would drop an entire attention row without any error surfacing.
    #[test]
    fn softmax_survives_an_extreme_spread() {
        let mut x = vec![0i64; 4];
        x[0] = i64::MAX / 2;
        x[1] = i64::MIN / 2;
        softmax(&mut x);
        let total: f64 = from_q(&x).iter().sum();
        assert!(
            total > 0.9 && total < 1.1,
            "degenerate softmax summed to {total}"
        );
    }

    #[test]
    fn rope_matches_the_rotation_it_encodes() {
        let hs = 8usize;
        let v: Vec<f64> = (0..hs).map(|i| 0.1 * (i as f64 + 1.0)).collect();
        let angles: Vec<f64> = (0..hs / 2).map(|j| 0.3 * (j as f64 + 1.0)).collect();
        let cos: Vec<f64> = angles.iter().map(|a| a.cos()).collect();
        let sin: Vec<f64> = angles.iter().map(|a| a.sin()).collect();

        let mut want = v.clone();
        for j in 0..hs / 2 {
            let (a, b) = (v[j], v[j + hs / 2]);
            want[j] = a * cos[j] - b * sin[j];
            want[j + hs / 2] = a * sin[j] + b * cos[j];
        }

        let mut got = to_q(&v);
        rope(&mut got, &to_q(&cos), &to_q(&sin));
        close(&from_q(&got), &want, 1e-6, "rope");
    }

    /// Rotation preserves length. This catches a swapped sign or a mispaired index, both of
    /// which leave plausible-looking numbers behind.
    #[test]
    fn rope_preserves_norm() {
        let hs = 16usize;
        let v: Vec<f64> = (0..hs)
            .map(|i| ((i * 37 % 11) as f64 - 5.0) / 8.0)
            .collect();
        let angles: Vec<f64> = (0..hs / 2).map(|j| 0.7 * j as f64).collect();
        let before: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();

        let mut got = to_q(&v);
        rope(
            &mut got,
            &to_q(&angles.iter().map(|a| a.cos()).collect::<Vec<_>>()),
            &to_q(&angles.iter().map(|a| a.sin()).collect::<Vec<_>>()),
        );
        let after: f64 = from_q(&got).iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((before - after).abs() < 1e-5, "norm {before} -> {after}");
    }

    #[test]
    fn swiglu_matches_float64() {
        let g = [-2.0f64, -0.5, 0.0, 0.5, 2.0];
        let u = [1.0f64, 2.0, -1.0, 0.5, -0.25];
        let want: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(a, b)| (a / (1.0 + (-a).exp())) * b)
            .collect();

        let mut out = vec![0i64; g.len()];
        swiglu(&to_q(&g), &to_q(&u), &mut out);
        close(&from_q(&out), &want, 1e-6, "swiglu");
    }

    #[test]
    fn scaled_dot_matches_float64() {
        let a = [0.5f64, -1.0, 2.0, 0.25];
        let b = [1.5f64, 0.5, -0.75, 2.0];
        let scale = 0.125f64;
        let want: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum::<f64>() * scale;

        let got = scaled_dot(&to_q(&a), &to_q(&b), (scale * ONE as f64) as i64);
        assert!(
            (got as f64 / ONE as f64 - want).abs() < 1e-6,
            "got {}, want {want}",
            got as f64 / ONE as f64
        );
    }
}
