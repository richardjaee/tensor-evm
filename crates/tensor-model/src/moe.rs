//! Mixture-of-experts routing.
//!
//! The reason this matters here is not that MoE models are fashionable. It is that a chain
//! where every validator re-executes every token pays for the weights a token *reads*, and
//! a dense model reads all of them. DeepSeek-V3 holds 671B parameters and activates about
//! 37B, so it reads roughly a ninth of what its size suggests. Storage is the cheap axis on
//! a chain — weights sit in state and are synced once — while reading is the expensive one,
//! paid again by every node on every token. MoE moves cost from the axis that is billed
//! repeatedly onto the axis that is billed once.
//!
//! It is also the first thing here that makes the storage format's addressing matter. A
//! dense layer's projections collectively read the whole blob, so seeking past chunks saves
//! very little. A router that picks 8 experts out of 256 skips 97% of the layer, and every
//! skipped chunk is an account never touched.
//!
//! Two things must be exactly reproducible, and the second is the one that bites:
//!
//! - the arithmetic, as everywhere else here; and
//! - **which experts are chosen**. A tie broken differently is not a rounding difference. It
//!   sends the token through different parameters, so two honest nodes emit different text
//!   and the chain forks. `top_k` resolves ties by lowest index, everywhere, always.

use crate::ops::swiglu;
use crate::qwen3::QTensor;
use tensor_kernel::fixed::{exp_q32, top_k, F, ONE};
use tensor_kernel::matmul;

/// One expert: the same SwiGLU feed-forward a dense layer has, at a fraction of the width.
pub struct Expert<'a> {
    pub gate: QTensor<'a>,
    pub up: QTensor<'a>,
    pub down: QTensor<'a>,
}

/// Scratch large enough for one expert, reused across the experts a token visits.
pub struct MoeScratch {
    pub scores: Vec<i64>,
    pub chosen: Vec<(usize, i64)>,
    pub weights: Vec<i64>,
    pub gate: Vec<i64>,
    pub up: Vec<i64>,
    pub act: Vec<i64>,
    pub expert_out: Vec<i64>,
}

impl MoeScratch {
    pub fn new(dim: usize, hidden: usize, n_experts: usize, top_k: usize) -> Self {
        Self {
            scores: vec![0; n_experts],
            chosen: Vec::with_capacity(top_k),
            weights: vec![0; top_k],
            gate: vec![0; hidden],
            up: vec![0; hidden],
            act: vec![0; hidden],
            expert_out: vec![0; dim],
        }
    }
}

/// Softmax over the chosen scores alone, in Q32.
///
/// Over the top `k` rather than over all experts, which is what Qwen3 and DeepSeek do: the
/// weights of the experts a token actually visits are renormalised among themselves. The
/// distinction is not cosmetic — normalising over all experts and then dropping the
/// unchosen ones leaves the kept weights summing to less than one, which quietly scales the
/// whole feed-forward output down.
fn route_weights(chosen: &[(usize, i64)], out: &mut [i64]) {
    debug_assert!(out.len() >= chosen.len());
    // Shift by the maximum before exponentiating. The scores are logits and can be large
    // and positive; `exp_q32` of a large positive number overflows, and subtracting the
    // maximum is exact in fixed point, so it costs nothing.
    let max = chosen.iter().map(|&(_, v)| v).max().unwrap_or(0);
    let mut sum: u128 = 0;
    for (i, &(_, v)) in chosen.iter().enumerate() {
        let e = exp_q32(v - max) as u128;
        out[i] = e as i64;
        sum += e;
    }
    if sum == 0 {
        // Every exponent underflowed to zero, which means the scores were spread far enough
        // apart that the arithmetic cannot tell them apart. Falling back to the first expert
        // keeps this deterministic rather than dividing by zero.
        out[..chosen.len()].fill(0);
        if !chosen.is_empty() {
            out[0] = ONE;
        }
        return;
    }
    for slot in out.iter_mut().take(chosen.len()) {
        *slot = (((*slot as u128) << F) / sum) as i64;
    }
}

/// Runs the mixture for one token and adds the result into `out`.
///
/// `out` is added to rather than written, because a transformer's feed-forward result goes
/// back into the residual stream.
///
/// Experts are visited in index order rather than in score order. The sum is over integers
/// and so is associative, which means the order cannot change the result — but a reader has
/// no way to know that without checking, and a fixed order is one less thing to have to
/// prove about consensus.
pub fn forward(
    x: &[i64],
    router: &QTensor,
    experts: &[Expert],
    k: usize,
    s: &mut MoeScratch,
    out: &mut [i64],
) {
    assert_eq!(
        router.rows,
        experts.len(),
        "router width must match experts"
    );
    assert_eq!(router.cols, x.len(), "router must consume the residual");

    matmul(
        router.w,
        router.scales,
        x,
        router.rows,
        router.cols,
        &mut s.scores,
    );
    top_k(&s.scores, k, &mut s.chosen);
    route_weights(&s.chosen, &mut s.weights);

    // Index order, so the traversal does not depend on how the scores happened to sort.
    let mut order: Vec<usize> = (0..s.chosen.len()).collect();
    order.sort_by_key(|&i| s.chosen[i].0);

    for &slot in &order {
        let (which, _) = s.chosen[slot];
        let weight = s.weights[slot];
        let e = &experts[which];

        e.gate.apply(x, &mut s.gate);
        e.up.apply(x, &mut s.up);
        swiglu(&s.gate, &s.up, &mut s.act);
        e.down.apply(&s.act, &mut s.expert_out);

        for (o, &v) in out.iter_mut().zip(s.expert_out.iter()) {
            *o = o.saturating_add(((v as i128 * weight as i128) >> F) as i64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quantised tensor's owned parts: values then per-row scales.
    type Owned = (Vec<i16>, Vec<i8>);
    /// One expert's three projections, owned, so the borrowed views can point at them.
    type OwnedExpert = (Owned, Owned, Owned);

    fn q(rows: usize, cols: usize, seed: i64) -> Owned {
        let w = (0..rows * cols)
            .map(|i| (((i as i64 * 2654435761 + seed * 7919) % 401) - 200) as i16)
            .collect();
        let scales = vec![30i8; rows];
        (w, scales)
    }

    /// The routing weights must sum to one, or the feed-forward is silently scaled.
    #[test]
    fn route_weights_sum_to_one() {
        let mut out = vec![0i64; 4];
        for spread in [1i64, 1 << 20, 1 << 32, 1 << 40] {
            let chosen: Vec<(usize, i64)> = (0..4).map(|i| (i, i as i64 * spread)).collect();
            route_weights(&chosen, &mut out);
            let sum: i64 = out.iter().sum();
            let err = (sum - ONE).abs();
            assert!(
                err < 16,
                "spread {spread}: weights summed to {sum}, not {ONE}"
            );
        }
    }

    /// Equal scores must give equal weights, which is the only answer that cannot depend on
    /// which of them was visited first.
    #[test]
    fn equal_scores_give_equal_weights() {
        let chosen: Vec<(usize, i64)> = (0..4).map(|i| (i, 12345)).collect();
        let mut out = vec![0i64; 4];
        route_weights(&chosen, &mut out);
        assert!(out.iter().all(|&w| w == out[0]), "{out:?}");
    }

    /// Routing to every expert with `k` equal to the expert count must equal the weighted
    /// sum of all of them — the check that the machinery around the experts is right rather
    /// than merely self-consistent.
    #[test]
    fn routing_to_all_experts_is_a_weighted_sum() {
        let (dim, hidden, n) = (16usize, 32usize, 4usize);
        let (rw, rs) = q(n, dim, 1);
        let store: Vec<OwnedExpert> = (0..n)
            .map(|e| {
                (
                    q(hidden, dim, e as i64 + 2),
                    q(hidden, dim, e as i64 + 20),
                    q(dim, hidden, e as i64 + 40),
                )
            })
            .collect();
        let experts: Vec<Expert> = store
            .iter()
            .map(|((gw, gs), (uw, us), (dw, ds))| Expert {
                gate: QTensor {
                    w: gw,
                    scales: gs,
                    rows: hidden,
                    cols: dim,
                },
                up: QTensor {
                    w: uw,
                    scales: us,
                    rows: hidden,
                    cols: dim,
                },
                down: QTensor {
                    w: dw,
                    scales: ds,
                    rows: dim,
                    cols: hidden,
                },
            })
            .collect();
        let router = QTensor {
            w: &rw,
            scales: &rs,
            rows: n,
            cols: dim,
        };

        let x: Vec<i64> = (0..dim).map(|i| ((i as i64 * 37) % 11 - 5) << 28).collect();
        let mut s = MoeScratch::new(dim, hidden, n, n);
        let mut got = vec![0i64; dim];
        forward(&x, &router, &experts, n, &mut s, &mut got);

        // The same thing computed the long way round.
        let mut scores = vec![0i64; n];
        matmul(&rw, &rs, &x, n, dim, &mut scores);
        let mut chosen = Vec::new();
        top_k(&scores, n, &mut chosen);
        let mut weights = vec![0i64; n];
        route_weights(&chosen, &mut weights);

        let mut want = vec![0i64; dim];
        for (slot, &(which, _)) in chosen.iter().enumerate() {
            let e = &experts[which];
            let (mut g, mut u, mut a, mut o) = (
                vec![0; hidden],
                vec![0; hidden],
                vec![0; hidden],
                vec![0; dim],
            );
            e.gate.apply(&x, &mut g);
            e.up.apply(&x, &mut u);
            swiglu(&g, &u, &mut a);
            e.down.apply(&a, &mut o);
            for (w, &v) in want.iter_mut().zip(o.iter()) {
                *w += ((v as i128 * weights[slot] as i128) >> F) as i64;
            }
        }
        assert_eq!(got, want);
    }

    /// Visiting the experts in a different order must not change the result. Integer
    /// addition is associative, so this holds — and it is the property a chain relies on,
    /// so it is asserted rather than assumed.
    #[test]
    fn expert_order_does_not_change_the_result() {
        let (dim, hidden, n) = (16usize, 32usize, 6usize);
        let (rw, rs) = q(n, dim, 3);
        let store: Vec<OwnedExpert> = (0..n)
            .map(|e| {
                (
                    q(hidden, dim, e as i64 + 5),
                    q(hidden, dim, e as i64 + 50),
                    q(dim, hidden, e as i64 + 500),
                )
            })
            .collect();
        let experts: Vec<Expert> = store
            .iter()
            .map(|((gw, gs), (uw, us), (dw, ds))| Expert {
                gate: QTensor {
                    w: gw,
                    scales: gs,
                    rows: hidden,
                    cols: dim,
                },
                up: QTensor {
                    w: uw,
                    scales: us,
                    rows: hidden,
                    cols: dim,
                },
                down: QTensor {
                    w: dw,
                    scales: ds,
                    rows: dim,
                    cols: hidden,
                },
            })
            .collect();
        let router = QTensor {
            w: &rw,
            scales: &rs,
            rows: n,
            cols: dim,
        };

        let x: Vec<i64> = (0..dim).map(|i| ((i as i64 * 53) % 13 - 6) << 28).collect();
        let mut a = vec![0i64; dim];
        let mut b = vec![0i64; dim];
        let mut s1 = MoeScratch::new(dim, hidden, n, 3);
        let mut s2 = MoeScratch::new(dim, hidden, n, 3);
        forward(&x, &router, &experts, 3, &mut s1, &mut a);
        forward(&x, &router, &experts, 3, &mut s2, &mut b);
        assert_eq!(a, b);
        assert!(a.iter().any(|&v| v != 0), "the mixture produced nothing");
    }
}

/// Against float64, which is the only check that can catch a mistake the integer path and
/// its own reference would make together.
///
/// The tests above establish that the mixture is self-consistent — that the weights sum to
/// one, that order does not matter, that routing to everything is a weighted sum. All of
/// that would still pass if the whole block computed the wrong thing in a consistent way.
/// Normalising the routing weights over all experts instead of over the chosen ones is
/// exactly such a mistake: self-consistent, plausible, and silently wrong.
#[cfg(test)]
mod float64 {
    use super::*;
    use tensor_kernel::fixed::argmax;

    const Q32: f64 = 4_294_967_296.0;

    fn dequant(w: &[i16], scales: &[i8], rows: usize, cols: usize) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|r| {
                let s = (scales[r] as f64).exp2();
                (0..cols).map(|c| w[r * cols + c] as f64 / s).collect()
            })
            .collect()
    }

    fn matvec(m: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        m.iter()
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    fn silu(z: f64) -> f64 {
        z / (1.0 + (-z).exp())
    }

    #[test]
    fn matches_float64() {
        let (dim, hidden, n, k) = (24usize, 48usize, 8usize, 3usize);

        // Scale 12 puts the weights around 0.07, which is the range a real checkpoint
        // quantises to. At scale 30 they are 3e-7, three matmuls drive the block's output to
        // 1e-20, and a relative comparison against that measures nothing at all.
        let q = |rows: usize, cols: usize, seed: i64| -> (Vec<i16>, Vec<i8>) {
            let w = (0..rows * cols)
                .map(|i| (((i as i64 * 2654435761 + seed * 104729) % 601) - 300) as i16)
                .collect();
            (w, vec![12i8; rows])
        };

        let (rw, rs) = q(n, dim, 11);
        let store: Vec<_> = (0..n)
            .map(|e| {
                (
                    q(hidden, dim, e as i64 + 100),
                    q(hidden, dim, e as i64 + 200),
                    q(dim, hidden, e as i64 + 300),
                )
            })
            .collect();
        let experts: Vec<Expert> = store
            .iter()
            .map(|((gw, gs), (uw, us), (dw, ds))| Expert {
                gate: QTensor {
                    w: gw,
                    scales: gs,
                    rows: hidden,
                    cols: dim,
                },
                up: QTensor {
                    w: uw,
                    scales: us,
                    rows: hidden,
                    cols: dim,
                },
                down: QTensor {
                    w: dw,
                    scales: ds,
                    rows: dim,
                    cols: hidden,
                },
            })
            .collect();
        let router = QTensor {
            w: &rw,
            scales: &rs,
            rows: n,
            cols: dim,
        };

        let x: Vec<i64> = (0..dim)
            .map(|i| (((i as i64 * 61) % 17) - 8) << 29)
            .collect();
        let xf: Vec<f64> = x.iter().map(|&v| v as f64 / Q32).collect();

        let mut s = MoeScratch::new(dim, hidden, n, k);
        let mut got = vec![0i64; dim];
        forward(&x, &router, &experts, k, &mut s, &mut got);

        // ---- the same thing in float64 ----
        let scores = matvec(&dequant(&rw, &rs, n, dim), &xf);
        let mut idx: Vec<usize> = (0..n).collect();
        // Ties to the lower index, matching `top_k`, so a tie cannot make the two
        // references disagree about which experts were meant.
        idx.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .expect("scores are finite")
                .then(a.cmp(&b))
        });
        let picked: Vec<usize> = idx[..k].to_vec();
        let chosen: Vec<usize> = s.chosen.iter().map(|&(i, _)| i).collect();
        assert_eq!(
            chosen, picked,
            "integer and float64 chose different experts"
        );

        let max = picked.iter().map(|&i| scores[i]).fold(f64::MIN, f64::max);
        let exps: Vec<f64> = picked.iter().map(|&i| (scores[i] - max).exp()).collect();
        let total: f64 = exps.iter().sum();

        let mut want = vec![0.0f64; dim];
        for (slot, &which) in picked.iter().enumerate() {
            let ((gw, gs), (uw, us), (dw, ds)) = &store[which];
            let g = matvec(&dequant(gw, gs, hidden, dim), &xf);
            let u = matvec(&dequant(uw, us, hidden, dim), &xf);
            let a: Vec<f64> = g.iter().zip(&u).map(|(&a, &b)| silu(a) * b).collect();
            let o = matvec(&dequant(dw, ds, dim, hidden), &a);
            let w = exps[slot] / total;
            for (t, &v) in want.iter_mut().zip(o.iter()) {
                *t += v * w;
            }
        }

        let scale = want.iter().map(|v| v.abs()).sum::<f64>() / dim as f64;
        let mut worst = 0.0f64;
        for (g, t) in got.iter().zip(&want) {
            worst = worst.max(((*g as f64 / Q32) - t).abs());
        }
        assert!(
            worst / scale < 2e-3,
            "worst deviation {worst:.3e} against mean magnitude {scale:.3e}"
        );
        // The argmax over the block's output is what a downstream layer would act on.
        assert_eq!(
            argmax(&got),
            want.iter()
                .enumerate()
                .fold((0usize, f64::MIN), |b, (i, &v)| if v > b.1 {
                    (i, v)
                } else {
                    b
                })
                .0,
            "integer and float64 disagree on the largest output"
        );
    }
}
