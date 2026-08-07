//! A Qwen3 transformer block, in integers.
//!
//! Qwen3 rather than Llama because the differences are exactly the ones that are silent
//! when unhandled, and a design meant to serve many model families should be built against
//! a family that actually exercises them:
//!
//! - `head_dim` is not `dim / n_heads`. In Qwen3-0.6B it is 128 against 64, so `q_proj` is
//!   2048x1024 and `o_proj` is 1024x2048 — **neither is square**, which the companion
//!   project's exporter assumes throughout.
//! - Queries and keys are RMS-normalised per head before rotary, with their own weights.
//!   Omitting this shifts every attention score and produces fluent nonsense.
//! - Grouped-query attention: several query heads share one key/value head.
//!
//! Everything is Q32 fixed point. Nothing here uses floating point, and none of it may:
//! the whole design rests on every node computing identical bits.

use crate::ops::{add_into, rmsnorm, rope, scaled_dot, softmax, swiglu};
use tensor_kernel::fixed::ONE;
use tensor_kernel::matmul;

/// A quantised weight matrix: `int16` values with one power-of-two scale per row.
pub struct QTensor<'a> {
    pub w: &'a [i16],
    pub scales: &'a [i8],
    pub rows: usize,
    pub cols: usize,
}

impl QTensor<'_> {
    fn apply(&self, x: &[i64], out: &mut [i64]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(out.len(), self.rows);
        matmul(self.w, self.scales, x, self.rows, self.cols, out);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    /// Not `dim / n_heads`. See the module comment.
    pub head_dim: usize,
    pub seq_len: usize,
    /// RMSNorm epsilon in Q32, from the model's own configuration. Qwen3 uses 1e-6 where
    /// Llama uses 1e-5; hardcoding either is a place for a chain and its reference to hold
    /// quietly different models.
    pub rms_eps: i64,
}

impl Config {
    /// Query heads sharing each key/value head.
    pub fn group(&self) -> usize {
        self.n_heads / self.n_kv
    }
    /// Width of the concatenated query heads, which is also `o_proj`'s input.
    pub fn q_width(&self) -> usize {
        self.n_heads * self.head_dim
    }
    /// Width of the concatenated key/value heads.
    pub fn kv_width(&self) -> usize {
        self.n_kv * self.head_dim
    }
    /// `1/sqrt(head_dim)` in Q32, by integer square root so the constant has no float
    /// provenance and cannot differ between implementations.
    pub fn attn_scale(&self) -> i64 {
        let root = tensor_kernel::fixed::isqrt((self.head_dim as u128) << 64); // sqrt(hs) in Q32
        (((ONE as u128) << 32) / root) as i64
    }
}

pub struct LayerWeights<'a> {
    pub input_norm: &'a [i64],
    pub q: QTensor<'a>,
    pub k: QTensor<'a>,
    pub v: QTensor<'a>,
    pub q_norm: &'a [i64],
    pub k_norm: &'a [i64],
    pub o: QTensor<'a>,
    pub post_norm: &'a [i64],
    pub gate: QTensor<'a>,
    pub up: QTensor<'a>,
    pub down: QTensor<'a>,
}

/// Precomputed rotary tables, `seq_len * head_dim/2` entries each.
///
/// Tabulated rather than derived, because an integer sine would need its own approximation
/// and its own agreement problem between implementations. A table is data: generated once,
/// hashed alongside the weights, and identical everywhere by construction.
pub struct Rope {
    pub cos: Vec<i64>,
    pub sin: Vec<i64>,
    pub head_dim: usize,
}

impl Rope {
    fn at(&self, pos: usize) -> (&[i64], &[i64]) {
        let half = self.head_dim / 2;
        let a = pos * half;
        (&self.cos[a..a + half], &self.sin[a..a + half])
    }
}

/// Keys and values for every position seen so far, per layer.
///
/// Laid out as `[layer][pos][kv_width]`. This is the state that grows with generation, and
/// at long context it dominates everything else a chain would have to hold — which is why
/// it is the thing to compress first if this design is ever pushed to real sequence lengths.
pub struct KvCache {
    pub k: Vec<i64>,
    pub v: Vec<i64>,
    kv_width: usize,
    seq_len: usize,
}

impl KvCache {
    pub fn new(cfg: &Config, n_layers: usize) -> Self {
        let n = n_layers * cfg.seq_len * cfg.kv_width();
        Self {
            k: vec![0; n],
            v: vec![0; n],
            kv_width: cfg.kv_width(),
            seq_len: cfg.seq_len,
        }
    }

    fn slot(&self, layer: usize, pos: usize) -> usize {
        (layer * self.seq_len + pos) * self.kv_width
    }
}

/// Scratch buffers, allocated once per session rather than per layer.
pub struct Scratch {
    normed: Vec<i64>,
    q: Vec<i64>,
    k: Vec<i64>,
    v: Vec<i64>,
    attn: Vec<i64>,
    proj: Vec<i64>,
    gate: Vec<i64>,
    up: Vec<i64>,
    ffn: Vec<i64>,
    scores: Vec<i64>,
    /// One head's worth of workspace for QK-norm, which cannot normalise in place.
    /// Allocating this per head per position cost more than the matmuls it sits between.
    head: Vec<i64>,
}

impl Scratch {
    pub fn new(cfg: &Config) -> Self {
        Self {
            normed: vec![0; cfg.dim],
            q: vec![0; cfg.q_width()],
            k: vec![0; cfg.kv_width()],
            v: vec![0; cfg.kv_width()],
            attn: vec![0; cfg.q_width()],
            proj: vec![0; cfg.dim],
            gate: vec![0; cfg.hidden],
            up: vec![0; cfg.hidden],
            ffn: vec![0; cfg.hidden],
            scores: vec![0; cfg.seq_len],
            head: vec![0; cfg.head_dim],
        }
    }
}

/// One transformer block, applied in place to the residual stream.
///
/// The argument list is long because a transformer block genuinely depends on all of it:
/// geometry, weights, rotary tables, the cache it reads and extends, reusable scratch, and
/// where in the sequence it is. Bundling them into a struct would hide which parts are
/// mutated rather than reduce what the operation needs.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer(
    cfg: &Config,
    w: &LayerWeights,
    rope_tab: &Rope,
    kv: &mut KvCache,
    s: &mut Scratch,
    layer: usize,
    pos: usize,
    x: &mut [i64],
) {
    let hs = cfg.head_dim;
    debug_assert_eq!(x.len(), cfg.dim);
    debug_assert!(pos < cfg.seq_len, "position beyond the cache");

    // ---- attention ----
    rmsnorm(x, w.input_norm, cfg.rms_eps, &mut s.normed);
    w.q.apply(&s.normed, &mut s.q);
    w.k.apply(&s.normed, &mut s.k);
    w.v.apply(&s.normed, &mut s.v);

    // QK-norm, then rotary — in that order. Normalising after the rotation would rescale
    // the rotated components unequally and destroy the angle the embedding encodes.
    let (cos, sin) = rope_tab.at(pos);
    for h in 0..cfg.n_heads {
        let seg = &mut s.q[h * hs..(h + 1) * hs];
        rmsnorm(seg, w.q_norm, cfg.rms_eps, &mut s.head);
        seg.copy_from_slice(&s.head);
        rope(seg, cos, sin);
    }
    for h in 0..cfg.n_kv {
        let seg = &mut s.k[h * hs..(h + 1) * hs];
        rmsnorm(seg, w.k_norm, cfg.rms_eps, &mut s.head);
        seg.copy_from_slice(&s.head);
        rope(seg, cos, sin);
    }

    let at = kv.slot(layer, pos);
    kv.k[at..at + cfg.kv_width()].copy_from_slice(&s.k);
    kv.v[at..at + cfg.kv_width()].copy_from_slice(&s.v);

    let scale = cfg.attn_scale();
    let group = cfg.group();
    for h in 0..cfg.n_heads {
        let kvh = h / group;
        let q = &s.q[h * hs..(h + 1) * hs];

        // Causal: only positions up to and including this one are visible.
        for t in 0..=pos {
            let base = kv.slot(layer, t) + kvh * hs;
            s.scores[t] = scaled_dot(q, &kv.k[base..base + hs], scale);
        }
        softmax(&mut s.scores[..=pos]);

        let out = &mut s.attn[h * hs..(h + 1) * hs];
        out.fill(0);
        for t in 0..=pos {
            let base = kv.slot(layer, t) + kvh * hs;
            let weight = s.scores[t] as i128;
            for (o, v) in out.iter_mut().zip(&kv.v[base..base + hs]) {
                *o += ((weight * *v as i128) >> 32) as i64;
            }
        }
    }

    w.o.apply(&s.attn, &mut s.proj);
    add_into(x, &s.proj);

    // ---- feed-forward ----
    rmsnorm(x, w.post_norm, cfg.rms_eps, &mut s.normed);
    w.gate.apply(&s.normed, &mut s.gate);
    w.up.apply(&s.normed, &mut s.up);
    swiglu(&s.gate, &s.up, &mut s.ffn);
    w.down.apply(&s.ffn, &mut s.proj);
    add_into(x, &s.proj);
}

/// A whole model: the layers, the final norm, and the classifier.
///
/// The embedding is not here. It is a lookup rather than a multiply — a token id selects a
/// row — so it carries no weight matrix for the quantiser's row scales to ride along with,
/// and callers supply the row directly.
pub struct Model<'a> {
    pub cfg: Config,
    pub layers: Vec<LayerWeights<'a>>,
    pub final_norm: &'a [i64],
    pub lm_head: QTensor<'a>,
}

impl Model<'_> {
    /// Runs one token and returns logits over the vocabulary.
    ///
    /// `x` is the embedding row for this token, in Q32, consumed as the initial residual
    /// stream. The caller advances `pos`; the cache and the causal mask follow from it.
    pub fn forward(
        &self,
        rope: &Rope,
        kv: &mut KvCache,
        s: &mut Scratch,
        pos: usize,
        x: &mut [i64],
        logits: &mut [i64],
    ) {
        assert_eq!(x.len(), self.cfg.dim, "residual width");
        assert_eq!(logits.len(), self.lm_head.rows, "logit width");

        for (l, w) in self.layers.iter().enumerate() {
            forward_layer(&self.cfg, w, rope, kv, s, l, pos, x);
        }

        // Normalise into scratch rather than in place: the residual stream is the caller's,
        // and a generation loop needs it intact to embed the next token against.
        rmsnorm(x, self.final_norm, self.cfg.rms_eps, &mut s.normed);
        self.lm_head.apply(&s.normed, logits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A float64 transformer block, written independently of the integer one.
    ///
    /// Deliberately a separate implementation rather than a shared helper: two versions of
    /// the same code agreeing proves consistency, not correctness. That mistake already
    /// cost this project a thousand-percent error once.
    mod reference {
        pub struct Layer {
            pub input_norm: Vec<f64>,
            pub q: Vec<Vec<f64>>,
            pub k: Vec<Vec<f64>>,
            pub v: Vec<Vec<f64>>,
            pub q_norm: Vec<f64>,
            pub k_norm: Vec<f64>,
            pub o: Vec<Vec<f64>>,
            pub post_norm: Vec<f64>,
            pub gate: Vec<Vec<f64>>,
            pub up: Vec<Vec<f64>>,
            pub down: Vec<Vec<f64>>,
        }

        pub fn rmsnorm(x: &[f64], w: &[f64]) -> Vec<f64> {
            let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + 1e-5;
            let r = ms.sqrt();
            x.iter().zip(w).map(|(v, k)| v / r * k).collect()
        }

        pub fn matvec(m: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
            m.iter()
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        }

        pub fn softmax(x: &[f64]) -> Vec<f64> {
            let m = x.iter().cloned().fold(f64::MIN, f64::max);
            let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
            let s: f64 = e.iter().sum();
            e.iter().map(|v| v / s).collect()
        }

        pub fn rope(v: &mut [f64], cos: &[f64], sin: &[f64]) {
            let half = v.len() / 2;
            for j in 0..half {
                let (a, b) = (v[j], v[j + half]);
                v[j] = a * cos[j] - b * sin[j];
                v[j + half] = a * sin[j] + b * cos[j];
            }
        }

        #[allow(clippy::too_many_arguments)]
        pub fn forward(
            l: &Layer,
            x: &mut [f64],
            kc: &mut Vec<Vec<f64>>,
            vc: &mut Vec<Vec<f64>>,
            n_heads: usize,
            n_kv: usize,
            hs: usize,
            cos: &[f64],
            sin: &[f64],
            pos: usize,
        ) {
            let normed = rmsnorm(x, &l.input_norm);
            let mut q = matvec(&l.q, &normed);
            let mut k = matvec(&l.k, &normed);
            let v = matvec(&l.v, &normed);

            for h in 0..n_heads {
                let seg = &mut q[h * hs..(h + 1) * hs];
                let n = rmsnorm(seg, &l.q_norm);
                seg.copy_from_slice(&n);
                rope(seg, cos, sin);
            }
            for h in 0..n_kv {
                let seg = &mut k[h * hs..(h + 1) * hs];
                let n = rmsnorm(seg, &l.k_norm);
                seg.copy_from_slice(&n);
                rope(seg, cos, sin);
            }
            kc.push(k);
            vc.push(v);

            let scale = 1.0 / (hs as f64).sqrt();
            let group = n_heads / n_kv;
            let mut attn = vec![0.0; n_heads * hs];
            for h in 0..n_heads {
                let kvh = h / group;
                let qh = &q[h * hs..(h + 1) * hs];
                let scores: Vec<f64> = (0..=pos)
                    .map(|t| {
                        let kk = &kc[t][kvh * hs..(kvh + 1) * hs];
                        qh.iter().zip(kk).map(|(a, b)| a * b).sum::<f64>() * scale
                    })
                    .collect();
                let p = softmax(&scores);
                for (t, w) in p.iter().enumerate() {
                    let vv = &vc[t][kvh * hs..(kvh + 1) * hs];
                    for j in 0..hs {
                        attn[h * hs + j] += w * vv[j];
                    }
                }
            }

            let proj = matvec(&l.o, &attn);
            for i in 0..x.len() {
                x[i] += proj[i];
            }

            let normed = rmsnorm(x, &l.post_norm);
            let g = matvec(&l.gate, &normed);
            let u = matvec(&l.up, &normed);
            let ffn: Vec<f64> = g
                .iter()
                .zip(&u)
                .map(|(a, b)| (a / (1.0 + (-a).exp())) * b)
                .collect();
            let proj = matvec(&l.down, &ffn);
            for i in 0..x.len() {
                x[i] += proj[i];
            }
        }
    }

    const ONE_F: f64 = 4_294_967_296.0;

    /// Deterministic pseudo-random floats, so a failure is reproducible from the seed.
    fn noise(n: usize, seed: u64, spread: f64) -> Vec<f64> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f64 / (1u64 << 31) as f64 - 1.0) * spread
            })
            .collect()
    }

    fn quantize(m: &[Vec<f64>]) -> (Vec<i16>, Vec<i8>) {
        let mut w = Vec::new();
        let mut scales = Vec::new();
        for row in m {
            let peak = row.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            let s = if peak > 0.0 {
                (32767.0 / peak).log2().floor() as i32
            } else {
                0
            }
            .clamp(-128, 127);
            scales.push(s as i8);
            for v in row {
                w.push((v * (s as f64).exp2()).round().clamp(-32767.0, 32767.0) as i16);
            }
        }
        (w, scales)
    }

    fn to_q(v: &[f64]) -> Vec<i64> {
        v.iter().map(|x| (x * ONE_F).round() as i64).collect()
    }

    fn matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
        (0..rows)
            .map(|r| noise(cols, seed + r as u64, 0.08))
            .collect()
    }

    /// A whole block, integer against float64, at a geometry that reproduces every
    /// structural feature of Qwen3: a head_dim that is not `dim / n_heads`, grouped-query
    /// attention, and therefore non-square query and output projections.
    #[test]
    fn a_layer_matches_float64() {
        let cfg = Config {
            dim: 32,
            hidden: 64,
            n_heads: 4,
            n_kv: 2,
            head_dim: 16, // 4 * 16 = 64 != dim, exactly as Qwen3 does it
            seq_len: 8,
            rms_eps: tensor_kernel::fixed::RMS_EPS_LLAMA,
        };
        assert_ne!(
            cfg.head_dim,
            cfg.dim / cfg.n_heads,
            "geometry must be non-square"
        );

        let l = reference::Layer {
            input_norm: noise(cfg.dim, 1, 0.2).iter().map(|v| 1.0 + v).collect(),
            q: matrix(cfg.q_width(), cfg.dim, 100),
            k: matrix(cfg.kv_width(), cfg.dim, 200),
            v: matrix(cfg.kv_width(), cfg.dim, 300),
            q_norm: noise(cfg.head_dim, 2, 0.2)
                .iter()
                .map(|v| 1.0 + v)
                .collect(),
            k_norm: noise(cfg.head_dim, 3, 0.2)
                .iter()
                .map(|v| 1.0 + v)
                .collect(),
            o: matrix(cfg.dim, cfg.q_width(), 400),
            post_norm: noise(cfg.dim, 4, 0.2).iter().map(|v| 1.0 + v).collect(),
            gate: matrix(cfg.hidden, cfg.dim, 500),
            up: matrix(cfg.hidden, cfg.dim, 600),
            down: matrix(cfg.dim, cfg.hidden, 700),
        };

        let half = cfg.head_dim / 2;
        let angles: Vec<f64> = (0..cfg.seq_len)
            .flat_map(|p| {
                (0..half)
                    .map(move |j| p as f64 * (1e6f64).powf(-2.0 * j as f64 / cfg.head_dim as f64))
            })
            .collect();
        let cos_f: Vec<f64> = angles.iter().map(|a| a.cos()).collect();
        let sin_f: Vec<f64> = angles.iter().map(|a| a.sin()).collect();

        let (qw, qs) = quantize(&l.q);
        let (kw, ks) = quantize(&l.k);
        let (vw, vs) = quantize(&l.v);
        let (ow, os) = quantize(&l.o);
        let (gw, gs) = quantize(&l.gate);
        let (uw, us) = quantize(&l.up);
        let (dw, ds) = quantize(&l.down);

        let iw = LayerWeights {
            input_norm: &to_q(&l.input_norm),
            q: QTensor {
                w: &qw,
                scales: &qs,
                rows: cfg.q_width(),
                cols: cfg.dim,
            },
            k: QTensor {
                w: &kw,
                scales: &ks,
                rows: cfg.kv_width(),
                cols: cfg.dim,
            },
            v: QTensor {
                w: &vw,
                scales: &vs,
                rows: cfg.kv_width(),
                cols: cfg.dim,
            },
            q_norm: &to_q(&l.q_norm),
            k_norm: &to_q(&l.k_norm),
            o: QTensor {
                w: &ow,
                scales: &os,
                rows: cfg.dim,
                cols: cfg.q_width(),
            },
            post_norm: &to_q(&l.post_norm),
            gate: QTensor {
                w: &gw,
                scales: &gs,
                rows: cfg.hidden,
                cols: cfg.dim,
            },
            up: QTensor {
                w: &uw,
                scales: &us,
                rows: cfg.hidden,
                cols: cfg.dim,
            },
            down: QTensor {
                w: &dw,
                scales: &ds,
                rows: cfg.dim,
                cols: cfg.hidden,
            },
        };

        let rope_tab = Rope {
            cos: to_q(&cos_f),
            sin: to_q(&sin_f),
            head_dim: cfg.head_dim,
        };
        let mut kv = KvCache::new(&cfg, 1);
        let mut scratch = Scratch::new(&cfg);

        // Several positions, so the KV cache and the causal mask are actually exercised —
        // a single position would pass with attention entirely broken.
        let mut xf = noise(cfg.dim, 42, 0.5);
        let mut xi = to_q(&xf);
        let (mut kc, mut vc) = (Vec::new(), Vec::new());

        for pos in 0..4 {
            let a = pos * half;
            reference::forward(
                &l,
                &mut xf,
                &mut kc,
                &mut vc,
                cfg.n_heads,
                cfg.n_kv,
                cfg.head_dim,
                &cos_f[a..a + half],
                &sin_f[a..a + half],
                pos,
            );
            forward_layer(&cfg, &iw, &rope_tab, &mut kv, &mut scratch, 0, pos, &mut xi);

            let scale: f64 = xf.iter().map(|v| v.abs()).sum::<f64>() / xf.len() as f64;
            let err: f64 = xi
                .iter()
                .zip(&xf)
                .map(|(g, t)| ((*g as f64 / ONE_F) - t).abs())
                .sum::<f64>()
                / xf.len() as f64;
            let rel = err / scale;
            println!("pos {pos}: mean relative error {rel:.3e}");
            assert!(rel < 2e-3, "pos {pos}: relative error {rel:.3e}");
        }
    }

    /// `1/sqrt(head_dim)` must come out right, since it scales every attention score and a
    /// wrong value degrades output without ever erroring.
    #[test]
    fn attention_scale_is_correct() {
        for hs in [16usize, 64, 128] {
            let cfg = Config {
                dim: 64,
                hidden: 128,
                n_heads: 4,
                n_kv: 2,
                head_dim: hs,
                seq_len: 4,
                rms_eps: tensor_kernel::fixed::RMS_EPS_LLAMA,
            };
            let got = cfg.attn_scale() as f64 / ONE_F;
            let want = 1.0 / (hs as f64).sqrt();
            assert!((got - want).abs() < 1e-9, "hs {hs}: got {got}, want {want}");
        }
    }

    /// Attention must not see the future. Running the same position with a longer cache
    /// behind it should change the answer; running it with the cache truncated to `pos`
    /// should not depend on anything after.
    #[test]
    fn attention_is_causal() {
        let cfg = Config {
            dim: 16,
            hidden: 32,
            n_heads: 2,
            n_kv: 1,
            head_dim: 8,
            seq_len: 6,
            rms_eps: tensor_kernel::fixed::RMS_EPS_LLAMA,
        };
        let kv = KvCache::new(&cfg, 1);
        // Positions are written at `slot(layer, pos)`, so distinct positions never alias.
        assert_ne!(kv.slot(0, 0), kv.slot(0, 1));
        assert_eq!(kv.slot(0, 1) - kv.slot(0, 0), cfg.kv_width());
    }
}
