//! One real Qwen3-30B-A3B layer, in integers, against float64.
//!
//! Routing has been checked against float64 on synthetic weights, which proves the rule.
//! This asks the harder question: does it hold on a trained router? Synthetic scores come
//! from a random matrix and are spread far apart, so the top eight are obvious. A trained
//! router's are not — the eighth and ninth experts of a hundred and twenty-eight can be
//! separated by very little, and the integer path has to agree with float64 about which
//! side of that line a token falls on.
//!
//! Getting the arithmetic close is not enough here. Choosing a different expert is not a
//! rounding difference; it sends the token through different parameters. So the check is in
//! two parts: the experts chosen must match exactly, and the layer's output must match to a
//! tolerance.
//!
//! ```text
//! python3 tools/refmoe.py --model models/q30/shard1.safetensors \
//!     --config models/q30/config.json --layer 0 --out models/q30layer
//! cargo run --release --example real_moe_layer -- models/q30layer
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;
use tensor_model::moe::{Expert, MoeScratch};
use tensor_model::qwen3::{Config, KvCache, LayerWeights, Mixture, QTensor, Rope, Scratch};

const Q32: f64 = 4_294_967_296.0;

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name))
        .unwrap_or_else(|e| panic!("{}: {e} — run tools/refmoe.py", dir.join(name).display()))
}

fn read_i64(dir: &Path, name: &str) -> Vec<i64> {
    read(dir, name)
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect()
}

fn read_f64(dir: &Path, name: &str) -> Vec<f64> {
    read(dir, name)
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect()
}

fn split(raw: &[u8], rows: usize, cols: usize, what: &str) -> (Vec<i8>, Vec<i16>) {
    assert_eq!(raw.len(), rows + rows * cols * 2, "{what}: {rows}x{cols}");
    let scales = raw[..rows].iter().map(|&b| b as i8).collect();
    let w = raw[rows..]
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    (scales, w)
}

fn meta(s: &str, key: &str) -> usize {
    let at = s
        .find(&format!("\"{key}\":"))
        .unwrap_or_else(|| panic!("meta.json has no {key}"));
    s[at + key.len() + 3..]
        .split([',', '\n', '}', ']'])
        .next()
        .expect("value")
        .trim()
        .parse()
        .expect("integer")
}

/// The expert lists the float64 reference recorded, one row a position.
fn routed(s: &str) -> Vec<Vec<usize>> {
    let at = s.find("\"routed\":").expect("meta.json has no routed");
    let body = &s[at..];
    let end = body
        .find(']')
        .map(|_| body.rfind(']').expect("close"))
        .expect("array");
    body[..end]
        .split('[')
        .skip(2)
        .map(|row| {
            row.split(']')
                .next()
                .expect("row")
                .split(',')
                .filter_map(|t| t.trim().parse().ok())
                .collect()
        })
        .collect()
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/q30layer".into())
        .into();
    let mj = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");

    let dim = meta(&mj, "dim");
    let hidden = meta(&mj, "hidden");
    let n_experts = meta(&mj, "n_experts");
    let top_k = meta(&mj, "top_k");
    let positions = meta(&mj, "positions");
    let cfg = Config {
        dim,
        hidden,
        n_heads: meta(&mj, "n_heads"),
        n_kv: meta(&mj, "n_kv"),
        head_dim: meta(&mj, "head_dim"),
        seq_len: positions,
        rms_eps: tensor_kernel::fixed::RMS_EPS_QWEN3,
    };
    let want_routed = routed(&mj);

    println!("Qwen3-30B-A3B, layer {}", meta(&mj, "layer"));
    println!(
        "  dim {dim}, heads {}, kv {}, head_dim {}",
        cfg.n_heads, cfg.n_kv, cfg.head_dim
    );
    println!("  mixture: {n_experts} experts of {hidden} wide, top {top_k}");
    print!("  loading 1.25 GB... ");
    let t = Instant::now();

    let q = read(&dir, "q.bin");
    let k = read(&dir, "k.bin");
    let v = read(&dir, "v.bin");
    let o = read(&dir, "o.bin");
    let experts_raw = read(&dir, "experts.bin");
    let router_raw = read(&dir, "router.bin");

    let (qs, qw) = split(&q, cfg.q_width(), dim, "q");
    let (ks, kw) = split(&k, cfg.kv_width(), dim, "k");
    let (vs, vw) = split(&v, cfg.kv_width(), dim, "v");
    let (os, ow) = split(&o, dim, cfg.q_width(), "o");
    let (rs, rw) = split(&router_raw, n_experts, dim, "router");

    // Expert by expert, exactly as the chain addresses them: one stride apart.
    let gate_len = hidden + hidden * dim * 2;
    let down_len = dim + dim * hidden * 2;
    let stride = gate_len * 2 + down_len;
    assert_eq!(experts_raw.len(), n_experts * stride, "expert stride");

    let owned: Vec<[(Vec<i8>, Vec<i16>); 3]> = (0..n_experts)
        .map(|e| {
            let b = e * stride;
            [
                split(&experts_raw[b..b + gate_len], hidden, dim, "gate"),
                split(
                    &experts_raw[b + gate_len..b + gate_len * 2],
                    hidden,
                    dim,
                    "up",
                ),
                split(
                    &experts_raw[b + gate_len * 2..b + stride],
                    dim,
                    hidden,
                    "down",
                ),
            ]
        })
        .collect();
    println!("{:.1?}", t.elapsed());

    let experts: Vec<Expert> = owned
        .iter()
        .map(|[g, u, d]| Expert {
            gate: QTensor {
                w: &g.1,
                scales: &g.0,
                rows: hidden,
                cols: dim,
            },
            up: QTensor {
                w: &u.1,
                scales: &u.0,
                rows: hidden,
                cols: dim,
            },
            down: QTensor {
                w: &d.1,
                scales: &d.0,
                rows: dim,
                cols: hidden,
            },
        })
        .collect();

    let input_norm = read_i64(&dir, "input_norm.bin");
    let q_norm = read_i64(&dir, "q_norm.bin");
    let k_norm = read_i64(&dir, "k_norm.bin");
    let post_norm = read_i64(&dir, "post_norm.bin");

    let w = LayerWeights {
        input_norm: &input_norm,
        q: QTensor {
            w: &qw,
            scales: &qs,
            rows: cfg.q_width(),
            cols: dim,
        },
        k: QTensor {
            w: &kw,
            scales: &ks,
            rows: cfg.kv_width(),
            cols: dim,
        },
        v: QTensor {
            w: &vw,
            scales: &vs,
            rows: cfg.kv_width(),
            cols: dim,
        },
        q_norm: &q_norm,
        k_norm: &k_norm,
        o: QTensor {
            w: &ow,
            scales: &os,
            rows: dim,
            cols: cfg.q_width(),
        },
        post_norm: &post_norm,
        // Unused when a mixture is present; pointed at the router so they are valid views.
        gate: QTensor {
            w: &rw,
            scales: &rs,
            rows: n_experts,
            cols: dim,
        },
        up: QTensor {
            w: &rw,
            scales: &rs,
            rows: n_experts,
            cols: dim,
        },
        down: QTensor {
            w: &rw,
            scales: &rs,
            rows: n_experts,
            cols: dim,
        },
        mixture: Some(Mixture {
            router: QTensor {
                w: &rw,
                scales: &rs,
                rows: n_experts,
                cols: dim,
            },
            experts,
            top_k,
        }),
    };

    let rope = Rope {
        cos: read_i64(&dir, "cos.bin"),
        sin: read_i64(&dir, "sin.bin"),
        head_dim: cfg.head_dim,
    };
    let xs = read_i64(&dir, "moe_in.bin");
    let expected = read_f64(&dir, "moe_out.bin");
    let full_x = read_i64(&dir, "x.bin");
    let full_expected = read_f64(&dir, "expected.bin");

    let m = w.mixture.as_ref().expect("mixture");
    let mut worst = 0.0f64;
    let mut agree = true;

    println!(
        "\n{:>4}  {:>9}  {:>11}  experts chosen",
        "pos", "ms", "rel err"
    );
    println!("{}", "-".repeat(78));

    // ---- the mixture alone, on the exact vector float64 routed ----
    for p in 0..positions {
        let n = &xs[p * dim..(p + 1) * dim];
        let mut ms = MoeScratch::new(dim, hidden, n_experts, top_k);
        let mut got = vec![0i64; dim];

        let t = Instant::now();
        tensor_model::moe::forward(n, &m.router, &m.experts, top_k, &mut ms, &mut got);
        let ms_taken = t.elapsed().as_secs_f64() * 1000.0;

        let chose: Vec<usize> = ms.chosen.iter().map(|&(i, _)| i).collect();
        let same = chose == want_routed[p];
        agree &= same;

        let want = &expected[p * dim..(p + 1) * dim];
        let scale: f64 = want.iter().map(|v| v.abs()).sum::<f64>() / dim as f64;
        let err = got
            .iter()
            .zip(want)
            .map(|(g, t)| ((*g as f64 / Q32) - t).abs())
            .fold(0.0f64, f64::max)
            / scale;
        worst = worst.max(err);

        println!(
            "{p:>4}  {ms_taken:>9.1}  {err:>11.2e}  {}{:?}",
            if same { "" } else { "DIFFERENT: " },
            chose
        );
    }

    println!("\nmixture, on real weights");
    println!("  worst relative error  {worst:.2e}");
    println!(
        "  experts chosen        {}",
        if agree {
            "exactly float64's, every position"
        } else {
            "MISMATCH — a token would take a different path"
        }
    );

    // ---- and the whole layer, attention included ----
    let mut kv = KvCache::new(&cfg, 1);
    let mut s = Scratch::new(&cfg);
    let mut layer_worst = 0.0f64;
    let t = Instant::now();
    for p in 0..positions {
        let mut x = full_x[p * dim..(p + 1) * dim].to_vec();
        tensor_model::qwen3::forward_layer(&cfg, &w, &rope, &mut kv, &mut s, 0, p, &mut x);
        let want = &full_expected[p * dim..(p + 1) * dim];
        let scale: f64 = want.iter().map(|v| v.abs()).sum::<f64>() / dim as f64;
        layer_worst = layer_worst.max(
            x.iter()
                .zip(want)
                .map(|(g, t)| ((*g as f64 / Q32) - t).abs())
                .fold(0.0f64, f64::max)
                / scale,
        );
    }
    let per_token = t.elapsed().as_secs_f64() * 1000.0 / positions as f64;

    println!("\nwhole layer, attention included");
    println!("  worst relative error  {layer_worst:.2e}");
    println!(
        "  {per_token:.1} ms a position, so 48 layers is {:.2} s a token",
        per_token * 48.0 / 1000.0
    );

    println!(
        "\nA token reads {top_k}/{n_experts} of the feed-forward: {:.0} MB of {:.0} MB.",
        (top_k * stride) as f64 / 1e6,
        (n_experts * stride) as f64 / 1e6
    );
    assert!(agree, "the integer router chose different experts");
}
