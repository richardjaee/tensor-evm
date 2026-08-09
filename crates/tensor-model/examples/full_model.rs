//! The whole of Qwen3-0.6B, in integers, against float64 logits.
//!
//! One layer being right does not make twenty-eight of them right. Errors that cancel at a
//! single position can compound with depth, and the classifier amplifies whatever survives:
//! a logit gap of a few thousandths decides which token comes out. This is the check that
//! the model — not the arithmetic — is reproduced.
//!
//! ```text
//! python3 tools/export_model.py \
//!     --model models/qwen3-0.6b.safetensors \
//!     --config models/config.json --out models/full
//! cargo run --release --example full_model -- models/full
//! ```

use std::path::{Path, PathBuf};
use std::time::Instant;
use tensor_model::qwen3::{Config, KvCache, LayerWeights, Model, QTensor, Rope, Scratch};

const Q32: f64 = 4_294_967_296.0;

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — run tools/export_model.py",
            dir.join(name).display()
        )
    })
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

/// Splits a blob into its row scales and its `int16` values.
fn split(raw: Vec<u8>, rows: usize, cols: usize, what: &str) -> (Vec<i8>, Vec<i16>) {
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
        .split([',', '\n', '}'])
        .next()
        .expect("value")
        .trim()
        .parse()
        .expect("integer")
}

/// Owns every tensor, so the borrowed views in `Model` have something to point at.
struct Store {
    scales: Vec<Vec<i8>>,
    weights: Vec<Vec<i16>>,
    norms: Vec<Vec<i64>>,
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/full".to_string())
        .into();
    let mj = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");

    let n_layers = meta(&mj, "n_layers");
    let vocab = meta(&mj, "vocab");
    let positions = meta(&mj, "positions");
    let cfg = Config {
        dim: meta(&mj, "dim"),
        hidden: meta(&mj, "hidden"),
        n_heads: meta(&mj, "n_heads"),
        n_kv: meta(&mj, "n_kv"),
        head_dim: meta(&mj, "head_dim"),
        seq_len: positions,
        // From the checkpoint's own configuration. Qwen3 specifies 1e-6 where Llama
        // specifies 1e-5, and the float64 reference reads the same field.
        rms_eps: tensor_kernel::fixed::RMS_EPS_QWEN3,
    };

    println!(
        "Qwen3-0.6B: {n_layers} layers, dim {}, hidden {}, heads {}, kv {}, head_dim {}, vocab {vocab}",
        cfg.dim, cfg.hidden, cfg.n_heads, cfg.n_kv, cfg.head_dim
    );
    print!("loading 1.2 GB of weights... ");
    let load = Instant::now();

    let mut st = Store {
        scales: Vec::new(),
        weights: Vec::new(),
        norms: Vec::new(),
    };
    // Shapes in the order they are pushed, so the borrowed views below can be built by
    // index without carrying names around.
    let shapes = [
        ("q", cfg.q_width(), cfg.dim),
        ("k", cfg.kv_width(), cfg.dim),
        ("v", cfg.kv_width(), cfg.dim),
        ("o", cfg.dim, cfg.q_width()),
        ("gate", cfg.hidden, cfg.dim),
        ("up", cfg.hidden, cfg.dim),
        ("down", cfg.dim, cfg.hidden),
    ];
    for l in 0..n_layers {
        for (name, rows, cols) in shapes {
            let (s, w) = split(read(&dir, &format!("l{l}_{name}.bin")), rows, cols, name);
            st.scales.push(s);
            st.weights.push(w);
        }
        for name in ["input_norm", "q_norm", "k_norm", "post_norm"] {
            st.norms.push(read_i64(&dir, &format!("l{l}_{name}.bin")));
        }
    }
    let (head_s, head_w) = split(read(&dir, "lm_head.bin"), vocab, cfg.dim, "lm_head");
    let final_norm = read_i64(&dir, "final_norm.bin");
    let embed = read_i64(&dir, "embed.bin");
    let rope = Rope {
        cos: read_i64(&dir, "cos.bin"),
        sin: read_i64(&dir, "sin.bin"),
        head_dim: cfg.head_dim,
    };
    let expected = read_f64(&dir, "logits.bin");
    println!("{:.1?}", load.elapsed());

    let layers: Vec<LayerWeights> = (0..n_layers)
        .map(|l| {
            let t = |k: usize, rows, cols| QTensor {
                w: &st.weights[l * 7 + k],
                scales: &st.scales[l * 7 + k],
                rows,
                cols,
            };
            LayerWeights {
                input_norm: &st.norms[l * 4],
                q: t(0, cfg.q_width(), cfg.dim),
                k: t(1, cfg.kv_width(), cfg.dim),
                v: t(2, cfg.kv_width(), cfg.dim),
                q_norm: &st.norms[l * 4 + 1],
                k_norm: &st.norms[l * 4 + 2],
                o: t(3, cfg.dim, cfg.q_width()),
                post_norm: &st.norms[l * 4 + 3],
                gate: t(4, cfg.hidden, cfg.dim),
                up: t(5, cfg.hidden, cfg.dim),
                down: t(6, cfg.dim, cfg.hidden),
                mixture: None,
            }
        })
        .collect();

    let model = Model {
        cfg,
        layers,
        final_norm: &final_norm,
        lm_head: QTensor {
            w: &head_w,
            scales: &head_s,
            rows: vocab,
            cols: cfg.dim,
        },
    };

    let mut kv = KvCache::new(&cfg, n_layers);
    let mut scratch = Scratch::new(&cfg);
    let mut logits = vec![0i64; vocab];

    println!("\nrunning {positions} positions through {n_layers} layers\n");
    let mut per_token = Vec::new();
    for pos in 0..positions {
        let mut x = embed[pos * cfg.dim..(pos + 1) * cfg.dim].to_vec();
        let t = Instant::now();
        model.forward(&rope, &mut kv, &mut scratch, pos, &mut x, &mut logits);
        let e = t.elapsed();
        per_token.push(e);
        println!("  position {pos}: {e:.2?}");
    }

    // ---- agreement with float64 ----
    let scale: f64 = expected.iter().map(|v| v.abs()).sum::<f64>() / vocab as f64;
    let mut sum = 0.0;
    let mut max = 0.0f64;
    for (g, t) in logits.iter().zip(&expected) {
        let d = ((*g as f64 / Q32) - t).abs();
        sum += d;
        max = max.max(d);
    }
    println!(
        "\nlogits vs float64: mean relative {:.3e}, max relative {:.3e}",
        sum / vocab as f64 / scale,
        max / scale
    );

    // ---- the check that actually matters ----
    // Numerical closeness is necessary and not sufficient. What a language model emits is
    // an argmax, so the question is whether the same token wins.
    let mut order: Vec<usize> = (0..vocab).collect();
    order.sort_by(|a, b| logits[*b].cmp(&logits[*a]));
    let mut want: Vec<usize> = (0..vocab).collect();
    want.sort_by(|a, b| expected[*b].partial_cmp(&expected[*a]).expect("finite"));

    println!(
        "\n{:<6} {:>10} {:>14} {:>10} {:>14}",
        "rank", "int id", "int logit", "f64 id", "f64 logit"
    );
    println!("{}", "-".repeat(60));
    for r in 0..10 {
        println!(
            "{:<6} {:>10} {:>14.4} {:>10} {:>14.4}",
            r + 1,
            order[r],
            logits[order[r]] as f64 / Q32,
            want[r],
            expected[want[r]]
        );
    }

    let top1 = order[0] == want[0];
    let top5 = order[..5].iter().collect::<std::collections::HashSet<_>>()
        == want[..5].iter().collect::<std::collections::HashSet<_>>();

    let avg: f64 = per_token.iter().map(|d| d.as_secs_f64()).sum::<f64>() / positions as f64;
    println!(
        "\n{:.0} ms per token, {:.1} tokens/sec",
        avg * 1e3,
        1.0 / avg
    );

    println!(
        "\ntop-1 agrees: {}\ntop-5 set agrees: {}",
        if top1 { "yes" } else { "NO" },
        if top5 { "yes" } else { "NO" }
    );
    if !top1 {
        eprintln!("\nthe integer model selects a different token from float64");
        std::process::exit(1);
    }
}
