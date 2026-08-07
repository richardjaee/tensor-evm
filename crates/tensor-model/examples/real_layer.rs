//! Runs one real Qwen3-0.6B layer in integers and reports the error against float64.
//!
//! An example rather than a test, deliberately. It needs a 31 MB export that cannot be
//! committed, and a test that quietly skips when its data is absent is worse than no test —
//! this project has already been caught twice by measurements that silently measured
//! nothing. An example either runs and prints numbers or fails to start.
//!
//! Generate the input with:
//!
//! ```text
//! python3 tools/reflayer.py \
//!     --model models/qwen3-0.6b.safetensors \
//!     --config models/config.json \
//!     --layer 0 --positions 4 --out models/layer0
//! ```
//!
//! then `cargo run --release --example real_layer -- models/layer0`.

use std::path::{Path, PathBuf};
use tensor_model::qwen3::{forward_layer, Config, KvCache, LayerWeights, QTensor, Rope, Scratch};

const Q32: f64 = 4_294_967_296.0;

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — has tools/reflayer.py been run?",
            dir.join(name).display()
        )
    })
}

/// Little-endian `i64`, matching what numpy wrote.
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

/// A quantised tensor: `rows` scale bytes, then row-major big-endian `i16`.
fn read_tensor(dir: &Path, name: &str, rows: usize, cols: usize) -> (Vec<i8>, Vec<i16>) {
    let raw = read(dir, &format!("{name}.bin"));
    assert_eq!(
        raw.len(),
        rows + rows * cols * 2,
        "{name}: blob does not match {rows}x{cols}"
    );
    let scales = raw[..rows].iter().map(|&b| b as i8).collect();
    let w = raw[rows..]
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    (scales, w)
}

/// Minimal reading of the metadata this example needs.
fn meta(dir: &Path, key: &str) -> usize {
    let s = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");
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

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/layer0".to_string())
        .into();
    assert!(
        dir.is_dir(),
        "{} is not a directory — run tools/reflayer.py first",
        dir.display()
    );

    let cfg = Config {
        dim: meta(&dir, "dim"),
        hidden: meta(&dir, "hidden"),
        n_heads: meta(&dir, "n_heads"),
        n_kv: meta(&dir, "n_kv"),
        head_dim: meta(&dir, "head_dim"),
        seq_len: meta(&dir, "positions"),
    };
    let positions = cfg.seq_len;

    println!(
        "Qwen3-0.6B layer {}: dim {}, hidden {}, heads {}, kv {}, head_dim {}",
        meta(&dir, "layer"),
        cfg.dim,
        cfg.hidden,
        cfg.n_heads,
        cfg.n_kv,
        cfg.head_dim
    );
    println!(
        "  q_proj {}x{}, o_proj {}x{} — neither square, which is the point\n",
        cfg.q_width(),
        cfg.dim,
        cfg.dim,
        cfg.q_width()
    );

    let (qs, qw) = read_tensor(&dir, "q", cfg.q_width(), cfg.dim);
    let (ks, kw) = read_tensor(&dir, "k", cfg.kv_width(), cfg.dim);
    let (vs, vw) = read_tensor(&dir, "v", cfg.kv_width(), cfg.dim);
    let (os_, ow) = read_tensor(&dir, "o", cfg.dim, cfg.q_width());
    let (gs, gw) = read_tensor(&dir, "gate", cfg.hidden, cfg.dim);
    let (us, uw) = read_tensor(&dir, "up", cfg.hidden, cfg.dim);
    let (ds, dw) = read_tensor(&dir, "down", cfg.dim, cfg.hidden);

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
        q_norm: &q_norm,
        k_norm: &k_norm,
        o: QTensor {
            w: &ow,
            scales: &os_,
            rows: cfg.dim,
            cols: cfg.q_width(),
        },
        post_norm: &post_norm,
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

    let rope = Rope {
        cos: read_i64(&dir, "cos.bin"),
        sin: read_i64(&dir, "sin.bin"),
        head_dim: cfg.head_dim,
    };

    let xs = read_i64(&dir, "x.bin");
    let expected = read_f64(&dir, "expected.bin");
    assert_eq!(xs.len(), positions * cfg.dim, "input size");
    assert_eq!(expected.len(), positions * cfg.dim, "reference size");

    let mut kv = KvCache::new(&cfg, 1);
    let mut scratch = Scratch::new(&cfg);
    let mut worst = 0.0f64;

    println!(
        "{:<6} {:>16} {:>16}",
        "pos", "mean rel error", "max rel error"
    );
    println!("{}", "-".repeat(42));

    for pos in 0..positions {
        let mut x: Vec<i64> = xs[pos * cfg.dim..(pos + 1) * cfg.dim].to_vec();
        let start = std::time::Instant::now();
        forward_layer(&cfg, &w, &rope, &mut kv, &mut scratch, 0, pos, &mut x);
        let elapsed = start.elapsed();

        let want = &expected[pos * cfg.dim..(pos + 1) * cfg.dim];
        let scale: f64 = want.iter().map(|v| v.abs()).sum::<f64>() / cfg.dim as f64;
        let mut sum = 0.0;
        let mut max = 0.0f64;
        for (g, t) in x.iter().zip(want) {
            let d = ((*g as f64 / Q32) - t).abs();
            sum += d;
            max = max.max(d);
        }
        let mean_rel = sum / cfg.dim as f64 / scale;
        let max_rel = max / scale;
        worst = worst.max(mean_rel);
        println!("{pos:<6} {mean_rel:>16.3e} {max_rel:>16.3e}   ({elapsed:.2?})");
    }

    println!("\nworst mean relative error: {worst:.3e}");
    // int16 weights and Q32 activations, against float64 on the original bf16. Anything
    // near this is the encoding; anything far above it is a bug.
    if worst < 1e-3 {
        println!("the integer layer reproduces the model");
    } else {
        println!("ERROR: {worst:.3e} is too large to be quantisation alone");
        std::process::exit(1);
    }
}
