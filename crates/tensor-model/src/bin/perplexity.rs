//! What does the integer path cost in prediction quality?
//!
//! Determinism says every machine computes the same bits. It says nothing about whether
//! those bits are any good — a model can agree with itself perfectly and still be broken.
//! The two claims need separate evidence, and this is the second one.
//!
//! Perplexity is the standard measure, and the one the comparable projects report, so it
//! is the number that makes this repository's arithmetic comparable to theirs rather than
//! only to itself. `tools/ppl_f64.py` computes the same quantity from the original bf16
//! checkpoint in float64; the pair is the result, not either alone.
//!
//! ```text
//! cargo run --release --bin perplexity -- models/ppl
//! ```
//!
//! The export must carry the whole evaluation sequence: `meta.json`'s `prompt_ids` are the
//! tokens scored, and position `p`'s logits are graded against `prompt_ids[p + 1]`. A
//! sequence of `P` tokens therefore yields `P - 1` predictions.

use std::path::{Path, PathBuf};
use tensor_kernel::fixed::RMS_EPS_QWEN3;
use tensor_model::qwen3::{Config, KvCache, LayerWeights, Model, QTensor, Rope, Scratch};

/// 2^32, the scale a Q32 logit is expressed in.
const Q32: f64 = 4_294_967_296.0;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir: Option<PathBuf> = None;
    let mut threads: Option<usize> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--threads" => threads = args.next().and_then(|v| v.parse().ok()),
            other => dir = Some(other.into()),
        }
    }
    if let Some(n) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .expect("thread pool");
    }
    let dir = dir.unwrap_or_else(|| {
        panic!("usage: perplexity <export-dir> [--threads N]\nThe export's prompt_ids are the evaluation sequence.")
    });

    let mj = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");
    let ids = meta_list(&mj, "prompt_ids");
    let positions = meta(&mj, "positions");
    assert_eq!(
        ids.len(),
        positions,
        "meta.json: {} prompt_ids for {positions} positions",
        ids.len()
    );
    assert!(
        positions >= 2,
        "perplexity needs at least two tokens; this export has {positions}"
    );

    let n_layers = meta(&mj, "n_layers");
    let vocab = meta(&mj, "vocab");
    let cfg = Config {
        dim: meta(&mj, "dim"),
        hidden: meta(&mj, "hidden"),
        n_heads: meta(&mj, "n_heads"),
        n_kv: meta(&mj, "n_kv"),
        head_dim: meta(&mj, "head_dim"),
        seq_len: positions,
        rms_eps: RMS_EPS_QWEN3,
    };

    println!("host");
    println!("  arch            {}", std::env::consts::ARCH);
    println!("  threads         {}", rayon::current_num_threads());
    println!("  export          {}", dir.display());
    println!("  tokens          {positions}");
    println!("  predictions     {}", positions - 1);

    let mut st = Store {
        scales: Vec::new(),
        weights: Vec::new(),
        norms: Vec::new(),
    };
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

    // Summed in f64 rather than accumulated in the fixed-point domain on purpose: the
    // quantity being reported is a real number compared against another implementation's
    // real number, and rounding it here would put this side's thumb on the scale.
    let mut nll = 0.0f64;
    let mut top1 = 0usize;

    for pos in 0..positions {
        let mut x = embed[pos * cfg.dim..(pos + 1) * cfg.dim].to_vec();
        model.forward(&rope, &mut kv, &mut scratch, pos, &mut x, &mut logits);

        // The last position predicts a token the sequence does not contain, so it is run
        // for its effect on the cache and then not scored.
        if pos + 1 == positions {
            break;
        }
        let target = ids[pos + 1];
        assert!(target < vocab, "token {target} outside vocab {vocab}");
        nll -= log_softmax_at(&logits, target);
        if argmax_f(&logits) == target {
            top1 += 1;
        }
    }

    let n = (positions - 1) as f64;
    let mean_nll = nll / n;
    println!();
    println!("integer path");
    println!("  mean NLL        {mean_nll:.6} nats");
    println!("  perplexity      {:.4}", mean_nll.exp());
    println!(
        "  top-1 accuracy  {:.2}% ({top1}/{})",
        100.0 * top1 as f64 / n,
        positions - 1
    );
    println!();
    println!("Compare against tools/ppl_f64.py on the same token ids. Determinism is a");
    println!("separate claim from quality; this is the quality one.");
}

/// `log softmax(l)[i]`, computed in the shifted form so the exponential cannot overflow.
fn log_softmax_at(logits: &[i64], i: usize) -> f64 {
    let max = logits.iter().copied().max().expect("non-empty") as f64 / Q32;
    let sum: f64 = logits.iter().map(|&l| (l as f64 / Q32 - max).exp()).sum();
    (logits[i] as f64 / Q32) - max - sum.ln()
}

fn argmax_f(logits: &[i64]) -> usize {
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join(name).display()))
}

fn read_i64(dir: &Path, name: &str) -> Vec<i64> {
    read(dir, name)
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect()
}

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

/// `meta()` stops at the first comma, which a list of ids is full of.
fn meta_list(s: &str, key: &str) -> Vec<usize> {
    let at = s
        .find(&format!("\"{key}\":"))
        .unwrap_or_else(|| panic!("meta.json has no {key}"));
    let open = s[at..].find('[').expect("list opens") + at;
    let close = s[open..].find(']').expect("list closes") + open;
    s[open + 1..close]
        .split(',')
        .map(|v| v.trim().parse().expect("token id"))
        .collect()
}

struct Store {
    scales: Vec<Vec<i8>>,
    weights: Vec<Vec<i16>>,
    norms: Vec<Vec<i64>>,
}
