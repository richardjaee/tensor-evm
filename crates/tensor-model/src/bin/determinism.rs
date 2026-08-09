//! Does this machine compute the same bits as that one?
//!
//! Every claim in this repository rests on an answer of yes. A chain verifies inference by
//! having each node re-execute it, so two honest nodes that disagree on a single bit fork
//! the chain, and the guarantee is gone. Floating point cannot give that answer across
//! heterogeneous hardware — reduction order follows thread count, kernel selection follows
//! vendor, and fused multiply-add changes results depending on what the compiler emitted.
//! Integer arithmetic has none of those failure modes. That is the argument. This is the
//! measurement.
//!
//! It prints one digest per stage and one for the whole run. Two machines agree if and only
//! if the digests match, and where they stop matching says which stage broke.
//!
//! Deliberately free of revm, solc and the network: determinism is a property of the
//! arithmetic, not of the EVM plumbing around it, so this needs nothing but a Rust
//! toolchain and the repository. That is what makes it runnable on a machine you were
//! handed five minutes ago.
//!
//! ```text
//! cargo run --release --bin determinism
//! cargo run --release --bin determinism -- --threads 1   # must not change a digest
//! ```

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tensor_kernel::fixed::{argmax, exp_q32, isqrt, silu, RMS_EPS_QWEN3};
use tensor_kernel::{matmul, matmul_scalar};
use tensor_model::qwen3::{Config, KvCache, LayerWeights, Model, QTensor, Rope, Scratch};

/// Accumulates the values a stage produced, in order, into one digest.
///
/// Values go in as little-endian `i64` rather than as formatted text, so the digest covers
/// the arithmetic instead of the way this program happens to print numbers.
#[derive(Default)]
struct Hasher {
    inner: Sha256,
    count: usize,
}

impl Hasher {
    fn feed(&mut self, v: i64) {
        self.inner.update(v.to_le_bytes());
        self.count += 1;
    }

    fn feed_all(&mut self, vs: &[i64]) {
        for &v in vs {
            self.feed(v);
        }
    }

    fn finish(self) -> (String, usize) {
        (hex(&self.inner.finalize()), self.count)
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Weights and activations from a fixed integer recurrence: identical everywhere, and
/// spanning the sign range so a discrepancy in rounding or shifting has somewhere to show.
fn synth(rows: usize, cols: usize) -> (Vec<i16>, Vec<i8>, Vec<i64>) {
    let w = (0..rows * cols)
        .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
        .collect();
    let scales = (0..rows).map(|r| (r % 5) as i8 + 28).collect();
    let x = (0..cols)
        .map(|i| ((i as i64 * 40503 + 17) % 131071) - 65535)
        .collect();
    (w, scales, x)
}

/// The kernel across shapes that straddle the parallel threshold, so both the scalar path
/// and the work-stealing one are covered in a single digest.
fn stage_matmul() -> (String, usize) {
    let mut h = Hasher::default();
    let mut out = Vec::new();
    for &(rows, cols) in &[
        (1, 1),
        (7, 13), // not a multiple of anything, so no lane divides it evenly
        (64, 96),
        (256, 256),
        (1024, 1024), // over PARALLEL_THRESHOLD, so rayon actually splits this
        (2048, 1024), // Qwen3-0.6B's q_proj
        (1024, 3072), // and its down_proj, wide rather than tall
    ] {
        let (w, scales, x) = synth(rows, cols);
        out.resize(rows, 0);

        matmul(&w, &scales, &x, rows, cols, &mut out);
        h.feed_all(&out);

        // The parallel result must equal the scalar one, not merely be reproducible. If
        // these ever diverge the digest is meaningless, so it is asserted rather than
        // hashed and hoped over.
        let mut serial = vec![0i64; rows];
        matmul_scalar(&w, &scales, &x, rows, cols, &mut serial);
        assert_eq!(
            out, serial,
            "parallel disagreed with scalar at {rows}x{cols}"
        );
    }
    h.finish()
}

/// The fixed-point primitives, swept rather than spot-checked. `isqrt` and `exp_q32` are
/// where a port to another architecture would most plausibly drift, since both are
/// iterative and both are the kind of thing a compiler is tempted to reassociate.
fn stage_fixed() -> (String, usize) {
    let mut h = Hasher::default();

    for i in 0..4096u64 {
        let a = (i as u128) * 1_000_003;
        h.feed(isqrt(a) as i64);
        h.feed(isqrt(a << 40) as i64);
    }
    // exp over the range softmax actually asks for: exponents of non-positive shifted
    // scores, where the function decays across its whole useful span.
    for i in 0..4096i64 {
        h.feed(exp_q32(-i * 8_000_000) as i64);
    }
    for i in -2048..2048i64 {
        h.feed(silu(i * 4_000_000));
    }
    // argmax settles ties by lowest index; a different rule would change which token comes
    // out of a generation whose top two logits are equal.
    for k in 0..64usize {
        let v: Vec<i64> = (0..97).map(|i| ((i * 37 + k) % 41) as i64).collect();
        h.feed(argmax(&v) as i64);
    }

    h.finish()
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name)).unwrap_or_else(|e| {
        panic!(
            "{}: {e} — regenerate with tools/synth_model.py",
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

struct Store {
    scales: Vec<Vec<i8>>,
    weights: Vec<Vec<i16>>,
    norms: Vec<Vec<i64>>,
}

/// The whole model, every position, hashing the full logit vector each time.
///
/// The stage that matters most, because it is the one with depth. A single matmul agreeing
/// says the multiplier agrees; a four-layer forward pass agreeing says nothing accumulated
/// along the way, which is the property a chain actually depends on. Hashing every logit
/// rather than the argmax means a disagreement is caught even when it was too small to
/// change the token — the chain forks on the bits, not on the answer.
fn stage_model(dir: &Path) -> (String, usize, Vec<usize>) {
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
        rms_eps: RMS_EPS_QWEN3,
    };

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
            let (s, w) = split(read(dir, &format!("l{l}_{name}.bin")), rows, cols, name);
            st.scales.push(s);
            st.weights.push(w);
        }
        for name in ["input_norm", "q_norm", "k_norm", "post_norm"] {
            st.norms.push(read_i64(dir, &format!("l{l}_{name}.bin")));
        }
    }
    let (head_s, head_w) = split(read(dir, "lm_head.bin"), vocab, cfg.dim, "lm_head");
    let final_norm = read_i64(dir, "final_norm.bin");
    let embed = read_i64(dir, "embed.bin");
    let rope = Rope {
        cos: read_i64(dir, "cos.bin"),
        sin: read_i64(dir, "sin.bin"),
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
    let mut h = Hasher::default();
    let mut tokens = Vec::new();

    for pos in 0..positions {
        let mut x = embed[pos * cfg.dim..(pos + 1) * cfg.dim].to_vec();
        model.forward(&rope, &mut kv, &mut scratch, pos, &mut x, &mut logits);
        h.feed_all(&logits);
        // The residual stream too, not only the logits. A layer could drift in a direction
        // the classifier happens to flatten, and hashing only the output would miss it.
        h.feed_all(&x);
        tokens.push(argmax(&logits));
    }

    let (digest, count) = h.finish();
    (digest, count, tokens)
}

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
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root")
            .join("testdata/tiny")
    });

    // Printed so a mismatch can be attributed rather than merely noticed. None of it should
    // affect a single digest below — that is the entire claim.
    println!("host");
    println!("  arch            {}", std::env::consts::ARCH);
    println!("  os              {}", std::env::consts::OS);
    println!("  pointer width   {}", usize::BITS);
    println!("  threads         {}", rayon::current_num_threads());
    println!("  fixture         {}", dir.display());

    let (m_digest, m_n) = stage_matmul();
    let (f_digest, f_n) = stage_fixed();
    let (o_digest, o_n, tokens) = stage_model(&dir);

    println!("\n{:<10} {:>10}  sha256", "stage", "values");
    println!("{}", "-".repeat(88));
    println!("{:<10} {:>10}  {}", "matmul", m_n, m_digest);
    println!("{:<10} {:>10}  {}", "fixed", f_n, f_digest);
    println!("{:<10} {:>10}  {}", "model", o_n, o_digest);

    // One line to compare, so agreement is checkable by eye across two terminals.
    let mut all = Sha256::new();
    for d in [&m_digest, &f_digest, &o_digest] {
        all.update(d.as_bytes());
    }
    println!("\ntokens generated  {tokens:?}");
    println!("OVERALL           {}", hex(&all.finalize()));
    println!(
        "\nEvery digest above must be identical on every machine, at any thread count. A\n\
         single differing character means two honest nodes would fork the chain."
    );
}
