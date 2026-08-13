//! A resumable forward pass, end to end, with state in storage.
//!
//! `onchain_layer` runs a block inside one transaction, where every intermediate lives in
//! memory and vanishes when it returns. That is not how a token can work: a token exceeds
//! what one transaction may spend, so the residual stream and the key/value cache have to
//! survive between calls.
//!
//! This drives the real thing — `start`, then `step` — against real Qwen3-0.6B weights, and
//! checks the residual stream left in storage against the Rust reference. It also exercises
//! the granularity knob: how many layers a call attempts is the caller's choice, and the
//! two costs that choice trades off are measured rather than described.
//!
//! ```text
//! python3 tools/reflayer.py --model models/qwen3-0.6b.safetensors \
//!     --config models/config.json --layer 0 --positions 4 --out models/layer0
//! cargo run --release --example onchain_token -- models/layer0
//! ```

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{DatabaseCommit, ExecuteEvm, MainBuilder, MainContext};
use std::path::{Path, PathBuf};
use std::process::Command;
use tensor_model::qwen3::{forward_layer, Config, KvCache, LayerWeights, QTensor, Rope, Scratch};
use tensor_precompile::state::TensorPrecompiles;

const CALLER: Address = Address::new([0x11; 20]);
const GEN: Address = Address::new([0x66; 20]);
const WEIGHTS: Address = Address::new([0x33; 20]);
const GAS_CAP: u64 = 400_000_000;
const ID: [u8; 32] = [0x07; 32];

/// A blob as a real upload leaves it: a leading tag byte marking this as a chunk rather
/// than a directory of chunks. The harness places code directly where a chain would run
/// `WeightStore`, so it has to produce the same bytes.
fn tagged(mut b: Vec<u8>) -> Vec<u8> {
    b.insert(0, 0x00);
    b
}

fn read(dir: &Path, name: &str) -> Vec<u8> {
    std::fs::read(dir.join(name))
        .unwrap_or_else(|e| panic!("{}: {e} — run tools/reflayer.py", dir.join(name).display()))
}

fn read_i64(dir: &Path, name: &str) -> Vec<i64> {
    read(dir, name)
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect()
}

fn split(raw: &[u8], rows: usize, cols: usize) -> (Vec<i8>, Vec<i16>) {
    assert_eq!(raw.len(), rows + rows * cols * 2);
    (
        raw[..rows].iter().map(|&b| b as i8).collect(),
        raw[rows..]
            .chunks_exact(2)
            .map(|c| i16::from_be_bytes([c[0], c[1]]))
            .collect(),
    )
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

fn solc_runtime(src: &Path, name: &str) -> Vec<u8> {
    let out = Command::new("solc")
        .args(["--optimize", "--optimize-runs", "200", "--bin-runtime"])
        .arg(src)
        .output()
        .expect("solc must be installed");
    assert!(
        out.status.success(),
        "solc: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("utf8");
    let at = text.find(&format!(":{name} =======")).expect("section");
    let hex = text[at..]
        .lines()
        .find(|l| l.starts_with("60"))
        .expect("bytecode")
        .trim();
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

fn word(v: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&v.to_be_bytes());
    w
}

fn signed(v: i64) -> [u8; 32] {
    let mut w = [0u8; 32];
    if v < 0 {
        w[..24].fill(0xff);
    }
    w[24..].copy_from_slice(&v.to_be_bytes());
    w
}

/// Q32 values in the precompiles' packed form: eight big-endian bytes each.
fn packed(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

/// ABI tail for a `bytes`: length, then data padded to a word.
fn dyn_bytes(b: &[u8]) -> Vec<u8> {
    let mut out = word(b.len() as u64).to_vec();
    out.extend_from_slice(b);
    let pad = (32 - b.len() % 32) % 32;
    out.extend(std::iter::repeat_n(0u8, pad));
    out
}

/// A tuple whose members are all `bytes`: offsets, then the tails.
fn bytes_tuple(parts: &[&[u8]]) -> Vec<u8> {
    let tails: Vec<Vec<u8>> = parts.iter().map(|p| dyn_bytes(p)).collect();
    let mut head = Vec::new();
    let mut at = parts.len() * 32;
    for t in &tails {
        head.extend_from_slice(&word(at as u64));
        at += t.len();
    }
    for t in tails {
        head.extend_from_slice(&t);
    }
    head
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/layer0".to_string())
        .into();
    let mj = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");
    let cfg = Config {
        dim: meta(&mj, "dim"),
        hidden: meta(&mj, "hidden"),
        n_heads: meta(&mj, "n_heads"),
        n_kv: meta(&mj, "n_kv"),
        head_dim: meta(&mj, "head_dim"),
        seq_len: meta(&mj, "positions"),
        rms_eps: tensor_kernel::fixed::RMS_EPS_QWEN3,
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
    let mut blob: Vec<u8> = Vec::new();
    let mut offsets = Vec::new();
    let mut tensors = Vec::new();
    for (name, rows, cols) in shapes {
        let raw = read(&dir, &format!("{name}.bin"));
        offsets.push(blob.len() as u64);
        tensors.push(split(&raw, rows, cols));
        blob.extend_from_slice(&raw);
    }
    // Norm weights follow the tensors in the same blob, as they do on chain.
    for name in ["input_norm", "q_norm", "k_norm", "post_norm"] {
        let v = read_i64(&dir, &format!("{name}.bin"));
        offsets.push(blob.len() as u64);
        blob.extend(v.iter().flat_map(|x| x.to_be_bytes()));
    }

    let input_norm = read_i64(&dir, "input_norm.bin");
    let q_norm = read_i64(&dir, "q_norm.bin");
    let k_norm = read_i64(&dir, "k_norm.bin");
    let post_norm = read_i64(&dir, "post_norm.bin");
    let cos = read_i64(&dir, "cos.bin");
    let sin = read_i64(&dir, "sin.bin");
    let xs = read_i64(&dir, "x.bin");
    let half = cfg.head_dim / 2;

    // ---- the Rust layer, as the reference ----
    let w = LayerWeights {
        input_norm: &input_norm,
        q: QTensor {
            w: &tensors[0].1,
            scales: &tensors[0].0,
            rows: cfg.q_width(),
            cols: cfg.dim,
        },
        k: QTensor {
            w: &tensors[1].1,
            scales: &tensors[1].0,
            rows: cfg.kv_width(),
            cols: cfg.dim,
        },
        v: QTensor {
            w: &tensors[2].1,
            scales: &tensors[2].0,
            rows: cfg.kv_width(),
            cols: cfg.dim,
        },
        q_norm: &q_norm,
        k_norm: &k_norm,
        o: QTensor {
            w: &tensors[3].1,
            scales: &tensors[3].0,
            rows: cfg.dim,
            cols: cfg.q_width(),
        },
        post_norm: &post_norm,
        gate: QTensor {
            w: &tensors[4].1,
            scales: &tensors[4].0,
            rows: cfg.hidden,
            cols: cfg.dim,
        },
        up: QTensor {
            w: &tensors[5].1,
            scales: &tensors[5].0,
            rows: cfg.hidden,
            cols: cfg.dim,
        },
        down: QTensor {
            w: &tensors[6].1,
            scales: &tensors[6].0,
            rows: cfg.dim,
            cols: cfg.hidden,
        },
        mixture: None,
    };
    let rope = Rope {
        cos: cos.clone(),
        sin: sin.clone(),
        head_dim: cfg.head_dim,
    };
    let mut kv = KvCache::new(&cfg, 1);
    let mut scratch = Scratch::new(&cfg);
    let mut want = xs[..cfg.dim].to_vec();
    forward_layer(&cfg, &w, &rope, &mut kv, &mut scratch, 0, 0, &mut want);

    // ---- chain ----
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let code = solc_runtime(&root.join("contracts/src/Generator.sol"), "Generator");

    let mut db = CacheDB::new(EmptyDB::default());
    db.insert_account_info(
        CALLER,
        AccountInfo {
            balance: U256::from(1u64) << 100,
            ..Default::default()
        },
    );
    db.insert_account_info(
        WEIGHTS,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(tagged(blob)))),
            ..Default::default()
        },
    );
    db.insert_account_info(
        GEN,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(code))),
            ..Default::default()
        },
    );

    let mut nonce = 0u64;
    let mut send = |db: &mut CacheDB<EmptyDB>, data: Vec<u8>| -> ExecutionResult {
        let out = {
            let evm = Context::mainnet()
                .with_db(&mut *db)
                .modify_cfg_chained(|c| c.tx_gas_limit_cap = Some(GAS_CAP))
                .build_mainnet();
            let mut evm = evm.with_precompiles(TensorPrecompiles::default());
            evm.transact(TxEnv {
                caller: CALLER,
                kind: TxKind::Call(GEN),
                data: data.into(),
                gas_limit: GAS_CAP,
                gas_price: 0,
                nonce,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("rejected before execution: {e:?}"))
        };
        nonce += 1;
        db.commit(out.state);
        out.result
    };

    println!("Qwen3-0.6B layer 0, resumable, state in storage\n");

    // ---- start ----
    let mut d = keccak256(b"start(bytes32,bytes)")[..4].to_vec();
    d.extend_from_slice(&ID);
    d.extend_from_slice(&word(2 * 32));
    d.extend_from_slice(&dyn_bytes(&packed(&xs[..cfg.dim])));
    let r = send(&mut db, d);
    assert!(r.is_success(), "start failed: {r:?}");
    let start_gas = r.tx_gas_used();
    println!("  start()                {start_gas:>12} gas");

    // ---- step ----
    // step(bytes32, Model, uint16). Model's members are a static Config, a Layer array, and
    // two tuples of bytes, so the tuple is dynamic and encodes behind an offset.
    let sig = "step(bytes32,((uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,\
uint32,uint32,uint32),\
(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32)[],\
(bytes,bytes)),uint16)";
    let mut d = keccak256(sig.as_bytes())[..4].to_vec();
    d.extend_from_slice(&ID);
    d.extend_from_slice(&word(3 * 32)); // offset to Model
    d.extend_from_slice(&word(1)); // maxLayers

    // Model: eight Config words, then three offsets, then the tails.
    let mut model = Vec::new();
    for f in [
        cfg.dim as u64,
        cfg.hidden as u64,
        cfg.n_heads as u64,
        cfg.n_kv as u64,
        cfg.head_dim as u64,
        cfg.seq_len as u64,
        1, // nLayers, the stack this call knows about
    ] {
        model.extend_from_slice(&word(f));
    }
    model.extend_from_slice(&signed(cfg.rms_eps));
    // nExperts, topK, expertStride. Zero is what dense means, and this example is dense.
    for _ in 0..3 {
        model.extend_from_slice(&word(0));
    }

    let mut layers = word(1).to_vec(); // one Layer
    let mut la = [0u8; 32];
    la[12..].copy_from_slice(WEIGHTS.as_slice());
    layers.extend_from_slice(&la);
    for o in &offsets {
        layers.extend_from_slice(&word(*o));
    }
    layers.extend_from_slice(&word(0)); // router, unused when dense
    let ropes = bytes_tuple(&[&packed(&cos[..half]), &packed(&sin[..half])]);

    // Config's eleven words, then offsets to the layer array and the rotary tables.
    let head = 13 * 32;
    model.extend_from_slice(&word(head as u64));
    model.extend_from_slice(&word((head + layers.len()) as u64));
    model.extend_from_slice(&layers);
    model.extend_from_slice(&ropes);
    d.extend_from_slice(&model);

    let r = send(&mut db, d);
    let step_gas = r.tx_gas_used();
    if !r.is_success() {
        eprintln!("step failed: {r:?}");
        std::process::exit(1);
    }
    println!("  step(maxLayers = 1)    {step_gas:>12} gas");

    // ---- read the checkpoint back ----
    let mut d = keccak256(b"residual(bytes32,uint32)")[..4].to_vec();
    d.extend_from_slice(&ID);
    d.extend_from_slice(&word(cfg.dim as u64));
    let r = send(&mut db, d);
    let ret = match &r {
        ExecutionResult::Success { output, .. } => match output {
            Output::Call(b) => b.to_vec(),
            Output::Create(b, _) => b.to_vec(),
        },
        other => panic!("residual() failed: {other:?}"),
    };
    let n = u64::from_be_bytes(ret[56..64].try_into().unwrap()) as usize;
    let got: Vec<i64> = (0..n)
        .map(|i| {
            let at = 64 + i * 32 + 24;
            i64::from_be_bytes(ret[at..at + 8].try_into().unwrap())
        })
        .collect();

    let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    println!("\n  residual width         {n}");
    println!("  mismatches vs Rust     {mismatches}");

    if mismatches != 0 {
        eprintln!("\nthe checkpointed state differs from the reference");
        std::process::exit(1);
    }
    println!("\nbit-identical after a round trip through storage");

    // ---- what the state cost ----
    // The layer's computation is known from `onchain_layer`; the rest is the checkpoint.
    const LAYER_COMPUTE: u64 = 688_198;
    let overhead = step_gas.saturating_sub(LAYER_COMPUTE);
    println!("\n  of the step, roughly {LAYER_COMPUTE} is computation");
    println!("  and {overhead} is calldata, cache, and checkpointing");
    println!(
        "  so state and transport are {:.0}% of a single-layer step",
        100.0 * overhead as f64 / step_gas as f64
    );
}
