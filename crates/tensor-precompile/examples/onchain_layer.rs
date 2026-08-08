//! A real Qwen3 layer, executed by contract code, inside the EVM.
//!
//! This is the claim the project exists to make. Everything before it showed that *a
//! matmul* runs on chain while the forward pass itself lived in Rust, beside the chain
//! rather than on it. Here the orchestration is Solidity: the contract decides what to
//! multiply and in what order, reads weights from another contract's code, and calls out
//! only for the arithmetic a precompile accelerates.
//!
//! Checked against `tensor-model`'s Rust layer on the same weights, and **bit for bit**
//! rather than within a tolerance. Two implementations of one model that disagree in the
//! last place are two models, and the divergence would first appear as an occasional
//! different token rather than as an error.
//!
//! ```text
//! python3 tools/reflayer.py --model models/qwen3-0.6b.safetensors \
//!     --config models/config.json --layer 0 --positions 4 --out models/layer0
//! cargo run --release --example onchain_layer -- models/layer0
//! ```

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{ExecuteEvm, MainBuilder, MainContext};
use std::path::{Path, PathBuf};
use std::process::Command;
use tensor_model::qwen3::{forward_layer, Config, KvCache, LayerWeights, QTensor, Rope, Scratch};
use tensor_precompile::state::TensorPrecompiles;

const CALLER: Address = Address::new([0x11; 20]);
const RUNNER: Address = Address::new([0x55; 20]);
const WEIGHTS: Address = Address::new([0x33; 20]);
/// What this chain allows one transaction to spend.
///
/// Ethereum's default is EIP-7825's 2^24, and a layer does not fit in it. Raising the cap
/// is one of the changes this design already asks of a chain — alongside a code size limit
/// above EIP-170's 24KB, since the weight blob is 31 MB. Both are configuration rather than
/// consensus changes, but both are real, and a layer that only runs with them lifted should
/// say so rather than quietly assume them.
const GAS_CAP: u64 = 400_000_000;

/// Ethereum's own cap, kept for the arithmetic at the end.
const ETHEREUM_TX_CAP: u64 = 16_777_216;

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

fn split(raw: &[u8], rows: usize, cols: usize, what: &str) -> (Vec<i8>, Vec<i16>) {
    assert_eq!(raw.len(), rows + rows * cols * 2, "{what}: {rows}x{cols}");
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
        "solc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("utf8");
    let at = text
        .find(&format!(":{name} ======="))
        .unwrap_or_else(|| panic!("{name} section missing"));
    let hex = text[at..]
        .lines()
        .find(|l| l.starts_with("60"))
        .expect("runtime bytecode")
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

fn signed_word(v: i64) -> [u8; 32] {
    let mut w = [0u8; 32];
    if v < 0 {
        w[..24].fill(0xff);
    }
    w[24..].copy_from_slice(&v.to_be_bytes());
    w
}

/// ABI-encodes an `int256[]`: length, then sign-extended values.
fn enc_array(v: &[i64]) -> Vec<u8> {
    let mut out = word(v.len() as u64).to_vec();
    for &x in v {
        out.extend_from_slice(&signed_word(x));
    }
    out
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

    // ---- one blob holding every tensor, with offsets ----
    // A real chain would deploy this as contract code. EIP-170 caps that at 24KB, which is
    // one of the changes this design already requires of a chain; here the account's code
    // is set directly, so the cap does not apply.
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
        tensors.push(split(&raw, rows, cols, name));
        blob.extend_from_slice(&raw);
    }
    println!(
        "Qwen3-0.6B layer 0: dim {}, heads {}, kv {}, head_dim {}",
        cfg.dim, cfg.n_heads, cfg.n_kv, cfg.head_dim
    );
    println!(
        "  weight blob {:.1} MB across 7 tensors\n",
        blob.len() as f64 / 1e6
    );

    let input_norm = read_i64(&dir, "input_norm.bin");
    let q_norm = read_i64(&dir, "q_norm.bin");
    let k_norm = read_i64(&dir, "k_norm.bin");
    let post_norm = read_i64(&dir, "post_norm.bin");
    let cos = read_i64(&dir, "cos.bin");
    let sin = read_i64(&dir, "sin.bin");
    let xs = read_i64(&dir, "x.bin");

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

    // ---- the same layer, in the EVM ----
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let code = solc_runtime(&root.join("contracts/src/LayerRunner.sol"), "LayerRunner");
    println!("  LayerRunner bytecode {} bytes", code.len());

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
            code: Some(Bytecode::new_raw(Bytes::from(blob))),
            ..Default::default()
        },
    );
    db.insert_account_info(
        RUNNER,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(code))),
            ..Default::default()
        },
    );

    // runLayer((uint32 x6,int256),(uint32 x7),address,uint256,(int256[] x7))
    let sig = "runLayer((uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256),\
(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32),uint256,\
(int256[],int256[],int256[],int256[],int256[],int256[],int256[]))";
    let h = keccak256(sig.as_bytes());
    let mut data = h[..4].to_vec();

    // Config and Layer are entirely static, so both encode inline.
    for f in [
        cfg.dim as u64,
        cfg.hidden as u64,
        cfg.n_heads as u64,
        cfg.n_kv as u64,
        cfg.head_dim as u64,
        cfg.seq_len as u64,
        1, // one layer here; the generator drives the full stack
    ] {
        data.extend_from_slice(&word(f));
    }
    data.extend_from_slice(&signed_word(cfg.rms_eps));
    // Layer: the weight contract, then the seven tensor offsets within it.
    let mut addr = [0u8; 32];
    addr[12..].copy_from_slice(WEIGHTS.as_slice());
    data.extend_from_slice(&addr);
    for o in &offsets {
        data.extend_from_slice(&word(*o));
    }
    data.extend_from_slice(&word(0)); // pos

    // Inputs is dynamic: a pointer, then its own head of seven pointers.
    let head_words = 8 + 8 + 1 + 1;
    data.extend_from_slice(&word(head_words as u64 * 32));

    let half = cfg.head_dim / 2;
    let arrays: Vec<Vec<i64>> = vec![
        input_norm.clone(),
        q_norm.clone(),
        k_norm.clone(),
        post_norm.clone(),
        cos[..half].to_vec(),
        sin[..half].to_vec(),
        xs[..cfg.dim].to_vec(),
    ];
    let encoded: Vec<Vec<u8>> = arrays.iter().map(|a| enc_array(a)).collect();
    let mut at = 7 * 32;
    for e in &encoded {
        data.extend_from_slice(&word(at as u64));
        at += e.len();
    }
    for e in &encoded {
        data.extend_from_slice(e);
    }

    println!("  calldata {:.1} KB\n", data.len() as f64 / 1024.0);

    let db2 = db.clone();
    let evm = Context::mainnet()
        .with_db(db)
        .modify_cfg_chained(|c| c.tx_gas_limit_cap = Some(GAS_CAP))
        .build_mainnet();
    let mut evm = evm.with_precompiles(TensorPrecompiles::default());
    let started = std::time::Instant::now();
    let result = evm
        .transact(TxEnv {
            caller: CALLER,
            kind: TxKind::Call(RUNNER),
            data: data.clone().into(),
            gas_limit: GAS_CAP,
            gas_price: 0,
            ..Default::default()
        })
        .expect("transaction must execute")
        .result;
    let elapsed = started.elapsed();

    let ret = match &result {
        ExecutionResult::Success { output, .. } => match output {
            Output::Call(b) => b.to_vec(),
            Output::Create(b, _) => b.to_vec(),
        },
        other => {
            eprintln!("the layer did not execute: {other:?}");
            std::process::exit(1);
        }
    };

    let gas = result.tx_gas_used();
    println!("gas used            : {gas}");
    println!("wall clock in revm  : {elapsed:.2?}");

    // ---- compare ----
    let len = u64::from_be_bytes(ret[56..64].try_into().unwrap()) as usize;
    assert_eq!(len, cfg.dim, "returned width");
    let got: Vec<i64> = (0..len)
        .map(|i| {
            let a = 64 + i * 32 + 24;
            i64::from_be_bytes(ret[a..a + 8].try_into().unwrap())
        })
        .collect();

    let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    println!("\nagainst the Rust layer, element by element");
    println!("  width       : {}", cfg.dim);
    println!("  mismatches  : {mismatches}");

    if mismatches == 0 {
        println!("\nbit-identical: the contract and the reference are one model");
    } else {
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .max()
            .unwrap_or(0);
        println!("  worst delta : {worst} (Q32 units)");
        eprintln!("\nthe contract and the reference disagree");
        std::process::exit(1);
    }

    // Extrapolation, clearly labelled as such: 28 of these plus a classifier.
    // ---- where the gas went ----
    // Measured inside the contract, so the transaction's own calldata is excluded. A
    // 1024-wide activation vector ABI-encodes to 32KB, which is about 500k gas of transport
    // that would otherwise be attributed to whichever operation happened to be under test.
    {
        let mut d = keccak256(
            b"profileLayer((uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256),\
(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32),uint256,\
(int256[],int256[],int256[],int256[],int256[],int256[],int256[]))",
        )[..4]
            .to_vec();
        d.extend_from_slice(&data[4..]);

        let evm = Context::mainnet()
            .with_db(db2)
            .modify_cfg_chained(|c| c.tx_gas_limit_cap = Some(GAS_CAP))
            .build_mainnet();
        let mut evm = evm.with_precompiles(TensorPrecompiles::default());
        let r = evm
            .transact(TxEnv {
                caller: CALLER,
                kind: TxKind::Call(RUNNER),
                data: d.into(),
                gas_limit: GAS_CAP,
                gas_price: 0,
                ..Default::default()
            })
            .expect("profile must execute")
            .result;
        let out = match &r {
            ExecutionResult::Success { output, .. } => match output {
                Output::Call(b) => b.to_vec(),
                Output::Create(b, _) => b.to_vec(),
            },
            other => panic!("profile failed: {other:?}"),
        };
        let n = u64::from_be_bytes(out[56..64].try_into().unwrap()) as usize;
        let vals: Vec<u64> = (0..n)
            .map(|i| {
                let a = 64 + i * 32 + 24;
                u64::from_be_bytes(out[a..a + 8].try_into().unwrap())
            })
            .collect();

        let msize = *vals.last().expect("msize slot");
        let vals = &vals[..vals.len() - 1];
        let names = [
            "rmsnorm (attn)",
            "k/v proj + QK-norm + rope + cache",
            "q proj",
            "QK-norm + rope (q)",
            "attention",
            "o proj",
            "rmsnorm (ffn)",
            "gate + up proj",
            "swiglu",
            "down proj",
        ];
        let total: u64 = vals.iter().sum();
        println!("\nwhere a layer's gas goes\n");
        println!("{:<38} {:>12} {:>7}", "section", "gas", "share");
        println!("{}", "-".repeat(60));
        let mut precompiled = 0u64;
        for (name, v) in names.iter().zip(vals) {
            println!(
                "{name:<38} {v:>12} {:>6.1}%",
                100.0 * *v as f64 / total as f64
            );
            if name.contains("proj") || *name == "swiglu" {
                precompiled += *v;
            }
        }
        println!("{}", "-".repeat(60));
        println!("{:<38} {total:>12}", "measured inside the contract");
        println!(
            "\n  through precompiles {:>12} ({:.0}%)",
            precompiled,
            100.0 * precompiled as f64 / total as f64
        );
        println!(
            "  still in the EVM    {:>12} ({:.0}%)",
            total - precompiled,
            100.0 * (total - precompiled) as f64 / total as f64
        );

        // What the precompiles actually charge, as opposed to what the sections cost. The
        // difference is everything Solidity does to hand them their arguments.
        let mm = |r: u64, c: u64| tensor_precompile::gas_cost(r, c).unwrap();
        let d = cfg.dim as u64;
        let qw = cfg.q_width() as u64;
        let kvw = cfg.kv_width() as u64;
        let hid = cfg.hidden as u64;
        let charged: u64 = mm(qw, d)
            + mm(kvw, d) * 2
            + mm(d, qw)
            + mm(hid, d) * 2
            + mm(d, hid)
            + tensor_precompile::activation::gas_cost(hid).unwrap()
            + tensor_precompile::norm::gas_cost(cfg.n_heads as u64, cfg.head_dim as u64).unwrap()
            + tensor_precompile::norm::gas_cost(cfg.n_kv as u64, cfg.head_dim as u64).unwrap()
            + tensor_precompile::attention::attention_gas(
                cfg.n_heads as u64,
                cfg.head_dim as u64,
                1,
            )
            .unwrap()
            + tensor_precompile::attention::rmsnorm_gas(d).unwrap() * 2;

        // Memory expansion is 3 gas a word plus words^2/512, so its share is worth naming.
        let words = msize.div_ceil(32);
        let mem = 3 * words + words * words / 512;
        println!(
            "\n  memory touched     {:>12} bytes ({} words), which costs {mem} gas ({:.0}%)",
            msize,
            words,
            100.0 * mem as f64 / total as f64
        );
        println!("  the precompiles themselves charge {charged} gas of that {total}.");
        println!(
            "  so {:.1}% of a layer is Solidity handing them their arguments, not arithmetic.",
            100.0 * (total - charged) as f64 / total as f64
        );
    }

    // Extrapolation, labelled as such: twenty-eight of these plus a classifier.
    println!(
        "\na 28-layer token is about {:.0}M gas at this cost.",
        gas as f64 * 28.0 / 1e6
    );
    println!(
        "  under Ethereum's 2^24 per-transaction cap that is {} transactions per token",
        (gas * 28).div_ceil(ETHEREUM_TX_CAP)
    );
}
