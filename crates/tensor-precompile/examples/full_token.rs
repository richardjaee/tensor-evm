//! A whole token, produced on chain.
//!
//! Twenty-eight layers, the classifier, and an argmax — driven by contract code, against
//! real Qwen3-0.6B weights held in chain state, checked against the token float64 says the
//! model should emit.
//!
//! Everything before this proved a piece. `onchain_layer` showed a block is correct in the
//! EVM; `onchain_token` showed state survives between transactions. Neither showed the
//! thing the project is for, which is a model producing the right word.
//!
//! ```text
//! python3 tools/export_model.py --model models/qwen3-0.6b.safetensors \
//!     --config models/config.json --out models/full
//! cargo run --release --example full_token -- models/full
//! ```

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{DatabaseCommit, ExecuteEvm, MainBuilder, MainContext};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use tensor_precompile::state::TensorPrecompiles;

const CALLER: Address = Address::new([0x11; 20]);
const GEN: Address = Address::new([0x66; 20]);
const HEAD: Address = Address::new([0x44; 20]);
const GAS_CAP: u64 = 2_000_000_000;
const ID: [u8; 32] = [0x09; 32];

/// A blob as a real upload leaves it: a leading tag byte marking this as a chunk rather
/// than a directory of chunks. The harness places code directly where a chain would run
/// `WeightStore`, so it has to produce the same bytes.
fn tagged(mut b: Vec<u8>) -> Vec<u8> {
    b.insert(0, 0x00);
    b
}

fn layer_addr(l: usize) -> Address {
    let mut a = [0u8; 20];
    a[0] = 0xa0;
    a[18..].copy_from_slice(&(l as u16).to_be_bytes());
    Address::new(a)
}

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

fn addr_word(a: Address) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(a.as_slice());
    w
}

fn packed(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

fn dyn_bytes(b: &[u8]) -> Vec<u8> {
    let mut out = word(b.len() as u64).to_vec();
    out.extend_from_slice(b);
    out.extend(std::iter::repeat_n(0u8, (32 - b.len() % 32) % 32));
    out
}

/// A tuple whose members are all `bytes`: offsets relative to the tuple, then the tails.
fn bytes_tuple(parts: &[&[u8]]) -> Vec<u8> {
    let tails: Vec<Vec<u8>> = parts.iter().map(|p| dyn_bytes(p)).collect();
    let mut out = Vec::new();
    let mut at = parts.len() * 32;
    for t in &tails {
        out.extend_from_slice(&word(at as u64));
        at += t.len();
    }
    for t in tails {
        out.extend_from_slice(&t);
    }
    out
}

fn main() {
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/full".to_string())
        .into();
    // The granularity knob, from the command line: how many layers a `step` attempts.
    let chunk: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(usize::MAX);
    let mj = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");

    let dim = meta(&mj, "dim");
    let hidden = meta(&mj, "hidden");
    let n_layers = meta(&mj, "n_layers");
    let n_heads = meta(&mj, "n_heads");
    let n_kv = meta(&mj, "n_kv");
    let head_dim = meta(&mj, "head_dim");
    let vocab = meta(&mj, "vocab");
    let positions = meta(&mj, "positions");
    let q_width = n_heads * head_dim;
    let kv_width = n_kv * head_dim;
    let half = head_dim / 2;

    let params = (n_layers * (2 * dim * q_width + 2 * dim * kv_width + 3 * dim * hidden)
        + vocab * dim) as f64;
    println!(
        "on chain: {n_layers} layers, dim {dim}, vocab {vocab}, {positions} positions, {:.1}B parameters\n",
        params / 1e9
    );

    let mut db = CacheDB::new(EmptyDB::default());
    db.insert_account_info(
        CALLER,
        AccountInfo {
            balance: U256::from(1u64) << 100,
            ..Default::default()
        },
    );
    db.insert_account_info(
        GEN,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(solc_runtime(
                &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .and_then(|p| p.parent())
                    .expect("root")
                    .join("contracts/src/Generator.sol"),
                "Generator",
            )))),
            ..Default::default()
        },
    );

    // ---- weights into chain state, one contract per layer ----
    print!("deploying weights");
    let load = Instant::now();
    let shapes: [(&str, usize, usize); 7] = [
        ("q", q_width, dim),
        ("k", kv_width, dim),
        ("v", kv_width, dim),
        ("o", dim, q_width),
        ("gate", hidden, dim),
        ("up", hidden, dim),
        ("down", dim, hidden),
    ];
    // Seven tensors, then four norm vectors, all in one blob per layer.
    let mut offsets = vec![[0u64; 11]; n_layers];
    for (l, off) in offsets.iter_mut().enumerate() {
        let mut blob = Vec::new();
        for (i, (name, rows, cols)) in shapes.iter().enumerate() {
            let raw = read(&dir, &format!("l{l}_{name}.bin"));
            assert_eq!(raw.len(), rows + rows * cols * 2, "l{l}_{name}");
            off[i] = blob.len() as u64;
            blob.extend_from_slice(&raw);
        }
        // The four norm vectors follow the seven tensors in the same blob, so a layer is
        // one deployment and one address rather than weights on chain and norms in
        // calldata.
        for (j, name) in ["input_norm", "q_norm", "k_norm", "post_norm"]
            .iter()
            .enumerate()
        {
            let v = packed(&read_i64(&dir, &format!("l{l}_{name}.bin")));
            off[7 + j] = blob.len() as u64;
            blob.extend_from_slice(&v);
        }
        db.insert_account_info(
            layer_addr(l),
            AccountInfo {
                code: Some(Bytecode::new_raw(Bytes::from(tagged(blob)))),
                ..Default::default()
            },
        );
    }
    db.insert_account_info(
        HEAD,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(tagged(read(
                &dir,
                "lm_head.bin",
            ))))),
            ..Default::default()
        },
    );
    println!(" — {:.1?}", load.elapsed());

    let final_norm = read_i64(&dir, "final_norm.bin");
    let embed = read_i64(&dir, "embed.bin");
    let cos = read_i64(&dir, "cos.bin");
    let sin = read_i64(&dir, "sin.bin");
    // The argmax float64 produces at every position, not only the last. A chain that
    // reached the right final token by a wrong path would be indistinguishable otherwise.
    let expect: Vec<u32> = read(&dir, "tokens.bin")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
        .collect();
    assert_eq!(expect.len(), positions, "one expected token per position");

    // ---- driving it ----
    let mut nonce = 0u64;
    let mut total_gas = 0u64;
    // Gas is returned rather than accumulated inside, so the caller can attribute it per
    // position without the closure holding a borrow for the whole loop.
    let mut send = |db: &mut CacheDB<EmptyDB>, data: Vec<u8>, what: &str| -> (Vec<u8>, u64) {
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
            .unwrap_or_else(|e| panic!("{what} rejected: {e:?}"))
        };
        nonce += 1;
        let result = out.result.clone();
        db.commit(out.state);
        let used = result.tx_gas_used();
        let ret = match &result {
            ExecutionResult::Success { output, .. } => match output {
                Output::Call(b) => b.to_vec(),
                Output::Create(b, _) => b.to_vec(),
            },
            other => panic!("{what} failed: {other:?}"),
        };
        (ret, used)
    };

    // The model, encoded once and reused: config, layers, per-layer norms, rotary tables.
    let build_model = |pos: usize| -> Vec<u8> {
        let mut m = Vec::new();
        for f in [dim, hidden, n_heads, n_kv, head_dim, positions, n_layers] {
            m.extend_from_slice(&word(f as u64));
        }
        m.extend_from_slice(&signed(tensor_kernel::fixed::RMS_EPS_QWEN3));
        // nExperts, topK, expertStride. Zero is what dense means, and this model is dense.
        for _ in 0..3 {
            m.extend_from_slice(&word(0));
        }

        let mut layers = word(n_layers as u64).to_vec();
        for (l, off) in offsets.iter().enumerate() {
            layers.extend_from_slice(&addr_word(layer_addr(l)));
            for o in off {
                layers.extend_from_slice(&word(*o));
            }
            layers.extend_from_slice(&word(0)); // router, unused when dense
        }
        let a = pos * half;
        let rope = bytes_tuple(&[&packed(&cos[a..a + half]), &packed(&sin[a..a + half])]);

        // Config's eleven words, then offsets to the layer array and the rotary tables.
        let head = 13 * 32;
        m.extend_from_slice(&word(head as u64));
        m.extend_from_slice(&word((head + layers.len()) as u64));
        m.extend_from_slice(&layers);
        m.extend_from_slice(&rope);
        m
    };

    let step_sig = "step(bytes32,((uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,\
uint32,uint32,uint32),\
(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32)[],\
(bytes,bytes)),uint16)";

    println!(
        "{:<10} {:>14} {:>10} {:>10}",
        "position", "gas", "token", "expected"
    );
    println!("{}", "-".repeat(48));

    let mut got_token = 0u32;
    for pos in 0..positions {
        // Seed this position's embedding.
        let row = &embed[pos * dim..(pos + 1) * dim];
        let sel: &[u8] = if pos == 0 {
            &keccak256(b"start(bytes32,bytes)")[..4]
        } else {
            &keccak256(b"feed(bytes32,bytes)")[..4]
        };
        let mut d = sel.to_vec();
        d.extend_from_slice(&ID);
        d.extend_from_slice(&word(2 * 32));
        d.extend_from_slice(&dyn_bytes(&packed(row)));
        let (_, mut used) = send(&mut db, d, "seed");

        // As many layers per call as asked for; the loop repeats until the stack is done.
        let per_call = chunk.min(n_layers);
        let model = build_model(pos);
        let mut done = 0usize;
        while done < n_layers {
            let mut d = keccak256(step_sig.as_bytes())[..4].to_vec();
            d.extend_from_slice(&ID);
            d.extend_from_slice(&word(3 * 32));
            d.extend_from_slice(&word(per_call as u64));
            d.extend_from_slice(&model);
            let (ret, g) = send(&mut db, d, &format!("step from layer {done}"));
            used += g;
            done = u64::from_be_bytes(ret[24..32].try_into().expect("layer word")) as usize;
        }

        // Final norm, classifier, argmax.
        let mut d = keccak256(
            b"emitToken(bytes32,(uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,uint32,uint32,uint32),bytes,address,uint32,uint32)",
        )[..4]
            .to_vec();
        d.extend_from_slice(&ID);
        for f in [dim, hidden, n_heads, n_kv, head_dim, positions, n_layers] {
            d.extend_from_slice(&word(f as u64));
        }
        d.extend_from_slice(&signed(tensor_kernel::fixed::RMS_EPS_QWEN3));
        // nExperts, topK, expertStride. Zero is what dense means, and this model is dense.
        for _ in 0..3 {
            d.extend_from_slice(&word(0));
        }
        // Three words further along than it was, the config having grown by three.
        d.extend_from_slice(&word(16 * 32)); // offset to finalNorm
        d.extend_from_slice(&addr_word(HEAD));
        d.extend_from_slice(&word(0)); // offset within the classifier blob
        d.extend_from_slice(&word(vocab as u64));
        d.extend_from_slice(&dyn_bytes(&packed(&final_norm)));
        let (ret, last) = send(&mut db, d, "emitToken");
        used += last;
        total_gas += used;

        got_token = u32::from_be_bytes(ret[28..32].try_into().expect("token word"));
        let want = expect[pos];
        println!(
            "{pos:<10} {used:>14} {got_token:>10} {want:>10}{}",
            if got_token == want {
                ""
            } else {
                "   <-- differs"
            }
        );
        assert_eq!(got_token, want, "position {pos} produced the wrong token");
    }

    println!("\n{:<30} {:>14}", "total gas, all positions", total_gas);
    println!(
        "{:<30} {:>14}",
        "gas per token",
        total_gas / positions as u64
    );
    println!("\nevery position matched float64's argmax");
    let _ = got_token;
}
