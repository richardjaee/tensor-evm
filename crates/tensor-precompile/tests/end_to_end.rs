//! The whole generation loop, in CI.
//!
//! `full_token` runs the real Qwen3-0.6B checkpoint and is the stronger check, but it needs
//! 1.2 GB that cannot be committed, so it only runs when somebody runs it. That is not a
//! theoretical weakness: a change moving norm weights on chain broke every example while
//! the test suite stayed green, and nothing caught it until the next manual run.
//!
//! So this drives the same path — `start`, `step`, `emitToken`, repeat — against a model
//! small enough to live in the repository, and asserts the tokens float64 says it emits.
//! The model is tiny but not simplified: `head_dim` is independent of `dim / n_heads` so
//! the query and output projections are not square, attention is grouped, norm weights are
//! per layer and far above unit scale, and several positions run so the causal mask and the
//! key/value cache are exercised rather than trivially satisfied at position zero. Every one
//! of those has caused a bug here.
//!
//! Regenerate with `python3 tools/synth_model.py --out testdata/tiny`.

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{DatabaseCommit, ExecuteEvm, MainBuilder, MainContext};
use std::path::{Path, PathBuf};
use std::process::Command;
use tensor_precompile::state::TensorPrecompiles;

const CALLER: Address = Address::new([0x11; 20]);
const GEN: Address = Address::new([0x66; 20]);
const HEAD: Address = Address::new([0x44; 20]);
/// Well above what the model needs; a chain running this sets its own cap, and Ethereum's
/// 2^24 is one of the settings this design already asks to be raised.
const GAS_CAP: u64 = 2_000_000_000;

/// A blob as a real upload leaves it: a leading tag byte marking this as a chunk rather
/// than a directory of chunks. The harness places code directly where a chain would run
/// `WeightStore`, so it has to produce the same bytes.
fn tagged(mut b: Vec<u8>) -> Vec<u8> {
    b.insert(0, 0x00);
    b
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
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
        .expect("solc must be installed to run the end-to-end test");
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

struct Chain {
    db: CacheDB<EmptyDB>,
    nonce: u64,
}

impl Chain {
    fn send(&mut self, data: Vec<u8>, what: &str) -> Vec<u8> {
        let out = {
            let evm = Context::mainnet()
                .with_db(&mut self.db)
                .modify_cfg_chained(|c| c.tx_gas_limit_cap = Some(GAS_CAP))
                .build_mainnet();
            let mut evm = evm.with_precompiles(TensorPrecompiles::default());
            evm.transact(TxEnv {
                caller: CALLER,
                kind: TxKind::Call(GEN),
                data: data.into(),
                gas_limit: GAS_CAP,
                gas_price: 0,
                nonce: self.nonce,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("{what} rejected before execution: {e:?}"))
        };
        self.nonce += 1;
        let result = out.result.clone();
        self.db.commit(out.state);
        match &result {
            ExecutionResult::Success { output, .. } => match output {
                Output::Call(b) => b.to_vec(),
                Output::Create(b, _) => b.to_vec(),
            },
            other => panic!("{what} failed: {other:?}"),
        }
    }
}

/// Generates every position and returns the tokens the chain emitted.
///
/// `per_call` is the granularity knob: how many layers a `step` attempts. The result must
/// not depend on it — a token split across transactions is the same token — which is the
/// property `granularity_does_not_change_the_output` checks.
fn generate(per_call: usize, session: u8) -> Vec<u32> {
    let dir = root().join("testdata/tiny");
    let mj = std::fs::read_to_string(dir.join("meta.json")).expect("meta.json");

    let dim = meta(&mj, "dim");
    let hidden = meta(&mj, "hidden");
    let n_layers = meta(&mj, "n_layers");
    let n_heads = meta(&mj, "n_heads");
    let n_kv = meta(&mj, "n_kv");
    let head_dim = meta(&mj, "head_dim");
    let vocab = meta(&mj, "vocab");
    let positions = meta(&mj, "positions");
    let (q_width, kv_width, half) = (n_heads * head_dim, n_kv * head_dim, head_dim / 2);
    let id = [session; 32];

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
                &root().join("contracts/src/Generator.sol"),
                "Generator",
            )))),
            ..Default::default()
        },
    );
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

    // Seven tensors then four norm vectors, one blob and one address per layer — the layout
    // the chain actually holds.
    let shapes: [(&str, usize, usize); 7] = [
        ("q", q_width, dim),
        ("k", kv_width, dim),
        ("v", kv_width, dim),
        ("o", dim, q_width),
        ("gate", hidden, dim),
        ("up", hidden, dim),
        ("down", dim, hidden),
    ];
    let mut offsets = vec![[0u64; 11]; n_layers];
    for (l, off) in offsets.iter_mut().enumerate() {
        let mut blob = Vec::new();
        for (i, (name, rows, cols)) in shapes.iter().enumerate() {
            let raw = read(&dir, &format!("l{l}_{name}.bin"));
            assert_eq!(raw.len(), rows + rows * cols * 2, "l{l}_{name} shape");
            off[i] = blob.len() as u64;
            blob.extend_from_slice(&raw);
        }
        for (j, name) in ["input_norm", "q_norm", "k_norm", "post_norm"]
            .iter()
            .enumerate()
        {
            off[7 + j] = blob.len() as u64;
            blob.extend(packed(&read_i64(&dir, &format!("l{l}_{name}.bin"))));
        }
        db.insert_account_info(
            layer_addr(l),
            AccountInfo {
                code: Some(Bytecode::new_raw(Bytes::from(tagged(blob)))),
                ..Default::default()
            },
        );
    }

    let final_norm = read_i64(&dir, "final_norm.bin");
    let embed = read_i64(&dir, "embed.bin");
    let cos = read_i64(&dir, "cos.bin");
    let sin = read_i64(&dir, "sin.bin");
    let eps = tensor_kernel::fixed::RMS_EPS_QWEN3;

    let config = |out: &mut Vec<u8>| {
        for f in [dim, hidden, n_heads, n_kv, head_dim, positions, n_layers] {
            out.extend_from_slice(&word(f as u64));
        }
        out.extend_from_slice(&signed(eps));
    };

    let mut chain = Chain { db, nonce: 0 };
    let step_sig = "step(bytes32,((uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256),\
(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32)[],\
(bytes,bytes)),uint16)";
    let emit_sig = "emitToken(bytes32,(uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256),\
bytes,address,uint32,uint32)";

    let mut tokens = Vec::with_capacity(positions);
    for pos in 0..positions {
        let sel: &[u8] = if pos == 0 {
            &keccak256(b"start(bytes32,bytes)")[..4]
        } else {
            &keccak256(b"feed(bytes32,bytes)")[..4]
        };
        let mut d = sel.to_vec();
        d.extend_from_slice(&id);
        d.extend_from_slice(&word(2 * 32));
        d.extend_from_slice(&dyn_bytes(&packed(&embed[pos * dim..(pos + 1) * dim])));
        chain.send(d, "seed");

        // The model, encoded once per position: the rotary table is the only part that
        // depends on where in the sequence we are.
        let mut model = Vec::new();
        config(&mut model);
        let mut layers = word(n_layers as u64).to_vec();
        for (l, off) in offsets.iter().enumerate() {
            layers.extend_from_slice(&addr_word(layer_addr(l)));
            for o in off {
                layers.extend_from_slice(&word(*o));
            }
        }
        let a = pos * half;
        let rope = bytes_tuple(&[&packed(&cos[a..a + half]), &packed(&sin[a..a + half])]);
        let head = 10 * 32;
        model.extend_from_slice(&word(head as u64));
        model.extend_from_slice(&word((head + layers.len()) as u64));
        model.extend_from_slice(&layers);
        model.extend_from_slice(&rope);

        let mut done = 0usize;
        while done < n_layers {
            let mut d = keccak256(step_sig.as_bytes())[..4].to_vec();
            d.extend_from_slice(&id);
            d.extend_from_slice(&word(3 * 32));
            d.extend_from_slice(&word(per_call.min(n_layers) as u64));
            d.extend_from_slice(&model);
            let ret = chain.send(d, &format!("step from layer {done}"));
            let next = u64::from_be_bytes(ret[24..32].try_into().expect("layer word")) as usize;
            assert!(next > done, "step did not advance past layer {done}");
            done = next;
        }

        let mut d = keccak256(emit_sig.as_bytes())[..4].to_vec();
        d.extend_from_slice(&id);
        config(&mut d);
        d.extend_from_slice(&word(13 * 32));
        d.extend_from_slice(&addr_word(HEAD));
        d.extend_from_slice(&word(0));
        d.extend_from_slice(&word(vocab as u64));
        d.extend_from_slice(&dyn_bytes(&packed(&final_norm)));
        let ret = chain.send(d, "emitToken");
        tokens.push(u32::from_be_bytes(
            ret[28..32].try_into().expect("token word"),
        ));
    }
    tokens
}

fn expected() -> Vec<u32> {
    read(&root().join("testdata/tiny"), "tokens.bin")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
        .collect()
}

/// The whole loop: seed, run every layer, classify, repeat — against what float64 emits.
///
/// Checked at every position rather than only the last. A chain that arrived at the right
/// final token by a wrong path would be indistinguishable otherwise, and the key/value
/// cache only starts mattering at the second position.
#[test]
fn generates_the_tokens_float64_produces() {
    assert_eq!(generate(usize::MAX, 0x01), expected());
}

/// Splitting a token across transactions must not change it.
///
/// This is the granularity knob's contract. Fewer layers per call means more checkpoints,
/// more gas, and more intermediate state visible on chain for auditing — but the same
/// token. If it did not hold, the knob would be a correctness setting rather than a cost
/// one, and every checkpoint would be a chance to diverge.
#[test]
fn granularity_does_not_change_the_output() {
    let want = expected();
    for per_call in [1usize, 2, 3] {
        assert_eq!(
            generate(per_call, 0x10 + per_call as u8),
            want,
            "{per_call} layers per call gave different tokens"
        );
    }
}

/// The fixture must exercise the shapes that have actually caused bugs, or passing it
/// proves less than it appears to.
#[test]
fn the_fixture_is_structurally_hard() {
    let mj = std::fs::read_to_string(root().join("testdata/tiny/meta.json")).expect("meta.json");
    let (dim, heads, kv, hs) = (
        meta(&mj, "dim"),
        meta(&mj, "n_heads"),
        meta(&mj, "n_kv"),
        meta(&mj, "head_dim"),
    );
    assert_ne!(hs, dim / heads, "head_dim must not be dim/heads");
    assert_ne!(heads * hs, dim, "query projection must not be square");
    assert!(kv < heads && heads % kv == 0, "attention must be grouped");
    assert!(
        meta(&mj, "n_layers") > 1,
        "one layer would not test per-layer norms"
    );
    assert!(
        meta(&mj, "positions") > 1,
        "one position would not test the cache"
    );

    // Norm weights above unit scale, which is what broke three separate activation bounds.
    let peak = read_i64(&root().join("testdata/tiny"), "l3_post_norm.bin")
        .iter()
        .map(|v| v.abs())
        .max()
        .unwrap_or(0) as f64
        / 4_294_967_296.0;
    assert!(peak > 10.0, "norm weights are too tame at {peak:.1}");
}
