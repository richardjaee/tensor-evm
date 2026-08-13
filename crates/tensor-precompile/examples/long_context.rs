//! How a generation's cost grows with the sequence behind it.
//!
//! Everything measured so far ran at most five positions, and the claims made about long
//! context were reasoning rather than measurement: that the key/value cache grows linearly,
//! that reading it costs account touches rather than bytes, and that attention's own
//! arithmetic is not what dominates. Those are exactly the kind of claims this project has
//! had to withdraw before.
//!
//! So this generates a long sequence and reports the cost of each position, plus the tokens,
//! so a divergence at depth shows up as a wrong token rather than as a plausible number.
//!
//! ```text
//! python3 tools/synth_model.py --out models/long --positions 64
//! cargo run --release --example long_context -- models/long
//! ```

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
const GAS_CAP: u64 = 4_000_000_000;
const ID: [u8; 32] = [0x21; 32];

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
            "{}: {e} — run tools/synth_model.py",
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
        .unwrap_or_else(|| "models/long".to_string())
        .into();
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

    println!("sequence of {positions} positions, {n_layers} layers, kv width {kv_width}\n");

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
            assert_eq!(raw.len(), rows + rows * cols * 2);
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
    let want: Vec<u32> = read(&dir, "tokens.bin")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
        .collect();

    let config = |out: &mut Vec<u8>| {
        for f in [dim, hidden, n_heads, n_kv, head_dim, positions, n_layers] {
            out.extend_from_slice(&word(f as u64));
        }
        out.extend_from_slice(&signed(eps));
        // nExperts, topK, expertStride. Zero is what dense means, and this model is dense.
        for _ in 0..3 {
            out.extend_from_slice(&word(0));
        }
    };

    let mut nonce = 0u64;
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
            other => panic!("{what} failed at: {other:?}"),
        };
        (ret, used)
    };

    println!(
        "{:<10} {:>12} {:>12} {:>8}",
        "position", "step gas", "emit gas", "token"
    );
    println!("{}", "-".repeat(46));

    let mut first_step = 0u64;
    let mut last_step = 0u64;
    for pos in 0..positions {
        let sel: &[u8] = if pos == 0 {
            &keccak256(b"start(bytes32,bytes)")[..4]
        } else {
            &keccak256(b"feed(bytes32,bytes)")[..4]
        };
        let mut d = sel.to_vec();
        d.extend_from_slice(&ID);
        d.extend_from_slice(&word(2 * 32));
        d.extend_from_slice(&dyn_bytes(&packed(&embed[pos * dim..(pos + 1) * dim])));
        send(&mut db, d, "seed");

        let mut model = Vec::new();
        config(&mut model);
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
        let head = 13 * 32;
        model.extend_from_slice(&word(head as u64));
        model.extend_from_slice(&word((head + layers.len()) as u64));
        model.extend_from_slice(&layers);
        model.extend_from_slice(&rope);

        let sig = "step(bytes32,((uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,\
uint32,uint32,uint32),\
(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32)[],\
(bytes,bytes)),uint16)";
        let mut d = keccak256(sig.as_bytes())[..4].to_vec();
        d.extend_from_slice(&ID);
        d.extend_from_slice(&word(3 * 32));
        d.extend_from_slice(&word(n_layers as u64));
        d.extend_from_slice(&model);
        let (_, step_gas) = send(&mut db, d, &format!("step at {pos}"));

        let mut d = keccak256(
            b"emitToken(bytes32,(uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,uint32,uint32,uint32),bytes,address,uint32,uint32)",
        )[..4]
            .to_vec();
        d.extend_from_slice(&ID);
        config(&mut d);
        // Three words further along than it was, the config having grown by three.
        d.extend_from_slice(&word(16 * 32));
        d.extend_from_slice(&addr_word(HEAD));
        d.extend_from_slice(&word(0));
        d.extend_from_slice(&word(vocab as u64));
        d.extend_from_slice(&dyn_bytes(&packed(&final_norm)));
        let (ret, emit_gas) = send(&mut db, d, "emitToken");
        let token = u32::from_be_bytes(ret[28..32].try_into().expect("token word"));

        assert_eq!(token, want[pos], "position {pos} produced the wrong token");
        if pos == 0 {
            first_step = step_gas;
        }
        last_step = step_gas;
        if pos < 4 || pos % 8 == 7 || pos == positions - 1 {
            println!("{pos:<10} {step_gas:>12} {emit_gas:>12} {token:>8}");
        }
    }

    println!("\nevery position matched float64\n");

    // What the growth actually is. Reading the cache touches one account per earlier
    // position per layer at 2,600 gas apiece, which was expected to dominate. It does not:
    // it is a little under half, and the rest is attention's own arithmetic plus the copy
    // that assembles the cache in memory. Worth stating plainly, because the reasoning that
    // said "account touches, not bytes" was half right and was about to become a design
    // decision.
    let slope = (last_step - first_step) as f64 / (positions - 1) as f64;
    let predicted = 2600.0 * n_layers as f64;
    println!("{:<40} {:>12}", "step at position 0", first_step);
    println!(
        "{:<40} {:>12}",
        format!("step at position {}", positions - 1),
        last_step
    );
    println!("{:<40} {:>12.0}", "measured growth per position", slope);
    println!(
        "{:<40} {:>12.0}",
        "cold account touches alone would be", predicted
    );
    println!(
        "\n  account touches are {:.0}% of the growth; the rest is attention and the copy.",
        100.0 * predicted / slope
    );
    println!(
        "\nGrowth is linear in positions rather than quadratic, which is the important part:\n\
         a position reads every earlier one but writes only itself. The slope is real though.\n\
         Scaled to Qwen3-0.6B's twenty-eight layers it is around 73k gas per position of\n\
         context from account touches alone, so a long prompt costs more per token than a\n\
         short one by a margin that grows without bound.\n\
         \n\
         Two things would flatten it. Packing several positions into one deployment trades\n\
         touches for rewriting, and is only worth it in batches. Latent attention shrinks\n\
         what is read at all. Neither is a change to this contract; both are changes to what\n\
         the cache is."
    );
}
