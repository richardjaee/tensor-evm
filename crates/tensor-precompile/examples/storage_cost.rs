//! What it costs to make a forward pass resumable.
//!
//! A token is about 76M gas of computation and Ethereum caps a transaction at 2^24, so
//! generation has to span several. Spanning them means the residual stream and the
//! key/value cache live in storage between calls, and storage is priced nothing like
//! memory: a slot that was previously zero costs 22,100 gas to write.
//!
//! This measures the checkpoint rather than reasoning about it, because the arithmetic is
//! easy to get wrong in the optimistic direction — four Q32 values pack into a slot, which
//! makes the residual stream sound cheap right up until it is multiplied by twenty-eight
//! layers and a cache that grows with every position.
//!
//! `cargo run --release --example storage_cost`

use revm::context::{Context, TxEnv};
use revm::context_interface::result::ExecutionResult;
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{DatabaseCommit, ExecuteEvm, MainBuilder, MainContext};
use std::path::{Path, PathBuf};
use std::process::Command;
use tensor_precompile::state::TensorPrecompiles;

const CALLER: Address = Address::new([0x11; 20]);
const GEN: Address = Address::new([0x66; 20]);
const GAS_CAP: u64 = 400_000_000;

/// Qwen3-0.6B: one layer's keys or values for one position.
const KV_WIDTH: usize = 1024; // nKv 8 x headDim 128
const LAYERS: usize = 28;
/// What a layer costs to compute, measured by `onchain_layer`.
const LAYER_COMPUTE: u64 = 688_198;

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

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let code = solc_runtime(&root.join("contracts/src/StorageBench.sol"), "StorageBench");

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
            code: Some(Bytecode::new_raw(Bytes::from(code))),
            ..Default::default()
        },
    );

    // One layer's keys for one position: kvWidth Q32 values.
    let payload: Vec<u8> = (0..KV_WIDTH)
        .flat_map(|i| ((i as i64 % 1000 - 500) << 20).to_be_bytes())
        .collect();
    let slots = payload.len() / 32;

    let mut nonce = 0u64;
    let mut call = |db: &mut CacheDB<EmptyDB>, sel: &[u8], arg: Vec<u8>| -> u64 {
        let mut data = sel.to_vec();
        data.extend_from_slice(&arg);
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
            .expect("must execute")
        };
        nonce += 1;
        let result = out.result.clone();
        db.commit(out.state);
        let ret = match &result {
            ExecutionResult::Success { output, .. } => match output {
                revm::context_interface::result::Output::Call(b) => b.to_vec(),
                revm::context_interface::result::Output::Create(b, _) => b.to_vec(),
            },
            other => panic!("call failed: {other:?}"),
        };
        u64::from_be_bytes(ret[24..32].try_into().unwrap())
    };

    let arg_bytes = {
        let mut a = word(32).to_vec();
        a.extend_from_slice(&word(payload.len() as u64));
        a.extend_from_slice(&payload);
        let pad = (32 - payload.len() % 32) % 32;
        a.extend(std::iter::repeat_n(0u8, pad));
        a
    };

    let store_w = call(
        &mut db,
        &keccak256(b"benchStorage(bytes)")[..4],
        arg_bytes.clone(),
    );
    let store_r = call(
        &mut db,
        &keccak256(b"benchStorageRead(uint256)")[..4],
        word(KV_WIDTH as u64).to_vec(),
    );
    let code_w = call(&mut db, &keccak256(b"benchCode(bytes)")[..4], arg_bytes);
    let code_r = call(
        &mut db,
        &keccak256(b"benchCodeRead(uint256)")[..4],
        word(payload.len() as u64).to_vec(),
    );

    println!(
        "one layer's keys for one position: {} values, {} bytes, {slots} slots\n",
        KV_WIDTH,
        payload.len()
    );
    println!("{:<34} {:>12} {:>12}", "", "storage", "code");
    println!("{}", "-".repeat(60));
    println!("{:<34} {store_w:>12} {code_w:>12}", "write");
    println!("{:<34} {store_r:>12} {code_r:>12}", "read");
    println!(
        "{:<34} {:>12} {:>12}",
        "write, per slot-equivalent",
        store_w / slots as u64,
        code_w / slots as u64
    );

    // ---- what that means for a token ----
    // Every layer contributes keys and values once per position.
    let per_token = 2 * LAYERS as u64;
    let compute = LAYER_COMPUTE * LAYERS as u64;
    let store_token = per_token * (store_w + store_r);
    let code_token = per_token * (code_w + code_r);

    println!("\none token of Qwen3-0.6B, {LAYERS} layers\n");
    println!("{:<34} {:>14}", "computation", compute);
    println!("{:<34} {:>14}", "cache in storage", store_token);
    println!("{:<34} {:>14}", "cache in code", code_token);
    println!("{}", "-".repeat(52));
    println!(
        "{:<34} {:>14}",
        "token, cache in storage",
        compute + store_token
    );
    println!(
        "{:<34} {:>14}",
        "token, cache in code",
        compute + code_token
    );
    println!(
        "\n  moving the cache from storage to code is {:.1}x on the cache",
        store_token as f64 / code_token as f64
    );
    println!(
        "  and {:.1}x on the token.",
        (compute + store_token) as f64 / (compute + code_token) as f64
    );
    println!(
        "\nCode is immutable, which is fatal for the residual stream — rewritten every\n\
         layer — and free for the cache, whose entries are produced once at one position\n\
         and read unchanged forever after. The constraint happens to match the data."
    );
}
