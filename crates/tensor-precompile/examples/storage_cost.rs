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
use revm::{ExecuteEvm, MainBuilder, MainContext};
use std::path::{Path, PathBuf};
use std::process::Command;
use tensor_precompile::state::TensorPrecompiles;

const CALLER: Address = Address::new([0x11; 20]);
const GEN: Address = Address::new([0x66; 20]);
const WEIGHTS: Address = Address::new([0x33; 20]);
const GAS_CAP: u64 = 400_000_000;

/// Qwen3-0.6B.
const DIM: usize = 1024;
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
            code: Some(Bytecode::new_raw(Bytes::from(vec![0u8; 1024]))),
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

    // start(bytes32,address,bytes) — writes the residual stream once.
    let mut data = keccak256(b"start(bytes32,address,bytes)")[..4].to_vec();
    data.extend_from_slice(&[0x01; 32]); // session id
    let mut a = [0u8; 32];
    a[12..].copy_from_slice(WEIGHTS.as_slice());
    data.extend_from_slice(&a);
    data.extend_from_slice(&word(3 * 32)); // offset to the embedding
    data.extend_from_slice(&word(DIM as u64 * 8));
    // A plausible embedding: non-zero throughout, so every slot is a zero-to-non-zero write
    // — which is the case that costs 22,100 rather than 2,900.
    for i in 0..DIM {
        data.extend_from_slice(&((i as i64 % 1000 - 500) << 20).to_be_bytes());
    }

    let evm = Context::mainnet()
        .with_db(db)
        .modify_cfg_chained(|c| c.tx_gas_limit_cap = Some(GAS_CAP))
        .build_mainnet();
    let mut evm = evm.with_precompiles(TensorPrecompiles::default());
    let result = evm
        .transact(TxEnv {
            caller: CALLER,
            kind: TxKind::Call(GEN),
            data: data.clone().into(),
            gas_limit: GAS_CAP,
            gas_price: 0,
            ..Default::default()
        })
        .expect("transaction must execute")
        .result;

    assert!(
        result.is_success(),
        "start failed: {:?}",
        match &result {
            ExecutionResult::Revert { output, .. } => format!("{output:?}"),
            other => format!("{other:?}"),
        }
    );

    let start_gas = result.tx_gas_used();
    // Everything the transaction did that was not storage: intrinsic, calldata, dispatch.
    let calldata: u64 = 21_000
        + data
            .iter()
            .map(|b| if *b == 0 { 4 } else { 16 })
            .sum::<u64>();
    let resid_slots = DIM / 4;

    println!("checkpointing a forward pass\n");
    println!("{:<44} {:>14}", "quantity", "gas");
    println!("{}", "-".repeat(60));
    println!("{:<44} {start_gas:>14}", "start() total");
    println!("{:<44} {calldata:>14}", "  intrinsic and calldata");
    println!(
        "{:<44} {:>14}",
        format!("  writing {resid_slots} residual slots"),
        start_gas.saturating_sub(calldata)
    );
    println!(
        "{:<44} {:>14}",
        "  per slot",
        (start_gas.saturating_sub(calldata)) / resid_slots as u64
    );

    // ---- what that implies for a token ----
    let per_slot = (start_gas.saturating_sub(calldata)) / resid_slots as u64;
    let kv_slots_per_layer = KV_WIDTH / 4;
    // Keys and values, every layer, once per position.
    let kv_per_token = 2 * kv_slots_per_layer * LAYERS;
    let kv_gas = kv_per_token as u64 * per_slot;
    let compute = LAYER_COMPUTE * LAYERS as u64;

    println!("\none token of Qwen3-0.6B\n");
    println!("{:<44} {:>14}", "quantity", "gas");
    println!("{}", "-".repeat(60));
    println!(
        "{:<44} {:>14}",
        format!("computation, {LAYERS} layers"),
        compute
    );
    println!(
        "{:<44} {:>14}",
        format!("KV cache, {kv_per_token} fresh slots"),
        kv_gas
    );
    println!(
        "{:<44} {:>14}",
        "residual checkpoints, one per transaction",
        per_slot * resid_slots as u64
    );
    println!("{}", "-".repeat(60));
    println!(
        "{:<44} {:>14}",
        "total",
        compute + kv_gas + per_slot * resid_slots as u64
    );
    println!(
        "\n  storage is {:.0}% of a token, and the cache is {:.0}% of that.",
        100.0 * (kv_gas + per_slot * resid_slots as u64) as f64
            / (compute + kv_gas + per_slot * resid_slots as u64) as f64,
        100.0 * kv_gas as f64 / (kv_gas + per_slot * resid_slots as u64) as f64
    );
    println!(
        "\nThe cache cannot be recomputed without redoing every earlier position, and it has\n\
         to outlive the transaction because the next token reads it. Compressing it is the\n\
         thing to do before pushing this to long sequences — the same conclusion the\n\
         companion project reached, now with a number attached."
    );
}
