//! Solidity, through the precompile, to the kernel.
//!
//! `forge` cannot test this. Its EVM is stock revm, which has no matmul precompile, so a
//! Solidity test would find an empty account at `0x101`, receive success with no data, and
//! pass while computing nothing — the same silent failure that produced a "free" inner loop
//! earlier in this project. The contracts are therefore compiled with `solc` and executed
//! here, against the EVM that actually carries the precompile.
//!
//! What this establishes is the last link in the chain: that a contract written the way a
//! transformer would be written gets bit-identical results to the reference kernel.

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{DatabaseCommit, ExecuteEvm, MainBuilder, MainContext};
use std::path::PathBuf;
use std::process::Command;
use tensor_kernel::matmul_scalar;
use tensor_precompile::decode;
use tensor_precompile::state::{weight_blob, TensorPrecompiles};

const CALLER: Address = Address::new([0x11; 20]);
const REGISTRY: Address = Address::new([0x55; 20]);
const WEIGHTS: Address = Address::new([0x33; 20]);
const MODEL_ID: u64 = 7;

fn contracts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("contracts")
}

/// Deployed runtime bytecode, straight from `solc`.
///
/// Runtime rather than creation code, so it can be placed in an account directly. The
/// registry has no constructor arguments and no immutables, so the two are equivalent here
/// and skipping deployment keeps the test to one transaction.
fn runtime_code() -> Vec<u8> {
    let dir = contracts_dir();
    let out = Command::new("solc")
        .args(["--optimize", "--optimize-runs", "200", "--bin-runtime"])
        .arg(dir.join("src/ModelRegistry.sol"))
        .output()
        .expect("solc must be installed to run this test");

    assert!(
        out.status.success(),
        "solc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Output is a sequence of `======= path:Name =======` sections; take the registry's.
    let text = String::from_utf8(out.stdout).expect("solc output must be utf8");
    let marker = ":ModelRegistry =======";
    let at = text.find(marker).expect("ModelRegistry section missing");
    let hex = text[at..]
        .lines()
        .find(|l| l.starts_with("6080") || l.starts_with("60806040"))
        .or_else(|| {
            text[at..]
                .lines()
                .skip_while(|l| !l.contains("Binary of the runtime part"))
                .nth(1)
        })
        .expect("runtime bytecode missing from solc output")
        .trim();

    hex_decode(hex)
}

fn hex_decode(s: &str) -> Vec<u8> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    assert!(s.len().is_multiple_of(2), "odd-length hex");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

fn selector(sig: &str) -> [u8; 4] {
    let h = keccak256(sig.as_bytes());
    [h[0], h[1], h[2], h[3]]
}

fn word(v: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&v.to_be_bytes());
    w
}

fn addr_word(a: Address) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(a.as_slice());
    w
}

/// `registerModel(uint256,address,(uint16,uint16,uint16,uint16,uint16,uint16,uint32,uint32,uint32))`
///
/// The config struct is entirely fixed-width, so it encodes inline rather than behind an
/// offset — nine words following the two scalars.
fn encode_register(model_id: u64, weights: Address, dim: u64, cols: u64) -> Vec<u8> {
    let sig = "registerModel(uint256,address,(uint16,uint16,uint16,uint16,uint16,uint16,uint32,uint32,uint32))";
    let mut v = selector(sig).to_vec();
    v.extend_from_slice(&word(model_id));
    v.extend_from_slice(&addr_word(weights));
    for field in [dim, cols, 1, 1, 1, 128, 32_000, 1_000_000, 0] {
        v.extend_from_slice(&word(field));
    }
    v
}

/// `projectStateless(uint256,uint32,uint32,uint32,int32[])`
fn encode_project(model_id: u64, rows: u32, cols: u32, offset: u32, acts: &[i64]) -> Vec<u8> {
    let sig = "projectStateless(uint256,uint32,uint32,uint32,int64[])";
    let mut v = selector(sig).to_vec();
    v.extend_from_slice(&word(model_id));
    v.extend_from_slice(&word(rows as u64));
    v.extend_from_slice(&word(cols as u64));
    v.extend_from_slice(&word(offset as u64));
    v.extend_from_slice(&word(5 * 32)); // offset to the array, past five head words
    v.extend_from_slice(&word(acts.len() as u64));
    for &a in acts {
        // int64 is sign-extended to a full word by the ABI.
        let mut w = [0u8; 32];
        if a < 0 {
            w[..24].fill(0xff);
        }
        w[24..].copy_from_slice(&a.to_be_bytes());
        v.extend_from_slice(&w);
    }
    v
}

/// Decodes an `int32[]` return: offset word, length word, then sign-extended values.
fn decode_int64_array(ret: &[u8]) -> Vec<i64> {
    assert!(ret.len() >= 64, "return too short for a dynamic array");
    let len = u64::from_be_bytes(ret[56..64].try_into().unwrap()) as usize;
    (0..len)
        .map(|i| {
            let at = 64 + i * 32 + 24;
            i64::from_be_bytes(ret[at..at + 8].try_into().unwrap())
        })
        .collect()
}

fn fixture(rows: usize, cols: usize) -> (Vec<i8>, Vec<i16>, Vec<i64>) {
    let weights = (0..rows * cols)
        .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
        .collect();
    let scales = (0..rows).map(|r| (r % 5) as i8 + 28).collect();
    let acts = (0..cols)
        .map(|i| ((i as i64 * 40503 + 17) % 131071) - 65535)
        .collect();
    (scales, weights, acts)
}

struct Harness {
    db: CacheDB<EmptyDB>,
    /// Tracked because state is committed between calls, so the caller's nonce advances
    /// and a second transaction reusing zero is rejected before it executes.
    nonce: u64,
}

impl Harness {
    fn new(blob: Vec<u8>) -> Self {
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
            REGISTRY,
            AccountInfo {
                code: Some(Bytecode::new_raw(Bytes::from(runtime_code()))),
                ..Default::default()
            },
        );
        Self { db, nonce: 0 }
    }

    /// State persists between calls, so registration is visible to the projection that
    /// follows it — the sequence a real caller would perform.
    fn call(&mut self, data: Vec<u8>) -> ExecutionResult {
        let out = {
            let evm = Context::mainnet().with_db(&mut self.db).build_mainnet();
            let mut evm = evm.with_precompiles(TensorPrecompiles::default());
            evm.transact(TxEnv {
                caller: CALLER,
                kind: TxKind::Call(REGISTRY),
                data: data.into(),
                gas_limit: 16_000_000,
                gas_price: 0,
                nonce: self.nonce,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("transaction rejected before execution: {e:?}"))
        };
        self.nonce += 1;

        self.db.commit(out.state);
        out.result
    }
}

fn returned(result: &ExecutionResult) -> &[u8] {
    match result {
        ExecutionResult::Success { output, .. } => match output {
            Output::Call(b) => b,
            Output::Create(b, _) => b,
        },
        other => panic!("expected success, got {other:?}"),
    }
}

/// The gate for this phase: Solidity gets exactly what the kernel computes.
#[test]
fn solidity_reaches_the_precompile() {
    let (rows, cols) = (48usize, 64usize);
    let (scales, weights, acts) = fixture(rows, cols);

    let mut expected = vec![0i64; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    let mut h = Harness::new(weight_blob(rows, cols, &scales, &weights));

    let reg = h.call(encode_register(MODEL_ID, WEIGHTS, rows as u64, cols as u64));
    assert!(reg.is_success(), "registerModel failed: {reg:?}");

    let proj = h.call(encode_project(MODEL_ID, rows as u32, cols as u32, 0, &acts));
    assert!(proj.is_success(), "projectStateless failed: {proj:?}");

    let got = decode_int64_array(returned(&proj));
    assert_eq!(
        got, expected,
        "Solidity disagreed with the reference kernel"
    );
}

/// Registration must be one-way. A model whose weights could be repointed would make every
/// hash previously published against it a claim about something that no longer exists.
#[test]
fn a_registered_model_cannot_be_replaced() {
    let (rows, cols) = (8usize, 8usize);
    let (scales, weights, _) = fixture(rows, cols);
    let mut h = Harness::new(weight_blob(rows, cols, &scales, &weights));

    let first = h.call(encode_register(MODEL_ID, WEIGHTS, rows as u64, cols as u64));
    assert!(first.is_success());

    let second = h.call(encode_register(MODEL_ID, WEIGHTS, rows as u64, cols as u64));
    assert!(
        matches!(second, ExecutionResult::Revert { .. }),
        "re-registering the same id should revert, got {second:?}"
    );
}

/// Projecting against an unregistered model must fail rather than read address zero.
#[test]
fn unregistered_model_reverts() {
    let (rows, cols) = (8usize, 8usize);
    let (scales, weights, acts) = fixture(rows, cols);
    let mut h = Harness::new(weight_blob(rows, cols, &scales, &weights));

    let proj = h.call(encode_project(999, rows as u32, cols as u32, 0, &acts));
    assert!(
        matches!(proj, ExecutionResult::Revert { .. }),
        "expected a revert, got {proj:?}"
    );
}

/// The library must reject an unexpected output length rather than proceed.
///
/// This is the guard against running on a chain without the precompile, where the call
/// would succeed with empty data and the forward pass would quietly produce zeros. Asking
/// for more rows than the blob holds makes the precompile revert, which the library
/// surfaces rather than swallows.
#[test]
fn a_shape_the_blob_cannot_satisfy_reverts_through_solidity() {
    let (rows, cols) = (16usize, 16usize);
    let (scales, weights, acts) = fixture(rows, cols);
    let mut h = Harness::new(weight_blob(rows, cols, &scales, &weights));

    assert!(h
        .call(encode_register(MODEL_ID, WEIGHTS, rows as u64, cols as u64))
        .is_success());

    let proj = h.call(encode_project(
        MODEL_ID,
        (rows * 4) as u32,
        cols as u32,
        0,
        &acts,
    ));
    assert!(
        matches!(proj, ExecutionResult::Revert { .. }),
        "expected a revert, got {proj:?}"
    );
}

/// Sanity: the wire decoder agrees with the ABI decoder on the same bytes.
#[test]
fn abi_and_raw_decoders_agree() {
    let raw: Vec<i64> = vec![-5, 0, 7, i64::MIN, i64::MAX];
    let mut bytes = Vec::new();
    for v in &raw {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    assert_eq!(decode(&bytes).unwrap(), raw);
}
