//! `WeightStore`, at the sizes a real model actually reaches.
//!
//! Every other test places weights in the database directly, which is the right shortcut
//! for testing the precompile and the wrong one for testing the uploader — the bytes are
//! assumed correct rather than produced. So this runs the contract.
//!
//! The size that matters is not the chunk's, which is capped at 24,000 and was always
//! within the deploy prologue's two-byte length field. It is the directory's. Qwen3-0.6B's
//! classifier is 311MB and needs 13,219 chunks, so its directory is 264,381 bytes, and a
//! `uint16` length turned that into 2,237. The deploy succeeded, an address came back, and
//! the stored directory listed 111 of its chunks. Nothing reverted; the classifier read
//! weights that were not there.
//!
//! A directory is the only blob that grows without bound here, and it grows with the model,
//! so this is the case no small fixture can reach and every real model does.

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{DatabaseCommit, ExecuteEvm, MainBuilder, MainContext};
use std::path::PathBuf;
use std::process::Command;
use tensor_precompile::state::TensorPrecompiles;

const CALLER: Address = Address::new([0x11; 20]);
const UPLOADER: Address = Address::new([0x99; 20]);
/// Uploading a directory of a quarter of a megabyte costs 200 gas a byte to deposit, so the
/// cap here is well above Ethereum's and matches what this chain sets.
const GAS: u64 = 2_000_000_000;

fn uploader_runtime() -> Vec<u8> {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("contracts/src/WeightStore.sol");
    let out = Command::new("solc")
        .args(["--optimize", "--optimize-runs", "200", "--bin-runtime"])
        .arg(&src)
        .output()
        .expect("solc must be installed to run this test");
    assert!(
        out.status.success(),
        "solc failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("utf8");
    let at = text.find(":Uploader =======").expect("Uploader section");
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

fn addr_word(a: Address) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(a.as_slice());
    w
}

struct Harness {
    db: CacheDB<EmptyDB>,
    nonce: u64,
}

impl Harness {
    fn new() -> Self {
        let mut db = CacheDB::new(EmptyDB::default());
        db.insert_account_info(
            CALLER,
            AccountInfo {
                balance: U256::from(1u64) << 100,
                ..Default::default()
            },
        );
        db.insert_account_info(
            UPLOADER,
            AccountInfo {
                code: Some(Bytecode::new_raw(Bytes::from(uploader_runtime()))),
                ..Default::default()
            },
        );
        Self { db, nonce: 0 }
    }

    fn call(&mut self, data: Vec<u8>) -> ExecutionResult {
        let out = {
            let evm = Context::mainnet()
                .with_db(&mut self.db)
                .modify_cfg_chained(|c| {
                    c.tx_gas_limit_cap = Some(GAS);
                    // The chain this targets raises the code size limit; a directory for a
                    // real classifier is far past Ethereum's 24,576.
                    c.limit_contract_code_size = Some(64 * 1024 * 1024);
                })
                .build_mainnet();
            let mut evm = evm.with_precompiles(TensorPrecompiles::default());
            evm.transact(TxEnv {
                caller: CALLER,
                kind: TxKind::Call(UPLOADER),
                data: data.into(),
                gas_limit: GAS,
                gas_price: 0,
                nonce: self.nonce,
                ..Default::default()
            })
            .expect("transaction must execute")
        };
        self.nonce += 1;
        let result = out.result.clone();
        self.db.commit(out.state);
        result
    }

    fn code_len(&mut self, at: Address) -> usize {
        self.db
            .load_account(at)
            .ok()
            .and_then(|a| a.info.code.clone())
            .map(|c| c.original_bytes().len())
            .unwrap_or(0)
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

/// A directory listing `n` chunks must deploy as exactly `1 + 20n` bytes.
///
/// The addresses need not exist: a directory is a list, and what is being checked is that
/// the list survives being written. Sizes bracket the two-byte boundary at 65,536 bytes —
/// 3,276 entries is just under, 3,277 just over, and 13,219 is the real classifier's.
#[test]
fn a_directory_is_not_truncated_by_its_length_field() {
    let mut h = Harness::new();
    let finish = keccak256(b"finish(address[])")[..4].to_vec();

    for n in [1usize, 100, 3_276, 3_277, 4_000, 13_219] {
        let mut d = finish.clone();
        d.extend_from_slice(&word(32));
        d.extend_from_slice(&word(n as u64));
        for i in 0..n {
            let mut a = [0u8; 20];
            a[0] = 0xc0;
            a[16..].copy_from_slice(&(i as u32).to_be_bytes());
            d.extend_from_slice(&addr_word(Address::new(a)));
        }

        let result = h.call(d);
        let out = returned(&result);
        let ptr = Address::from_slice(&out[12..32]);
        assert_eq!(
            h.code_len(ptr),
            1 + 20 * n,
            "a directory of {n} chunks deployed at the wrong size"
        );
    }
}

/// A chunk must deploy as exactly its bytes plus the tag, at every size up to the cap.
#[test]
fn a_chunk_deploys_at_its_stated_size() {
    let mut h = Harness::new();
    let store = keccak256(b"store(bytes)")[..4].to_vec();

    for n in [1usize, 32, 255, 256, 24_000] {
        let mut d = store.clone();
        d.extend_from_slice(&word(32));
        d.extend_from_slice(&word(n as u64));
        let mut body = vec![0xa5u8; n];
        body.extend(std::iter::repeat_n(0u8, (32 - n % 32) % 32));
        d.extend_from_slice(&body);

        let result = h.call(d);
        let ptr = Address::from_slice(&returned(&result)[12..32]);
        assert_eq!(h.code_len(ptr), 1 + n, "a {n}-byte chunk deployed wrong");
    }
}

/// Past the cap it must revert rather than store a truncated chunk.
#[test]
fn an_oversized_chunk_is_refused() {
    let mut h = Harness::new();
    let mut d = keccak256(b"store(bytes)")[..4].to_vec();
    let n = 24_001usize;
    d.extend_from_slice(&word(32));
    d.extend_from_slice(&word(n as u64));
    let mut body = vec![0xa5u8; n];
    body.extend(std::iter::repeat_n(0u8, (32 - n % 32) % 32));
    d.extend_from_slice(&body);

    assert!(
        matches!(h.call(d), ExecutionResult::Revert { .. }),
        "an oversized chunk must be refused, not truncated"
    );
}
