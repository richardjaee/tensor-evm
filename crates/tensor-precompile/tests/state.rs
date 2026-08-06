//! The state-reading matmul, end to end.
//!
//! Phase 1 established that weights cannot travel as calldata: transport cost exceeded the
//! arithmetic by 31,549x, and EIP-7623's floor put the computation's price out of reach
//! entirely. These tests check the answer — weights resident in chain state, the caller
//! passing a reference — and measure what it recovers.

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{ExecuteEvm, MainBuilder, MainContext};
use tensor_kernel::matmul_scalar;
use tensor_precompile::state::{
    encode_state, weight_blob, TensorPrecompiles, COLD_ACCOUNT_GAS, MATMUL_STATE_ADDRESS,
};
use tensor_precompile::{decode, encode, gas_cost, MATMUL_ADDRESS};

const CALLER: Address = Address::new([0x11; 20]);
const WEIGHTS: Address = Address::new([0x33; 20]);

fn fixture(rows: usize, cols: usize) -> (Vec<i8>, Vec<i16>, Vec<i32>) {
    let weights = (0..rows * cols)
        .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
        .collect();
    let scales = (0..rows).map(|r| (r % 5) as i8 + 28).collect();
    let acts = (0..cols)
        .map(|i| (((i as i64 * 40503 + 17) % 131071) - 65535) as i32)
        .collect();
    (scales, weights, acts)
}

/// Places a weight blob in an account's code, which is how a chain would hold it: read in
/// bulk, contiguous, and identified by a hash that doubles as provenance.
fn db_with_weights(blob: Vec<u8>) -> CacheDB<EmptyDB> {
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
    db
}

fn run(db: CacheDB<EmptyDB>, to: Address, data: Vec<u8>, gas_limit: u64) -> ExecutionResult {
    let evm = Context::mainnet().with_db(db).build_mainnet();
    let mut evm = evm.with_precompiles(TensorPrecompiles::default());

    evm.transact(TxEnv {
        caller: CALLER,
        kind: TxKind::Call(to),
        data: data.into(),
        gas_limit,
        gas_price: 0,
        ..Default::default()
    })
    .expect("transaction must execute")
    .result
}

fn output(result: &ExecutionResult) -> &[u8] {
    match result {
        ExecutionResult::Success { output, .. } => match output {
            Output::Call(b) => b,
            Output::Create(b, _) => b,
        },
        other => panic!("expected success, got {other:?}"),
    }
}

/// The gate for this phase. Weights never appear in the transaction, and the answer is
/// still exactly the kernel's.
#[test]
fn reads_weights_from_state() {
    let (rows, cols) = (64, 96);
    let (scales, weights, acts) = fixture(rows, cols);

    let mut expected = vec![0i32; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    let db = db_with_weights(weight_blob(rows, cols, &scales, &weights));
    let input = encode_state(rows, cols, WEIGHTS, 0, &acts);

    // Calldata now carries only the activations and a 32-byte header.
    assert_eq!(input.len(), 32 + cols * 4);

    let result = run(db, MATMUL_STATE_ADDRESS, input, 10_000_000);
    let got = decode(output(&result)).expect("output must decode");

    assert_eq!(
        got, expected,
        "state-resident weights gave a different answer"
    );
}

/// The measurement this phase exists to produce: what moving weights off the wire saves.
#[test]
fn state_residency_collapses_the_transport_cost() {
    let (rows, cols) = (256, 256);
    let (scales, weights, acts) = fixture(rows, cols);
    let blob = weight_blob(rows, cols, &scales, &weights);

    let calldata_input = encode(rows, cols, &scales, &weights, &acts);
    let state_input = encode_state(rows, cols, WEIGHTS, 0, &acts);

    let calldata_gas = run(
        db_with_weights(blob.clone()),
        MATMUL_ADDRESS,
        calldata_input.clone(),
        16_000_000,
    )
    .tx_gas_used();

    let state_gas = run(
        db_with_weights(blob),
        MATMUL_STATE_ADDRESS,
        state_input.clone(),
        16_000_000,
    )
    .tx_gas_used();

    let compute = gas_cost(rows as u64, cols as u64).unwrap();

    println!("shape                 : {rows} x {cols}");
    println!("calldata form");
    println!("  input bytes         : {}", calldata_input.len());
    println!("  total gas           : {calldata_gas}");
    println!("state form");
    println!("  input bytes         : {}", state_input.len());
    println!("  total gas           : {state_gas}");
    println!("  matmul              : {compute}");
    println!("  cold account        : {COLD_ACCOUNT_GAS}");
    println!("saving                : {}x", calldata_gas / state_gas);

    assert!(
        state_gas * 10 < calldata_gas,
        "state form at {state_gas} should be far below calldata's {calldata_gas}"
    );
    // The remaining cost is dominated by activations, which scale with cols rather than
    // rows*cols. That is the asymptotic win: transport no longer grows with the model.
    assert!(state_input.len() < calldata_input.len() / 100);
}

/// Cost must not depend on how big the weight blob is, only on the shape requested.
/// Otherwise a 64 GB model would be unusable regardless of how little of it a call touches.
#[test]
fn price_is_independent_of_blob_size() {
    let (rows, cols) = (32, 32);
    let (scales, weights, acts) = fixture(rows, cols);

    let small = weight_blob(rows, cols, &scales, &weights);
    let mut large = small.clone();
    large.extend(std::iter::repeat_n(0u8, 200_000)); // same matmul, far more code

    let input = encode_state(rows, cols, WEIGHTS, 0, &acts);
    let a = run(
        db_with_weights(small),
        MATMUL_STATE_ADDRESS,
        input.clone(),
        10_000_000,
    )
    .tx_gas_used();
    let b = run(
        db_with_weights(large),
        MATMUL_STATE_ADDRESS,
        input,
        10_000_000,
    )
    .tx_gas_used();

    assert_eq!(a, b, "price varied with blob size");
}

/// Reading past the end of the blob must be refused, not served whatever follows.
#[test]
fn refuses_a_shape_the_blob_cannot_satisfy() {
    let (rows, cols) = (32, 32);
    let (scales, weights, acts) = fixture(rows, cols);
    let blob = weight_blob(rows, cols, &scales, &weights);

    // Same blob, but ask for twice the rows.
    let input = encode_state(rows * 2, cols, WEIGHTS, 0, &acts);
    let result = run(
        db_with_weights(blob),
        MATMUL_STATE_ADDRESS,
        input,
        10_000_000,
    );

    assert!(
        matches!(result, ExecutionResult::Revert { .. }),
        "expected a revert, got {result:?}"
    );
}

/// An offset lets one blob hold many tensors, which is how a real model would be laid out.
#[test]
fn honours_an_offset_into_the_blob() {
    let (rows, cols) = (16, 24);
    let (scales, weights, acts) = fixture(rows, cols);

    let mut expected = vec![0i32; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    // Pad the front, as a second tensor in the same blob would.
    let pad = 4096usize;
    let mut blob = vec![0xabu8; pad];
    blob.extend(weight_blob(rows, cols, &scales, &weights));

    let input = encode_state(rows, cols, WEIGHTS, pad as u32, &acts);
    let result = run(
        db_with_weights(blob),
        MATMUL_STATE_ADDRESS,
        input,
        10_000_000,
    );
    let got = decode(output(&result)).expect("output must decode");

    assert_eq!(got, expected, "offset was not applied correctly");
}

/// A missing account has empty code, which cannot satisfy any shape.
#[test]
fn missing_weights_account_reverts() {
    let (rows, cols) = (8, 8);
    let (_, _, acts) = fixture(rows, cols);

    let input = encode_state(rows, cols, Address::new([0x99; 20]), 0, &acts);
    let db = db_with_weights(vec![]);
    let result = run(db, MATMUL_STATE_ADDRESS, input, 10_000_000);

    assert!(
        matches!(result, ExecutionResult::Revert { .. }),
        "expected a revert, got {result:?}"
    );
}

/// Ethereum's own precompiles must still work — this provider wraps them rather than
/// replacing them.
#[test]
fn ethereum_precompiles_still_reachable() {
    // identity at 0x04
    let data = b"round trip".to_vec();
    let db = db_with_weights(vec![]);
    let result = run(db, Address::with_last_byte(4), data.clone(), 1_000_000);
    assert_eq!(
        output(&result),
        &data[..],
        "identity precompile was shadowed"
    );
}

/// The calldata form must remain reachable alongside the state form.
#[test]
fn both_forms_coexist() {
    let (rows, cols) = (8, 8);
    let (scales, weights, acts) = fixture(rows, cols);

    let mut expected = vec![0i32; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    let db = db_with_weights(weight_blob(rows, cols, &scales, &weights));
    let result = run(
        db,
        MATMUL_ADDRESS,
        encode(rows, cols, &scales, &weights, &acts),
        10_000_000,
    );
    assert_eq!(decode(output(&result)).unwrap(), expected);
}
