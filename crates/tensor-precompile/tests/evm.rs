//! End to end through a real EVM.
//!
//! The unit tests prove the precompile computes correctly. These prove it is *reachable*:
//! registered at its address, warm, priced by the interpreter's gas accounting, and
//! callable from contract bytecode the way Solidity would call it. That is the actual
//! integration risk — everything else is a library that happens to have an EVM-shaped
//! signature.

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::handler::EthPrecompiles;
use revm::precompile::{PrecompileSpecId, Precompiles};
use revm::primitives::{hardfork::SpecId, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{ExecuteEvm, MainBuilder, MainContext};
use std::sync::OnceLock;
use tensor_kernel::matmul_scalar;
use tensor_precompile::{decode, encode, gas_cost, precompile, BASE_GAS, MATMUL_ADDRESS};

/// Ethereum's precompiles plus ours.
///
/// Leaked deliberately: `EthPrecompiles` holds a `&'static Precompiles`, and a chain's
/// precompile set is fixed for the life of the process, so a single initialisation is the
/// natural lifetime rather than a leak in any meaningful sense.
fn precompiles() -> &'static Precompiles {
    static SET: OnceLock<Precompiles> = OnceLock::new();
    SET.get_or_init(|| {
        let mut p = Precompiles::new(PrecompileSpecId::from_spec_id(SpecId::default())).clone();
        p.extend([precompile()]);
        p
    })
}

const CALLER: Address = Address::new([0x11; 20]);
const CONTRACT: Address = Address::new([0x22; 20]);

/// Bytecode that forwards its calldata to the precompile and returns whatever comes back.
///
/// This is what `solc` emits for a bare `staticcall` to a fixed address, minus the ABI
/// decoding — the point is to exercise the same path a contract would take, not to
/// reproduce a particular compiler's output.
///
/// ```text
///   36 5f 5f 37        calldatacopy(0, 0, calldatasize())
///   5f                 retSize   = 0   (unknown until the call returns)
///   5f                 retOffset = 0
///   36                 argsSize  = calldatasize()
///   5f                 argsOffset= 0
///   61 0100            address   = 0x100
///   5a                 gas       = all remaining
///   fa                 staticcall
///   50                 discard the success flag
///   3d 5f 5f 3e        returndatacopy(0, 0, returndatasize())
///   3d 5f f3           return(0, returndatasize())
/// ```
const FORWARDER: &[u8] = &[
    0x36, 0x5f, 0x5f, 0x37, // calldatacopy
    0x5f, 0x5f, 0x36, 0x5f, 0x61, 0x01, 0x00, 0x5a, 0xfa, // staticcall(0x100)
    0x50, // pop success
    0x3d, 0x5f, 0x5f, 0x3e, // returndatacopy
    0x3d, 0x5f, 0xf3, // return
];

fn db() -> CacheDB<EmptyDB> {
    let mut db = CacheDB::new(EmptyDB::default());
    db.insert_account_info(
        CALLER,
        AccountInfo {
            balance: U256::from(1u64) << 100,
            ..Default::default()
        },
    );
    db.insert_account_info(
        CONTRACT,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from_static(FORWARDER))),
            ..Default::default()
        },
    );
    db
}

/// Runs one transaction against an EVM carrying the tensor precompile, surfacing
/// validation failures rather than hiding them — some of the properties under test are
/// about the transaction being refused before it executes.
fn try_run(
    to: Address,
    data: Vec<u8>,
    gas_limit: u64,
) -> Result<ExecutionResult, Box<dyn std::fmt::Debug>> {
    let mut evm = Context::mainnet().with_db(db()).build_mainnet();
    // The whole point of the fork: swap the precompile set for one that includes ours.
    evm.precompiles = EthPrecompiles {
        precompiles: precompiles(),
        spec: SpecId::default(),
    };

    evm.transact(TxEnv {
        caller: CALLER,
        kind: TxKind::Call(to),
        data: data.into(),
        gas_limit,
        gas_price: 0,
        ..Default::default()
    })
    .map(|o| o.result)
    .map_err(|e| Box::new(format!("{e:?}")) as Box<dyn std::fmt::Debug>)
}

fn run(to: Address, data: Vec<u8>, gas_limit: u64) -> ExecutionResult {
    try_run(to, data, gas_limit).expect("transaction must execute")
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

fn output(result: &ExecutionResult) -> &[u8] {
    match result {
        ExecutionResult::Success { output, .. } => match output {
            Output::Call(b) => b,
            Output::Create(b, _) => b,
        },
        other => panic!("expected success, got {other:?}"),
    }
}

#[test]
fn reachable_at_its_address() {
    let (rows, cols) = (16, 24);
    let (scales, weights, acts) = fixture(rows, cols);

    let mut expected = vec![0i64; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    let result = run(
        MATMUL_ADDRESS,
        encode(rows, cols, &scales, &weights, &acts),
        10_000_000,
    );
    let got = decode(output(&result)).expect("output must decode");

    assert_eq!(got, expected, "precompile disagreed with the kernel");
}

/// The gate for this phase: a contract, not a transaction, gets the right bytes back.
#[test]
fn callable_from_contract_bytecode() {
    let (rows, cols) = (32, 48);
    let (scales, weights, acts) = fixture(rows, cols);

    let mut expected = vec![0i64; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    let result = run(
        CONTRACT,
        encode(rows, cols, &scales, &weights, &acts),
        10_000_000,
    );
    let got = decode(output(&result)).expect("output must decode");

    assert_eq!(got, expected, "value returned through STATICCALL was wrong");
}

/// Gas must be charged by the interpreter, not merely computed by the precompile.
#[test]
fn the_interpreter_charges_for_it() {
    let (rows, cols) = (64, 64);
    let (scales, weights, acts) = fixture(rows, cols);
    let input = encode(rows, cols, &scales, &weights, &acts);

    let result = run(MATMUL_ADDRESS, input.clone(), 10_000_000);
    let used = result.tx_gas_used();
    let priced = gas_cost(rows as u64, cols as u64).unwrap();

    assert!(
        used > priced,
        "total {used} should exceed the precompile's own {priced}, since intrinsic and \
         calldata costs are charged on top"
    );
    assert!(
        used < priced + 10_000_000,
        "total gas {used} is implausible"
    );
}

/// Quadrupling the work must quadruple the *marginal* price. Comparing totals would be
/// meaningless while the flat base cost dominates, which it does at these shapes — that
/// domination is measured in `calldata_dwarfs_compute` rather than asserted away here.
#[test]
fn price_scales_with_shape() {
    let small = gas_cost(64, 64).unwrap() - BASE_GAS;
    let large = gas_cost(128, 128).unwrap() - BASE_GAS;
    assert_eq!(
        large,
        small * 4,
        "four times the work should cost four times as much above base"
    );
}

/// The finding that decides the next phase.
///
/// Passing weights as calldata costs orders of magnitude more than multiplying by them.
/// Calldata is priced per byte at rates set for scarce block space; the matmul is priced
/// against measured throughput. At any real model width the transport dominates so
/// completely that making the arithmetic free changes nothing.
///
/// The conclusion is that weights cannot travel as arguments. They have to already be on
/// the chain, with the precompile reading them from state and the caller passing only a
/// reference — which is what phase 2 builds. This test pins the ratio so the argument
/// stays grounded in a number.
#[test]
fn calldata_dwarfs_compute() {
    let (rows, cols) = (256, 256);
    let (scales, weights, acts) = fixture(rows, cols);
    let input = encode(rows, cols, &scales, &weights, &acts);

    let total = run(MATMUL_ADDRESS, input.clone(), 16_000_000).tx_gas_used();
    let compute = gas_cost(rows as u64, cols as u64).unwrap();
    let transport = total - compute;

    // Recorded rather than merely asserted: this ratio is the reason for phase 2.
    println!("input bytes      : {}", input.len());
    println!("total gas        : {total}");
    println!("  matmul         : {compute}");
    println!("  transport etc  : {transport}");
    println!("  ratio          : {}x", transport / compute);

    assert!(
        transport > compute * 1000,
        "expected transport to dominate by orders of magnitude, got {transport} against \
         {compute}"
    );
}

/// Malformed input must revert the calling frame rather than taking down execution.
#[test]
fn malformed_input_reverts_the_transaction() {
    let result = run(MATMUL_ADDRESS, vec![0xff; 9], 1_000_000);
    assert!(
        matches!(result, ExecutionResult::Revert { .. }),
        "expected a revert, got {result:?}"
    );
}

/// The same finding from the other direction, and the sharper statement of it.
///
/// A natural test would be to starve the precompile: fund the calldata, withhold the gas
/// for the arithmetic, and check nothing is computed for free. That test cannot be
/// written. EIP-7623 imposes a floor on a transaction's gas derived from its calldata, and
/// at any real width that floor already exceeds the cost of the matmul by four orders of
/// magnitude — so a limit low enough to starve the arithmetic is rejected before execution
/// begins.
///
/// Which is the argument for phase 2 stated as a property rather than a ratio: while
/// weights travel as arguments, their transport cost so dominates that the price of the
/// computation is not merely small, it is unreachable.
#[test]
fn the_calldata_floor_makes_the_matmul_unpriceable() {
    let (rows, cols) = (256, 256);
    let (scales, weights, acts) = fixture(rows, cols);
    let input = encode(rows, cols, &scales, &weights, &acts);

    let enough = run(MATMUL_ADDRESS, input.clone(), 16_000_000).tx_gas_used();
    let compute = gas_cost(rows as u64, cols as u64).unwrap();

    // Ask for everything except the arithmetic. Validation refuses.
    let err = try_run(MATMUL_ADDRESS, input, enough - compute / 2)
        .expect_err("a limit below the calldata floor must be rejected");

    assert!(
        format!("{err:?}").contains("GasFloor"),
        "expected the calldata floor to reject this, got {err:?}"
    );
}
