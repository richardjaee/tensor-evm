//! What a matmul actually costs inside a forward pass.
//!
//! Phase 2 measured a transaction sent straight to the precompile, where the activation
//! vector arrives as transaction calldata at 16 gas per byte. That is the right price for
//! the *first* call of a token and the wrong price for every call after it, because a
//! forward pass does not re-enter the transaction between layers. The contract holds the
//! residual stream in memory and hands it to the precompile from there.
//!
//! Arguments passed from memory are priced as memory, not as calldata. This measures the
//! difference by running the same call a varying number of times within one transaction and
//! taking the increment, which isolates the marginal cost of a call in the inner loop from
//! every one-off cost around it.
//!
//! The result is that the overhead is flat — 217 gas, unchanged from 32 columns to 2048 —
//! so activations do not need to follow the weights into state. The projection that said
//! they did came from applying transaction-calldata pricing to calls that never touch
//! transaction calldata.

use revm::context::{Context, TxEnv};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{ExecuteEvm, MainBuilder, MainContext};
use tensor_precompile::gas_cost;
use tensor_precompile::state::{encode_state, weight_blob, TensorPrecompiles};

const CALLER: Address = Address::new([0x11; 20]);
const WEIGHTS: Address = Address::new([0x33; 20]);

/// `n` back-to-back calls to the state matmul, with the arguments carried in the
/// contract's own code and copied into memory once.
///
/// Embedding them matters for the measurement rather than for realism. Sending them as
/// transaction calldata triggers EIP-7623's floor: the transaction pays the greater of what
/// it executed and a figure derived from its input, and at any real activation width the
/// floor exceeds the execution entirely — one call and five then report identical gas,
/// which reads as "calls are free" when it actually means "nothing was measured". Carrying
/// the arguments in code leaves the transaction empty, so what the runs differ by is only
/// what the extra calls cost.
///
/// ```text
///   61 ll ll     PUSH2 len
///   61 oo oo     PUSH2 blob offset within this code
///   5f           PUSH0        ; destOffset
///   39           codecopy     ; arguments into memory[0..len]
///   repeated n times:
///     5f           retSize   0
///     61 80 00     retOffset 0x8000
///     61 ll ll     argsSize  len
///     5f           argsOffset 0
///     61 01 01     address   0x101
///     5a           gas
///     fa           staticcall
///     50           discard success
///   00           stop
///   <blob>
/// ```
fn forwarder(n: usize, args: &[u8]) -> Vec<u8> {
    let len = u16::try_from(args.len()).expect("arguments must fit a PUSH2");
    let [lh, ll] = len.to_be_bytes();

    // Header length is fixed, so the blob offset is known before the blob is appended.
    let prologue = 8usize;
    let body = 14usize * n;
    let off = u16::try_from(prologue + body + 1).expect("code must fit a PUSH2");
    let [oh, ol] = off.to_be_bytes();

    let mut code = vec![0x61, lh, ll, 0x61, oh, ol, 0x5f, 0x39];
    for _ in 0..n {
        code.extend_from_slice(&[
            0x5f, // retSize
            0x61, 0x80, 0x00, // retOffset 0x8000
            0x61, lh, ll,   // argsSize
            0x5f, // argsOffset
            0x61, 0x01, 0x01, // 0x101
            0x5a, // gas
            0xfa, // staticcall
            0x50, // pop
        ]);
    }
    code.push(0x00);
    assert_eq!(
        code.len(),
        off as usize,
        "blob offset must match the emitted code"
    );
    code.extend_from_slice(args);
    code
}

/// A blob as a real upload leaves it: a leading tag byte marking this as a chunk rather
/// than a directory of chunks. Without it the matmul reverts, and because the forwarder
/// discards the success flag that shows up as a suspiciously cheap loop rather than as a
/// failure — which is the exact shape of a bug this file was written to catch.
fn tagged(mut b: Vec<u8>) -> Vec<u8> {
    b.insert(0, 0x00);
    b
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

fn gas_for(calls: usize, rows: usize, cols: usize) -> u64 {
    let (scales, weights, acts) = fixture(rows, cols);
    let contract = Address::new([0x44; 20]);

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
            code: Some(Bytecode::new_raw(Bytes::from(tagged(weight_blob(
                rows, cols, &scales, &weights,
            ))))),
            ..Default::default()
        },
    );
    let args = encode_state(rows, cols, WEIGHTS, 0, &acts);
    db.insert_account_info(
        contract,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(forwarder(calls, &args)))),
            ..Default::default()
        },
    );

    let evm = Context::mainnet().with_db(db).build_mainnet();
    let mut evm = evm.with_precompiles(TensorPrecompiles::default());

    let result = evm
        .transact(TxEnv {
            caller: CALLER,
            kind: TxKind::Call(contract),
            // Empty, so EIP-7623's floor cannot mask what execution actually cost.
            data: Default::default(),
            gas_limit: 16_000_000,
            gas_price: 0,
            ..Default::default()
        })
        .expect("transaction must execute")
        .result;

    assert!(result.is_success(), "expected success, got {result:?}");
    result.tx_gas_used()
}

/// The number that matters for a forward pass.
///
/// The shape is chosen so execution exceeds EIP-7623's calldata floor. A transaction pays
/// the greater of what it executes and a floor derived from its input, and with a wide
/// activation vector the floor swallows the entire measurement — one call and five report
/// identical gas, which reads as "calls are free" rather than "the floor is binding". Few
/// columns and many rows puts the arithmetic above the floor where it can be seen.
#[test]
fn marginal_cost_of_a_call_in_the_inner_loop() {
    let (rows, cols) = (65536, 32);

    let one = gas_for(1, rows, cols);
    let five = gas_for(5, rows, cols);
    let marginal = (five - one) / 4;
    let compute = gas_cost(rows as u64, cols as u64).unwrap();

    println!("shape                       : {rows} x {cols}");
    println!("one call in a transaction   : {one}");
    println!("five calls                  : {five}");
    println!("marginal cost per call      : {marginal}");
    println!("  of which matmul           : {compute}");
    println!("  of which everything else  : {}", marginal - compute);

    // Arguments come from memory, which was already paid for. Were a 128-byte activation
    // vector priced like transaction calldata, every call would carry that cost again.
    assert!(
        marginal < compute * 2,
        "marginal call cost {marginal} exceeds the arithmetic {compute}, so something \
         other than the matmul is setting the price"
    );
}

/// The decisive one: does handing over a wider activation vector cost more?
///
/// If arguments were priced as transaction calldata, going from 32 to 2048 columns would
/// add roughly 8192 bytes at 16 gas each — about 131,000 gas per call, which at hundreds of
/// calls per token would dominate everything. If they are priced as memory, the increment
/// is a copy: three gas per 32-byte word, some two orders of magnitude less.
///
/// This is what decides whether activations need to move into state as the weights did.
#[test]
fn widening_the_activations_does_not_reprice_the_call() {
    // Constant arithmetic, sixty-four times the activation width.
    let narrow = (4096usize, 32usize);
    let wide = (64usize, 2048usize);

    let m_narrow = (gas_for(5, narrow.0, narrow.1) - gas_for(1, narrow.0, narrow.1)) / 4;
    let m_wide = (gas_for(5, wide.0, wide.1) - gas_for(1, wide.0, wide.1)) / 4;

    let c_narrow = gas_cost(narrow.0 as u64, narrow.1 as u64).unwrap();
    let c_wide = gas_cost(wide.0 as u64, wide.1 as u64).unwrap();

    let o_narrow = m_narrow.saturating_sub(c_narrow);
    let o_wide = m_wide.saturating_sub(c_wide);

    println!(
        "cols {:>5}: marginal {m_narrow:>7}, matmul {c_narrow:>6}, overhead {o_narrow:>7}",
        narrow.1
    );
    println!(
        "cols {:>5}: marginal {m_wide:>7}, matmul {c_wide:>6}, overhead {o_wide:>7}",
        wide.1
    );

    // Calldata pricing would put the wide overhead north of 100,000.
    assert!(
        o_wide < 20_000,
        "overhead {o_wide} at 2048 columns looks like calldata pricing, not memory"
    );
}

/// The marginal cost must track the matmul rather than the width of the activations, or
/// transport is still setting the price and phase 2 did not finish the job.
#[test]
fn marginal_cost_tracks_arithmetic_not_activation_width() {
    // Same activation width, sixteen times the arithmetic. Both above the calldata floor.
    let narrow = (16384usize, 32usize);
    let wide = (262144usize, 32usize);

    let m_narrow = (gas_for(5, narrow.0, narrow.1) - gas_for(1, narrow.0, narrow.1)) / 4;
    let m_wide = (gas_for(5, wide.0, wide.1) - gas_for(1, wide.0, wide.1)) / 4;

    let c_narrow = gas_cost(narrow.0 as u64, narrow.1 as u64).unwrap();
    let c_wide = gas_cost(wide.0 as u64, wide.1 as u64).unwrap();

    println!(
        "rows {:>5}: marginal {m_narrow:>6}, matmul {c_narrow:>6}",
        narrow.0
    );
    println!(
        "rows {:>5}: marginal {m_wide:>6}, matmul {c_wide:>6}",
        wide.0
    );

    // Sixteen times the work must show up as roughly sixteen times the marginal cost,
    // net of the flat per-call overhead that both pay.
    let overhead = m_narrow.saturating_sub(c_narrow);
    let scaled = m_wide.saturating_sub(overhead);
    assert!(
        scaled * 4 > c_wide,
        "marginal cost {m_wide} did not track the arithmetic {c_wide}"
    );
}
