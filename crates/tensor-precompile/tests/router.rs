//! The router, in a real EVM, reading its tensor from chain state.
//!
//! Two things are being established. The first is that the routing decision made inside the
//! VM is the same one the reference makes — if it were not, a contract and an off-chain
//! implementation would disagree about which experts a token visited, which is the failure
//! mode this whole design exists to rule out.
//!
//! The second is the point of routing at all: that the cost of a layer follows the experts
//! chosen and not the experts that exist. A dense layer's projections collectively read the
//! whole blob, which is why seeking past chunks was worth only a sixth of a token. A router
//! that takes two experts of sixty-four never touches the other sixty-two, and those are
//! the megabytes.

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{ExecuteEvm, MainBuilder, MainContext};
use tensor_kernel::matmul_scalar;
use tensor_precompile::router::{decode_output, encode, route, ROUTER_ADDRESS};
use tensor_precompile::state::{
    encode_state, weight_blob, TensorPrecompiles, MATMUL_STATE_ADDRESS,
};

const CALLER: Address = Address::new([0x11; 20]);
const WEIGHTS: Address = Address::new([0x33; 20]);

fn tagged(mut b: Vec<u8>) -> Vec<u8> {
    b.insert(0, 0x00);
    b
}

fn db_with(blob: Vec<u8>) -> CacheDB<EmptyDB> {
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
            code: Some(Bytecode::new_raw(Bytes::from(tagged(blob)))),
            ..Default::default()
        },
    );
    db
}

fn run(db: CacheDB<EmptyDB>, to: Address, data: Vec<u8>) -> ExecutionResult {
    let evm = Context::mainnet()
        .with_db(db)
        .modify_cfg_chained(|c| c.tx_gas_limit_cap = Some(4_000_000_000))
        .build_mainnet();
    let mut evm = evm.with_precompiles(TensorPrecompiles::default());
    evm.transact(TxEnv {
        caller: CALLER,
        kind: TxKind::Call(to),
        data: data.into(),
        gas_limit: 1_000_000_000,
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

/// Weights spread widely enough that the ordering is not decided by rounding, and
/// activations that are not all one sign.
fn fixture(rows: usize, cols: usize) -> (Vec<i8>, Vec<i16>, Vec<i64>) {
    let weights = (0..rows * cols)
        .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
        .collect();
    let scales = (0..rows).map(|r| (r % 5) as i8 + 12).collect();
    let acts = (0..cols)
        .map(|i| ((i as i64 * 40503 + 17) % 131071) - 65535)
        .collect();
    (scales, weights, acts)
}

/// The decision made inside the VM must be the decision the reference makes.
#[test]
fn routes_as_the_reference_does() {
    for (experts, cols, k) in [(8usize, 32usize, 2usize), (64, 128, 8), (3, 16, 3)] {
        let (scales, weights, acts) = fixture(experts, cols);

        let mut scores = vec![0i64; experts];
        matmul_scalar(&weights, &scales, &acts, experts, cols, &mut scores);
        let want = route(&scores, k);

        let db = db_with(weight_blob(experts, cols, &scales, &weights));
        let input = encode(experts as u32, cols as u32, WEIGHTS, 0, k as u32, &acts);
        let result = run(db, ROUTER_ADDRESS, input);
        let got = decode_output(output(&result)).expect("output must decode");

        assert_eq!(got, want, "{experts} experts, top {k}");
        assert_eq!(got.len(), k, "the router must return exactly k experts");
    }
}

/// Asking for more experts than exist is refused rather than quietly satisfied with fewer.
///
/// A caller who received `k - 1` weights would normalise over the wrong count and scale its
/// feed-forward down without noticing.
#[test]
fn refuses_to_choose_more_experts_than_exist() {
    let (experts, cols) = (4usize, 16usize);
    let (scales, weights, acts) = fixture(experts, cols);
    let db = db_with(weight_blob(experts, cols, &scales, &weights));
    let input = encode(experts as u32, cols as u32, WEIGHTS, 0, 5, &acts);
    assert!(
        matches!(
            run(db, ROUTER_ADDRESS, input),
            ExecutionResult::Revert { .. }
        ),
        "k past the expert count must revert"
    );
}

/// A shape the blob cannot satisfy must revert rather than read whatever follows.
#[test]
fn refuses_a_router_larger_than_its_blob() {
    let (experts, cols) = (4usize, 16usize);
    let (scales, weights, acts) = fixture(experts, cols);
    let db = db_with(weight_blob(experts, cols, &scales, &weights));
    let input = encode(experts as u32 * 4, cols as u32, WEIGHTS, 0, 2, &acts);
    assert!(
        matches!(
            run(db, ROUTER_ADDRESS, input),
            ExecutionResult::Revert { .. }
        ),
        "a router past the end of its blob must revert"
    );
}

/// What routing is for: the cost of a layer follows the experts chosen, not the experts
/// that exist.
///
/// A whole mixture layer is run through the precompiles — score, choose, then read only the
/// chosen experts — at two expert counts with the same `k`. Sixty-four experts is eight
/// times the weights of eight experts, and must not be meaningfully more gas.
#[test]
fn cost_follows_the_experts_chosen_not_the_experts_stored() {
    let (dim, hidden, k) = (64usize, 128usize, 2usize);

    let mut measured = Vec::new();
    for &experts in &[8usize, 64usize] {
        // One blob holding the router and then every expert's down-projection, which is the
        // layout a real mixture layer uses: one account, addressed by offset.
        let (r_scales, r_weights, acts) = fixture(experts, dim);
        let mut blob = weight_blob(experts, dim, &r_scales, &r_weights);

        let mut expert_at = Vec::with_capacity(experts);
        for _ in 0..experts {
            let (s, w, _) = fixture(hidden, dim);
            expert_at.push(blob.len());
            blob.extend(weight_blob(hidden, dim, &s, &w));
        }

        let db = db_with(blob);
        let input = encode(experts as u32, dim as u32, WEIGHTS, 0, k as u32, &acts);
        let routed = run(db.clone(), ROUTER_ADDRESS, input);
        let chosen = decode_output(output(&routed)).expect("output must decode");
        assert_eq!(chosen.len(), k);

        let mut gas = routed.tx_gas_used();
        for &(which, _) in &chosen {
            let at = expert_at[which as usize];
            let call = encode_state(hidden, dim, WEIGHTS, at as u32, &acts);
            let r = run(db.clone(), MATMUL_STATE_ADDRESS, call);
            assert!(r.is_success(), "expert {which} must run");
            gas += r.tx_gas_used();
        }
        measured.push((experts, gas));
    }

    let (small_n, small_gas) = measured[0];
    let (big_n, big_gas) = measured[1];
    println!("experts {small_n:>3}  top {k}  gas {small_gas}");
    println!("experts {big_n:>3}  top {k}  gas {big_gas}");
    println!(
        "{}x the experts for {:.2}x the gas",
        big_n / small_n,
        big_gas as f64 / small_gas as f64
    );

    // The router itself grows with the expert count — it scores every one — so some increase
    // is correct. What must not happen is the experts' weights being paid for.
    assert!(
        big_gas < small_gas * 2,
        "{big_n} experts cost {big_gas} against {small_n} experts at {small_gas}: \
         cost is following the experts stored rather than the experts chosen"
    );
}
