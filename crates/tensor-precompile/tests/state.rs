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

/// A blob as a real upload leaves it: a leading tag byte marking this as a chunk rather
/// than a directory of chunks. The harness places code directly where a chain would run
/// `WeightStore`, so it has to produce the same bytes.
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
            code: Some(Bytecode::new_raw(Bytes::from(tagged(blob)))),
            ..Default::default()
        },
    );
    db
}

fn run(db: CacheDB<EmptyDB>, to: Address, data: Vec<u8>, gas_limit: u64) -> ExecutionResult {
    // Reading a real layer's tensor is past EIP-7825's 2^24 cap, which is one of the limits
    // this chain raises on purpose. A harness that kept it could only test toy shapes.
    let evm = Context::mainnet()
        .with_db(db)
        .modify_cfg_chained(|c| c.tx_gas_limit_cap = Some(4_000_000_000))
        .build_mainnet();
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

    let mut expected = vec![0i64; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    let db = db_with_weights(weight_blob(rows, cols, &scales, &weights));
    let input = encode_state(rows, cols, WEIGHTS, 0, &acts);

    // Calldata now carries only the activations and a 32-byte header.
    assert_eq!(input.len(), 32 + cols * 8);

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
    // rows*cols. That is the asymptotic win: transport no longer grows with the model, and
    // the ratio widens with every row added.
    assert!(
        state_input.len() * 50 < calldata_input.len(),
        "{} against {}",
        state_input.len(),
        calldata_input.len()
    );
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

    let mut expected = vec![0i64; rows];
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

    let mut expected = vec![0i64; rows];
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

/// A blob split across several chunk accounts, with a directory listing them.
///
/// This is how every real model is stored — a Qwen3-0.6B layer is 31.5MB against a 24,000
/// byte chunk limit — and until this test existed, every test in this repository used a
/// single chunk instead. The path all production traffic takes was the one path never
/// exercised.
fn db_with_directory(blob: Vec<u8>, chunk: usize) -> CacheDB<EmptyDB> {
    let mut db = CacheDB::new(EmptyDB::default());
    db.insert_account_info(
        CALLER,
        AccountInfo {
            balance: U256::from(1u64) << 100,
            ..Default::default()
        },
    );
    // Twenty bytes of address then four of cumulative blob length, which is what the
    // uploader writes and what lets a reader seek rather than walk.
    let mut dir = vec![0x01u8]; // TAG_DIRECTORY
    let mut total = 0u32;
    for (i, part) in blob.chunks(chunk).enumerate() {
        let mut a = [0u8; 20];
        a[0] = 0xc0;
        a[18..].copy_from_slice(&(i as u16).to_be_bytes());
        let at = Address::new(a);
        db.insert_account_info(
            at,
            AccountInfo {
                code: Some(Bytecode::new_raw(Bytes::from(tagged(part.to_vec())))),
                ..Default::default()
            },
        );
        total += part.len() as u32;
        dir.extend_from_slice(at.as_slice());
        dir.extend_from_slice(&total.to_be_bytes());
    }
    db.insert_account_info(
        WEIGHTS,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(dir))),
            ..Default::default()
        },
    );
    db
}

/// A directory must give exactly what one chunk gives, at every offset and every split.
///
/// Chunk sizes are chosen so the boundary lands inside a weight row rather than between
/// rows, which is the case a size that happened to divide the tensor evenly would miss.
#[test]
fn a_directory_of_chunks_reads_the_same_as_one() {
    let (rows, cols) = (64, 96);
    let (scales, weights, acts) = fixture(rows, cols);

    let mut expected = vec![0i64; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    let blob = weight_blob(rows, cols, &scales, &weights);
    let input = encode_state(rows, cols, WEIGHTS, 0, &acts);

    for chunk in [7usize, 64, 1000, 4096, blob.len() - 1] {
        let result = run(
            db_with_directory(blob.clone(), chunk),
            MATMUL_STATE_ADDRESS,
            input.clone(),
            16_000_000,
        );
        let got = decode(output(&result)).expect("output must decode");
        assert_eq!(got, expected, "{chunk}-byte chunks gave a different answer");
    }
}

/// The same, but reading a tensor that does not start at the beginning of the blob — which
/// is every tensor in a real layer except the first.
#[test]
fn a_directory_reads_correctly_from_an_offset() {
    let (rows, cols) = (32, 48);
    let (scales, weights, acts) = fixture(rows, cols);

    let mut expected = vec![0i64; rows];
    matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

    // A blob holding the tensor after some other tensor's bytes, as a layer's does.
    let lead = 1234usize;
    let mut blob = vec![0xabu8; lead];
    blob.extend(weight_blob(rows, cols, &scales, &weights));

    for chunk in [17usize, 256, 999] {
        let input = encode_state(rows, cols, WEIGHTS, lead as u32, &acts);
        let result = run(
            db_with_directory(blob.clone(), chunk),
            MATMUL_STATE_ADDRESS,
            input,
            16_000_000,
        );
        let got = decode(output(&result)).expect("output must decode");
        assert_eq!(
            got, expected,
            "{chunk}-byte chunks at offset {lead} gave a different answer"
        );
    }
}

/// A directory at the scale a real layer reaches: 1,337 chunks of 23,552 bytes.
///
/// The fixtures above are a few kilobytes, and this is 31.5MB — the size at which the
/// seeking directory format was introduced and therefore the size at which its arithmetic
/// has to be checked. Reads are taken at the offsets a real layer uses, including the last
/// tensor, which is where a resolver that mis-seeks will land short.
#[test]
fn resolves_at_the_scale_of_a_real_layer() {
    const CHUNK: usize = 23_552;
    const CHUNKS: usize = 1_337;
    let total = 31_488_000usize;

    // Deterministic bytes, so a mis-seek gives the wrong values rather than plausible ones.
    let blob: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();

    let mut db = CacheDB::new(EmptyDB::default());
    db.insert_account_info(
        CALLER,
        AccountInfo {
            balance: U256::from(1u64) << 100,
            ..Default::default()
        },
    );
    let mut dir = vec![0x01u8];
    let mut cum = 0u32;
    for (i, part) in blob.chunks(CHUNK).enumerate() {
        let mut a = [0u8; 20];
        a[0] = 0xc0;
        a[16..].copy_from_slice(&(i as u32).to_be_bytes());
        let at = Address::new(a);
        db.insert_account_info(
            at,
            AccountInfo {
                code: Some(Bytecode::new_raw(Bytes::from(tagged(part.to_vec())))),
                ..Default::default()
            },
        );
        cum += part.len() as u32;
        dir.extend_from_slice(at.as_slice());
        dir.extend_from_slice(&cum.to_be_bytes());
    }
    assert_eq!(dir.len(), 1 + CHUNKS * 24, "directory size");
    db.insert_account_info(
        WEIGHTS,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(dir))),
            ..Default::default()
        },
    );

    // Qwen3-0.6B's own layout: the seven projections in the order a layer stores them.
    let (dim, hidden, q_width, kv_width) = (1024usize, 3072usize, 2048usize, 1024usize);
    let shapes = [
        (q_width, dim),
        (kv_width, dim),
        (kv_width, dim),
        (dim, q_width),
        (hidden, dim),
        (hidden, dim),
        (dim, hidden),
    ];
    let mut offset = 0usize;
    for (rows, cols) in shapes {
        let acts: Vec<i64> = (0..cols).map(|i| ((i as i64 * 7) % 1000) - 500).collect();
        let input = encode_state(rows, cols, WEIGHTS, offset as u32, &acts);

        let span = rows + rows * cols * 2;
        let scales: Vec<i8> = blob[offset..offset + rows]
            .iter()
            .map(|&b| b as i8)
            .collect();
        let weights: Vec<i16> = blob[offset + rows..offset + span]
            .chunks_exact(2)
            .map(|c| i16::from_be_bytes([c[0], c[1]]))
            .collect();
        let mut expected = vec![0i64; rows];
        matmul_scalar(&weights, &scales, &acts, rows, cols, &mut expected);

        let result = run(db.clone(), MATMUL_STATE_ADDRESS, input, 1_000_000_000);
        let got = decode(output(&result)).expect("output must decode");
        assert_eq!(got, expected, "{rows}x{cols} at offset {offset}");
        offset += span;
    }
    assert!(offset < total, "the layout must fit the blob");
}

/// The fused feed-forward must equal the three calls it replaces, exactly.
///
/// Not to a tolerance. It is the same arithmetic in the same order against the same weights;
/// the only thing removed is Solidity carrying `hidden`-wide intermediates across call
/// boundaries. If the two ever differ, the fused path is a second opinion about what a layer
/// computes rather than a cheaper way to reach the first.
#[test]
fn the_fused_feed_forward_equals_the_calls_it_replaces() {
    use tensor_precompile::state::{ffn_state_with, Outcome, FFN_HEADER};

    for (dim, hidden) in [(16usize, 24usize), (64, 128), (128, 384)] {
        let (gs, gw, x) = fixture(hidden, dim);
        let (us, uw, _) = fixture(hidden, dim);
        let (ds, dw, _) = fixture(dim, hidden);

        // One blob holding all three, as a layer does.
        let mut blob = Vec::new();
        let g_off = blob.len();
        blob.extend(weight_blob(hidden, dim, &gs, &gw));
        let u_off = blob.len();
        blob.extend(weight_blob(hidden, dim, &us, &uw));
        let d_off = blob.len();
        blob.extend(weight_blob(dim, hidden, &ds, &dw));

        // What the separate path produces.
        let (mut gate, mut up) = (vec![0i64; hidden], vec![0i64; hidden]);
        matmul_scalar(&gw, &gs, &x, hidden, dim, &mut gate);
        matmul_scalar(&uw, &us, &x, hidden, dim, &mut up);
        tensor_kernel::swiglu_vec(&mut gate, &up);
        let mut want = vec![0i64; dim];
        matmul_scalar(&dw, &ds, &gate, dim, hidden, &mut want);

        let mut input = Vec::with_capacity(FFN_HEADER + dim * 8);
        input.extend_from_slice(&(hidden as u32).to_be_bytes());
        input.extend_from_slice(&(dim as u32).to_be_bytes());
        input.extend_from_slice(WEIGHTS.as_slice());
        for o in [g_off, u_off, d_off] {
            input.extend_from_slice(&(o as u32).to_be_bytes());
        }
        for v in &x {
            input.extend_from_slice(&v.to_be_bytes());
        }

        let tagged_blob = tagged(blob);
        let mut code_of =
            |_a: Address| -> Result<(Bytes, bool), String> { Ok((tagged_blob.clone().into(), false)) };
        let got = match ffn_state_with(&mut code_of, &input, 1_000_000_000).expect("runs") {
            Outcome::Return(bytes, _) => bytes
                .chunks_exact(8)
                .map(|c| i64::from_be_bytes(c.try_into().expect("eight")))
                .collect::<Vec<i64>>(),
            _ => panic!("the fused feed-forward refused {dim}x{hidden}"),
        };
        assert_eq!(got, want, "{dim}x{hidden}");
    }
}

/// Attention fused with its output projection must equal the two calls it replaces.
///
/// Exactly, at every shape. Nothing but attention ever consumes an attention result, so
/// keeping it inside the VM changes what a layer costs and must not change what it computes.
#[test]
fn fused_attention_equals_attention_then_projection() {
    use tensor_precompile::attention::attend;
    use tensor_precompile::state::{attn_proj_state_with, Outcome, ATTN_PROJ_HEADER};

    for (heads, n_kv, hs, positions, dim) in [
        (4usize, 2usize, 8usize, 3usize, 16usize),
        (16, 8, 32, 5, 64),
    ] {
        let q_len = heads * hs;
        let cache = positions * n_kv * hs;
        let (os, ow, _) = fixture(dim, q_len);

        let vals = |n: usize, seed: i64| -> Vec<i64> {
            (0..n)
                .map(|i| ((i as i64 * 40503 + seed) % 131071) - 65535)
                .collect()
        };
        let (q, k, v) = (vals(q_len, 7), vals(cache, 11), vals(cache, 13));

        // The two calls, separately.
        let attn = attend(q.clone(), k.clone(), v.clone(), heads, n_kv, hs, positions);
        let mut want = vec![0i64; dim];
        matmul_scalar(&ow, &os, &attn, dim, q_len, &mut want);

        let mut input = Vec::new();
        for f in [heads, n_kv, hs, positions, dim] {
            input.extend_from_slice(&(f as u32).to_be_bytes());
        }
        input.extend_from_slice(WEIGHTS.as_slice());
        input.extend_from_slice(&0u32.to_be_bytes());
        for value in q.iter().chain(&k).chain(&v) {
            input.extend_from_slice(&value.to_be_bytes());
        }
        assert_eq!(input.len(), ATTN_PROJ_HEADER + (q_len + cache * 2) * 8);

        let blob = tagged(weight_blob(dim, q_len, &os, &ow));
        let mut code_of =
            |_a: Address| -> Result<(Bytes, bool), String> { Ok((blob.clone().into(), false)) };
        let got = match attn_proj_state_with(&mut code_of, &input, 1_000_000_000).expect("runs") {
            Outcome::Return(bytes, _) => bytes
                .chunks_exact(8)
                .map(|c| i64::from_be_bytes(c.try_into().expect("eight")))
                .collect::<Vec<i64>>(),
            _ => panic!("refused {heads}x{hs} over {positions}"),
        };
        assert_eq!(got, want, "{heads} heads, {positions} positions, dim {dim}");
    }
}

/// The whole-layer precompile must equal `forward_layer` called directly.
///
/// It *is* `forward_layer`, so this is not testing the arithmetic. It tests everything
/// around it: that the header is parsed into the shapes it names, that the cache arrives in
/// the right slots, and that the residual stream and the new cache entry come back in the
/// order the caller unpacks them. Marshalling is where every bug in this project has been.
#[test]
fn the_whole_layer_precompile_equals_calling_the_reference() {
    use tensor_model::qwen3::{Config, KvCache, LayerWeights, QTensor, Rope, Scratch};
    use tensor_precompile::state::{layer_state_with, Outcome, LAYER_HEADER};

    let (dim, hidden, heads, n_kv, hs, positions) =
        (16usize, 32usize, 4usize, 2usize, 8usize, 3usize);
    let cfg = Config {
        dim,
        hidden,
        n_heads: heads,
        n_kv,
        head_dim: hs,
        seq_len: positions,
        rms_eps: tensor_kernel::fixed::RMS_EPS_QWEN3,
    };
    let (qw, kvw, half, past) = (cfg.q_width(), cfg.kv_width(), hs / 2, positions - 1);

    let shapes = [
        (qw, dim),
        (kvw, dim),
        (kvw, dim),
        (dim, qw),
        (hidden, dim),
        (hidden, dim),
        (dim, hidden),
    ];
    let mut blob = Vec::new();
    let mut off = Vec::new();
    let mut owned = Vec::new();
    for (rows, cols) in shapes {
        let (s, w, _) = fixture(rows, cols);
        off.push(blob.len());
        blob.extend(weight_blob(rows, cols, &s, &w));
        owned.push((s, w));
    }
    let norm = |n: usize, seed: i64| -> Vec<i64> {
        (0..n)
            .map(|i| (1i64 << 32) + (((i as i64 + seed) % 5) << 30))
            .collect()
    };
    let norms: Vec<Vec<i64>> = [(dim, 1i64), (hs, 2), (hs, 3), (dim, 4)]
        .iter()
        .map(|&(n, s)| norm(n, s))
        .collect();
    for nv in &norms {
        off.push(blob.len());
        for v in nv {
            blob.extend_from_slice(&v.to_be_bytes());
        }
    }

    let vals = |n: usize, seed: i64| -> Vec<i64> {
        (0..n)
            .map(|i| (((i as i64 * 53 + seed) % 13) - 6) << 29)
            .collect()
    };
    let (x0, cos, sin) = (vals(dim, 1), vals(half, 2), vals(half, 3));
    // With room for the entry this call computes, as a contract's buffer has.
    let (pk, pv) = (vals(positions * kvw, 4), vals(positions * kvw, 5));

    // The reference, called directly.
    let t = |i: usize, rows: usize, cols: usize| QTensor {
        w: &owned[i].1,
        scales: &owned[i].0,
        rows,
        cols,
    };
    let w = LayerWeights {
        input_norm: &norms[0],
        q: t(0, qw, dim),
        k: t(1, kvw, dim),
        v: t(2, kvw, dim),
        q_norm: &norms[1],
        k_norm: &norms[2],
        o: t(3, dim, qw),
        post_norm: &norms[3],
        gate: t(4, hidden, dim),
        up: t(5, hidden, dim),
        down: t(6, dim, hidden),
        mixture: None,
    };
    let mut kv = KvCache::new(&cfg, 1);
    kv.k[..past * kvw].copy_from_slice(&pk[..past * kvw]);
    kv.v[..past * kvw].copy_from_slice(&pv[..past * kvw]);
    // The reference wants the whole table; the precompile is handed one row and places it.
    let mut cos_tab = vec![0i64; positions * half];
    let mut sin_tab = vec![0i64; positions * half];
    cos_tab[past * half..].copy_from_slice(&cos);
    sin_tab[past * half..].copy_from_slice(&sin);
    let rope = Rope {
        cos: cos_tab,
        sin: sin_tab,
        head_dim: hs,
    };
    let mut scratch = Scratch::new(&cfg);
    let mut want_x = x0.clone();
    tensor_model::qwen3::forward_layer(
        &cfg,
        &w,
        &rope,
        &mut kv,
        &mut scratch,
        0,
        past,
        &mut want_x,
    );
    let want_k = kv.k[past * kvw..(past + 1) * kvw].to_vec();
    let want_v = kv.v[past * kvw..(past + 1) * kvw].to_vec();

    let mut input = Vec::new();
    for f in [dim, hidden, heads, n_kv, hs, positions] {
        input.extend_from_slice(&(f as u32).to_be_bytes());
    }
    input.extend_from_slice(WEIGHTS.as_slice());
    for o in &off {
        input.extend_from_slice(&(*o as u32).to_be_bytes());
    }
    input.extend_from_slice(&cfg.rms_eps.to_be_bytes());
    for v in x0.iter().chain(&cos).chain(&sin).chain(&pk).chain(&pv) {
        input.extend_from_slice(&v.to_be_bytes());
    }
    assert_eq!(
        input.len(),
        LAYER_HEADER + (dim + half * 2 + positions * kvw * 2) * 8
    );

    let tagged_blob = tagged(blob);
    let mut code_of =
        |_a: Address| -> Result<(Bytes, bool), String> { Ok((tagged_blob.clone().into(), false)) };
    let got: Vec<i64> = match layer_state_with(&mut code_of, &input, 1_000_000_000).expect("runs") {
        Outcome::Return(bytes, _) => bytes
            .chunks_exact(8)
            .map(|c| i64::from_be_bytes(c.try_into().expect("eight")))
            .collect(),
        _ => panic!("the whole-layer precompile refused a well-formed call"),
    };

    assert_eq!(&got[..dim], &want_x[..], "residual stream");
    assert_eq!(&got[dim..dim + kvw], &want_k[..], "cache key");
    assert_eq!(&got[dim + kvw..], &want_v[..], "cache value");
}
