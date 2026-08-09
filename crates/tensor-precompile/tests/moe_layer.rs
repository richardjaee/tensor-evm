//! A mixture-of-experts layer, through the contracts.
//!
//! The router precompile is tested on its own elsewhere. What is checked here is everything
//! around it: that a layer reads the experts the router chose, that expert `e`'s tensors are
//! actually found a stride away from expert zero's, and that the weighted sum lands in the
//! residual stream.
//!
//! The check avoids needing a second reference implementation. With `topK` of one, the
//! chosen expert's routing weight is exactly one, so a mixture layer must produce precisely
//! what a dense layer produces when handed that expert's tensors — the same arithmetic
//! reached by two different paths through the contract. Any error in the stride, the offset
//! arithmetic, or the scaling shows up as a mismatch, and the routing weight being one means
//! the comparison is exact rather than approximate.
//!
//! The expert the router picks is then asserted against the Rust router, so the test cannot
//! pass by routing to expert zero and comparing against expert zero.

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{keccak256, Address, Bytes, TxKind, U256};
use revm::state::{AccountInfo, Bytecode};
use revm::{DatabaseCommit, ExecuteEvm, MainBuilder, MainContext};
use std::path::{Path, PathBuf};
use std::process::Command;
use tensor_kernel::matmul_scalar;
use tensor_precompile::router::route;
use tensor_precompile::state::{weight_blob, TensorPrecompiles};

const CALLER: Address = Address::new([0x11; 20]);
const GEN: Address = Address::new([0x66; 20]);
const LAYER: Address = Address::new([0xa0; 20]);
const GAS_CAP: u64 = 2_000_000_000;

// A model small enough to read by eye and awkward enough to catch a mistake: `head_dim` is
// not `dim / n_heads`, attention is grouped, and there are four experts so the stride is
// exercised past the first.
const DIM: usize = 16;
const HIDDEN: usize = 24;
const N_HEADS: usize = 4;
const N_KV: usize = 2;
const HEAD_DIM: usize = 8;
const EXPERTS: usize = 4;

fn q_width() -> usize {
    N_HEADS * HEAD_DIM
}
fn kv_width() -> usize {
    N_KV * HEAD_DIM
}

fn tagged(mut b: Vec<u8>) -> Vec<u8> {
    b.insert(0, 0x00);
    b
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

fn solc_runtime(src: &Path, name: &str) -> Vec<u8> {
    let out = Command::new("solc")
        .args(["--optimize", "--optimize-runs", "200", "--bin-runtime"])
        .arg(src)
        .output()
        .expect("solc must be installed to run this test");
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

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Deterministic weights, spread widely enough that the router's ordering is decided by the
/// values rather than by rounding.
fn tensor(rows: usize, cols: usize, seed: i64) -> (Vec<i8>, Vec<i16>) {
    let w = (0..rows * cols)
        .map(|i| (((i as i64 * 2654435761 + seed * 104729) % 801) - 400) as i16)
        .collect();
    let scales = (0..rows)
        .map(|r| ((r as i64 + seed) % 3) as i8 + 12)
        .collect();
    (scales, w)
}

fn norm_vec(n: usize, seed: i64) -> Vec<i64> {
    (0..n)
        .map(|i| (1i64 << 32) + (((i as i64 * 37 + seed) % 7) << 29))
        .collect()
}

/// Everything a layer needs, plus where each piece sits in the blob.
struct Layout {
    blob: Vec<u8>,
    off: [u64; 12], // q k v o gate up down inputNorm qNorm kNorm postNorm router
    stride: u64,
    router_scales: Vec<i8>,
    router_weights: Vec<i16>,
}

/// One layer holding attention, four experts, and a router.
///
/// The expert tensors are laid out expert by expert, so expert `e`'s gate, up and down each
/// sit `stride` bytes past expert zero's. That is the layout `Transformer` assumes, and it
/// is why a layer needs one stride rather than a table of offsets.
fn build_layer() -> Layout {
    let mut blob = Vec::new();
    let mut off = [0u64; 12];

    for (i, (rows, cols, seed)) in [
        (q_width(), DIM, 1i64),
        (kv_width(), DIM, 2),
        (kv_width(), DIM, 3),
        (DIM, q_width(), 4),
    ]
    .into_iter()
    .enumerate()
    {
        let (s, w) = tensor(rows, cols, seed);
        off[i] = blob.len() as u64;
        blob.extend(weight_blob(rows, cols, &s, &w));
    }

    // Expert zero's three tensors, then the rest, contiguously.
    let gate_at = blob.len();
    let mut expert_bytes = Vec::new();
    for e in 0..EXPERTS {
        for (rows, cols, seed) in [
            (HIDDEN, DIM, 10 + e as i64),
            (HIDDEN, DIM, 20 + e as i64),
            (DIM, HIDDEN, 30 + e as i64),
        ] {
            let (s, w) = tensor(rows, cols, seed);
            expert_bytes.extend(weight_blob(rows, cols, &s, &w));
        }
        if e == 0 {
            let gate_len = HIDDEN + HIDDEN * DIM * 2;
            let up_len = gate_len;
            off[4] = gate_at as u64;
            off[5] = (gate_at + gate_len) as u64;
            off[6] = (gate_at + gate_len + up_len) as u64;
        }
    }
    let stride = (expert_bytes.len() / EXPERTS) as u64;
    blob.extend(expert_bytes);

    for (i, (n, seed)) in [(DIM, 1i64), (HEAD_DIM, 2), (HEAD_DIM, 3), (DIM, 4)]
        .into_iter()
        .enumerate()
    {
        off[7 + i] = blob.len() as u64;
        blob.extend(packed(&norm_vec(n, seed)));
    }

    let (router_scales, router_weights) = tensor(EXPERTS, DIM, 99);
    off[11] = blob.len() as u64;
    blob.extend(weight_blob(EXPERTS, DIM, &router_scales, &router_weights));

    Layout {
        blob,
        off,
        stride,
        router_scales,
        router_weights,
    }
}

struct Harness {
    db: CacheDB<EmptyDB>,
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
            LAYER,
            AccountInfo {
                code: Some(Bytecode::new_raw(Bytes::from(tagged(blob)))),
                ..Default::default()
            },
        );
        Self { db, nonce: 0 }
    }

    fn send(&mut self, data: Vec<u8>, what: &str) -> Vec<u8> {
        let out = {
            let evm = Context::mainnet()
                .with_db(&mut self.db)
                .modify_cfg_chained(|c| c.tx_gas_limit_cap = Some(GAS_CAP))
                .build_mainnet();
            let mut evm = evm.with_precompiles(TensorPrecompiles::default());
            evm.transact(TxEnv {
                caller: CALLER,
                kind: TxKind::Call(GEN),
                data: data.into(),
                gas_limit: GAS_CAP,
                gas_price: 0,
                nonce: self.nonce,
                ..Default::default()
            })
            .unwrap_or_else(|e| panic!("{what} rejected: {e:?}"))
        };
        self.nonce += 1;
        let result = out.result.clone();
        self.db.commit(out.state);
        match &result {
            ExecutionResult::Success { output, .. } => match output {
                Output::Call(b) => b.to_vec(),
                Output::Create(b, _) => b.to_vec(),
            },
            other => panic!("{what} failed: {other:?}"),
        }
    }
}

const CFG_TY: &str =
    "(uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,uint32,uint32,uint32)";
const LAYER_TY: &str =
    "(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32)";

/// Runs one layer and returns the residual stream it produced.
///
/// `experts` of zero takes the dense path, in which case `gate`, `up` and `down` are used as
/// written. Anything else routes, and the offsets become expert zero's.
fn run_layer(
    h: &mut Harness,
    l: &Layout,
    x: &[i64],
    experts: u32,
    top_k: u32,
    base: u64,
) -> Vec<i64> {
    let id = [0x5au8; 32];

    let mut d = keccak256(b"start(bytes32,bytes)")[..4].to_vec();
    d.extend_from_slice(&id);
    d.extend_from_slice(&word(2 * 32));
    d.extend_from_slice(&dyn_bytes(&packed(x)));
    h.send(d, "start");

    let mut model = Vec::new();
    for f in [DIM, HIDDEN, N_HEADS, N_KV, HEAD_DIM, 1, 1] {
        model.extend_from_slice(&word(f as u64));
    }
    model.extend_from_slice(&signed(tensor_kernel::fixed::RMS_EPS_QWEN3));
    model.extend_from_slice(&word(experts as u64));
    model.extend_from_slice(&word(top_k as u64));
    model.extend_from_slice(&word(l.stride));

    let mut layers = word(1).to_vec();
    layers.extend_from_slice(&addr_word(LAYER));
    for (i, o) in l.off.iter().enumerate() {
        // The dense comparison points gate, up and down at one chosen expert by shifting
        // them a whole number of strides; everything else is where the layout put it.
        let shift = if (4..=6).contains(&i) { base } else { 0 };
        layers.extend_from_slice(&word(o + shift));
    }

    let half = HEAD_DIM / 2;
    let cos: Vec<i64> = (0..half)
        .map(|i| (1i64 << 32) - ((i as i64) << 28))
        .collect();
    let sin: Vec<i64> = (0..half).map(|i| (i as i64) << 27).collect();
    let rope = bytes_tuple(&[&packed(&cos), &packed(&sin)]);

    let head = 13 * 32;
    model.extend_from_slice(&word(head as u64));
    model.extend_from_slice(&word((head + layers.len()) as u64));
    model.extend_from_slice(&layers);
    model.extend_from_slice(&rope);

    let step_sig = format!("step(bytes32,({CFG_TY},{LAYER_TY}[],(bytes,bytes)),uint16)");
    let mut d = keccak256(step_sig.as_bytes())[..4].to_vec();
    d.extend_from_slice(&id);
    d.extend_from_slice(&word(3 * 32));
    d.extend_from_slice(&word(1));
    d.extend_from_slice(&model);
    h.send(d, "step");

    let mut d = keccak256(b"residual(bytes32,uint32)")[..4].to_vec();
    d.extend_from_slice(&id);
    d.extend_from_slice(&word(DIM as u64));
    let ret = h.send(d, "residual");
    // A dynamic `int256[]`: an offset, a length, then the values.
    ret[64..]
        .chunks_exact(32)
        .take(DIM)
        .map(|c| i64::from_be_bytes(c[24..].try_into().expect("eight bytes")))
        .collect()
}

/// Which expert the router will pick for a given input, decided by the same code the VM
/// runs — so the test knows where to point the dense comparison.
fn picks(l: &Layout, x: &[i64]) -> usize {
    let mut normed = vec![0i64; DIM];
    tensor_model::ops::rmsnorm(
        x,
        &norm_vec(DIM, 4),
        tensor_kernel::fixed::RMS_EPS_QWEN3,
        &mut normed,
    );
    let mut scores = vec![0i64; EXPERTS];
    matmul_scalar(
        &l.router_weights,
        &l.router_scales,
        &normed,
        EXPERTS,
        DIM,
        &mut scores,
    );
    let chosen = route(&scores, 1);
    assert_eq!(
        chosen[0].1,
        tensor_kernel::fixed::ONE,
        "a single chosen expert must carry the whole weight"
    );
    chosen[0].0 as usize
}

/// With one expert chosen its weight is exactly one, so routing must reproduce a dense layer
/// built from that expert — the same arithmetic reached two ways.
///
/// Run across several inputs, because a single input that happened to route to expert zero
/// would compare expert zero against expert zero and pass while testing nothing. The test
/// requires that more than one expert is actually reached.
#[test]
fn routing_to_one_expert_equals_a_dense_layer_of_it() {
    let l = build_layer();
    let mut reached = std::collections::BTreeSet::new();

    for seed in 0..8i64 {
        let x: Vec<i64> = (0..DIM)
            .map(|i| (((i as i64 * 53 + seed * 29) % 13) - 6) << 29)
            .collect();
        let which = picks(&l, &x);
        reached.insert(which);

        let mut h = Harness::new(l.blob.clone());
        let mixed = run_layer(&mut h, &l, &x, EXPERTS as u32, 1, 0);

        let mut h2 = Harness::new(l.blob.clone());
        let dense = run_layer(&mut h2, &l, &x, 0, 0, which as u64 * l.stride);

        assert_eq!(
            mixed, dense,
            "seed {seed}: routing to expert {which} did not match a dense layer of it"
        );
        assert!(
            mixed.iter().any(|&v| v != 0),
            "seed {seed}: the layer produced nothing"
        );
    }

    // Comparing expert zero against expert zero would hold trivially. The stride arithmetic
    // is only under test once the router reaches somewhere else.
    assert!(
        reached.len() > 1 && reached.iter().any(|&e| e != 0),
        "the router only ever reached {reached:?}, so the stride was never exercised"
    );
}

/// Visiting more experts must change the result, or the routing weights are being ignored.
#[test]
fn visiting_more_experts_changes_the_output() {
    let l = build_layer();
    let x: Vec<i64> = (0..DIM)
        .map(|i| (((i as i64 * 53) % 13) - 6) << 29)
        .collect();

    let mut h1 = Harness::new(l.blob.clone());
    let one = run_layer(&mut h1, &l, &x, EXPERTS as u32, 1, 0);

    let mut h2 = Harness::new(l.blob.clone());
    let two = run_layer(&mut h2, &l, &x, EXPERTS as u32, 2, 0);

    assert_ne!(one, two, "a second expert made no difference to the output");
}
