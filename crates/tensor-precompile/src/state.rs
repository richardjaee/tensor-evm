//! The matmul that reads its weights from chain state.
//!
//! # Why this exists
//!
//! Phase 1 measured what it costs to pass weights as calldata: 31,549 times more than
//! multiplying by them. Calldata is priced per byte against scarce block space; the matmul
//! is priced against measured throughput. Those schedules are nowhere near each other, and
//! the gap widens with model width. Making the arithmetic sixteen thousand times cheaper
//! accomplishes nothing while moving its operands costs that much more than the operation.
//!
//! So the weights do not move. They live in chain state, and the caller passes a
//! reference: an address and an offset. Calldata then carries only the activations, which
//! is `cols` values rather than `rows * cols` — for a 5120-wide model, twenty kilobytes
//! instead of fifty megabytes.
//!
//! # Where weights live
//!
//! In contract code, the way the companion project stores them. Code is contiguous, read
//! in bulk, and already content-addressed by its hash — which doubles as the provenance
//! check that makes an audit meaningful, since proving *which* weights ran is the point.
//! Storage slots would be the alternative and are hopeless for this: 32 bytes at a time,
//! each through a trie.
//!
//! # Why the read is not priced separately
//!
//! Because the kernel is memory-bound. Its cost is dominated by streaming weights, so
//! pricing the matmul against measured bandwidth already prices reading them — the two are
//! the same work. Charging per byte on top would be double-counting the only thing this
//! operation actually does. What remains is the cost of touching a cold account, which is
//! real and is charged.

use revm::context::{Cfg, ContextTr};
use revm::context_interface::JournalTr;
use revm::handler::{EthPrecompiles, PrecompileProvider};
use revm::interpreter::{CallInputs, Gas, InstructionResult, InterpreterResult};
use revm::precompile::{
    u64_to_address, Precompile, PrecompileId, PrecompileOutput, PrecompileResult, PrecompileSpecId,
    Precompiles,
};
use revm::primitives::{Address, Bytes};
use tensor_kernel::matmul;

use crate::{gas_cost, BASE_GAS, MAX_DIM};

/// First byte of a chunk's deployed code.
const TAG_CHUNK: u8 = 0x00;
/// First byte of a directory's.
const TAG_DIRECTORY: u8 = 0x01;
/// Bytes an address occupies in a directory.
/// Bytes per directory entry: a 20-byte address then a 4-byte cumulative blob offset.
const ENTRY: usize = 24;

/// How a host hands over an account's code, and whether this call was the first to touch it.
///
/// An indirection rather than a concrete type because there is more than one host. revm's
/// journal is one; a client that reaches state through its own abstraction is another, and
/// weight resolution is consensus-critical enough that the two must not each have a copy of
/// it.
///
/// `Bytes` rather than `Vec<u8>` on purpose. Code is refcounted inside the EVM, so handing
/// it over costs a refcount bump; copying it out costs the length of the chunk. Reading
/// weights is the dominant cost of a token — a 4B model moves 8 GB of them per token, over
/// a hundred times what the attention cache moves — so a deep copy per touch was doubling
/// the one thing worth making cheap.
pub type CodeSource<'a> = &'a mut dyn FnMut(Address) -> Result<(Bytes, bool), String>;

fn resolve<CTX: ContextTr>(
    context: &mut CTX,
    at: Address,
    offset: usize,
    len: usize,
) -> Result<Option<(Bytes, u64)>, String> {
    let mut code_of = |a: Address| -> Result<(Bytes, bool), String> {
        let loaded = context
            .journal_mut()
            .code(a)
            .map_err(|_| "failed to load weights".to_string())?;
        Ok((loaded.data.clone(), loaded.is_cold))
    };
    resolve_with(&mut code_of, at, offset, len)
}

/// Weight resolution, against any host that can produce an account's code.
///
/// Handles both forms a blob takes: a single chunk, and a directory whose entries carry
/// cumulative offsets so a reader seeks to the chunk it needs rather than walking to it.
pub fn resolve_with(
    code_of: CodeSource<'_>,
    at: Address,
    offset: usize,
    len: usize,
) -> Result<Option<(Bytes, u64)>, String> {
    let (code, is_cold) = code_of(at)?;
    let mut charged = if is_cold {
        COLD_ACCOUNT_GAS
    } else {
        WARM_ACCOUNT_GAS
    };
    if code.is_empty() {
        return Ok(None);
    }

    if code[0] == TAG_CHUNK {
        let end = match offset.checked_add(len) {
            Some(e) => e + 1,
            None => return Ok(None),
        };
        if end > code.len() {
            return Ok(None);
        }
        // A view into the chunk rather than a copy of it. `Bytes` is refcounted, so this
        // costs nothing and the matmul streams straight out of the code the EVM already
        // holds — which is the whole cost of a token, avoided, whenever a tensor does not
        // straddle a chunk boundary.
        return Ok(Some((code.slice(offset + 1..end), charged)));
    }
    if code[0] != TAG_DIRECTORY {
        return Ok(None);
    }

    // Each entry is an address and the blob's cumulative length through that chunk, so
    // where a byte lives is answerable from the directory alone. Without the offsets a
    // reader had to load every preceding chunk just to learn how long it was, which made
    // reading a tensor near the end of a 31.5MB layer touch all 1,337 of its chunks and
    // dominated the cost of a token.
    let n = (code.len() - 1) / ENTRY;
    let end = match offset.checked_add(len) {
        Some(e) => e,
        None => return Ok(None),
    };
    let cum = |i: usize| -> usize {
        let e = &code[1 + i * ENTRY..1 + (i + 1) * ENTRY];
        u32::from_be_bytes(e[20..24].try_into().expect("four bytes")) as usize
    };
    if n == 0 || end > cum(n - 1) {
        return Ok(None); // the directory does not span the range asked for
    }

    // The first chunk holding `offset`: the lowest `i` whose cumulative end exceeds it.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cum(mid) > offset {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    let mut out = Vec::with_capacity(len);
    let mut want = offset;
    let mut left = len;

    for i in lo..n {
        if left == 0 {
            break;
        }
        let entry = &code[1 + i * ENTRY..1 + (i + 1) * ENTRY];
        let chunk_at = Address::from_slice(&entry[..20]);
        let seen = if i == 0 { 0 } else { cum(i - 1) };

        let (data, chunk_cold) = code_of(chunk_at)?;
        charged += if chunk_cold {
            COLD_ACCOUNT_GAS
        } else {
            WARM_ACCOUNT_GAS
        };
        if data.is_empty() {
            return Ok(None);
        }
        let size = data.len() - 1; // the tag is not payload
                                   // The directory is written from the chunks' own sizes, so a mismatch means the two
                                   // have diverged and no answer derived from them can be trusted.
        if cum(i) != seen + size {
            return Ok(None);
        }

        let start = want - seen;
        let take = left.min(size - start);
        out.extend_from_slice(&data[1 + start..1 + start + take]);
        want += take;
        left -= take;
    }

    if left != 0 {
        return Ok(None); // the directory does not span the range asked for
    }
    // Spanning chunks leaves no choice but to stitch, since the pieces are not adjacent in
    // memory. Sizing chunks so a tensor fits inside one is what avoids this path.
    Ok(Some((out.into(), charged)))
}

/// Address of the state-reading matmul.
///
/// The calldata form at `0x100` is kept alongside it: it is the right shape for small
/// operands, and its cost is the measurement that justifies this one existing.
pub const MATMUL_STATE_ADDRESS: Address = u64_to_address(0x101);

/// First touch of the weights account, mirroring the EVM's own cold access cost. The bytes
/// themselves are paid for through the matmul price.
pub const COLD_ACCOUNT_GAS: u64 = 2600;

/// Every touch after the first. A forward pass reads the same weights account on every
/// call of every layer, so this is the price that governs — charging the cold figure
/// throughout would overcharge by 26x on all but one call.
pub const WARM_ACCOUNT_GAS: u64 = 100;

/// ```text
/// [0..4)     rows      u32
/// [4..8)     cols      u32
/// [8..28)    weights   address of the contract whose code holds them
/// [28..32)   offset    u32, byte offset within that code
/// [32..)     acts      i64 x cols
/// ```
///
/// The code at `weights[offset..]` is `rows` scale bytes followed by `rows * cols`
/// big-endian `i16` values, row-major — the same layout the calldata form uses, so a
/// weight blob can be fed to either.
const HEADER: usize = 32;

/// A caller too poor to pay even the base cost has run out of gas rather than made a
/// mistake, so the two cases are distinguished rather than collapsed.
fn revert(gas_limit: u64) -> InterpreterResult {
    let mut gas = Gas::new(gas_limit);
    if !gas.record_regular_cost(BASE_GAS) {
        return oog(gas_limit);
    }
    InterpreterResult::new(InstructionResult::Revert, Bytes::new(), gas)
}

fn oog(gas_limit: u64) -> InterpreterResult {
    let mut gas = Gas::new(gas_limit);
    gas.spend_all();
    InterpreterResult::new(InstructionResult::PrecompileOOG, Bytes::new(), gas)
}

/// A precompile set carrying both matmul forms alongside Ethereum's own.
///
/// The state-reading form cannot be an ordinary [`revm::precompile::Precompile`]: that
/// signature receives only its input bytes and never sees the chain. Reaching state means
/// implementing the provider itself, which is handed the context.
#[derive(Clone)]
pub struct TensorPrecompiles {
    inner: EthPrecompiles,
}

impl TensorPrecompiles {
    pub fn new(spec: revm::primitives::hardfork::SpecId) -> Self {
        Self {
            inner: EthPrecompiles {
                precompiles: eth_plus_matmuls(spec),
                spec,
            },
        }
    }
}

impl Default for TensorPrecompiles {
    fn default() -> Self {
        Self::new(revm::primitives::hardfork::SpecId::default())
    }
}

/// Placeholder occupying [`MATMUL_STATE_ADDRESS`] in the precompile set.
///
/// It is never executed — [`TensorPrecompiles::run`] intercepts the address first — but it
/// has to be registered, because membership in the set is what makes the interpreter treat
/// the address as a precompile at all. Without it a `STATICCALL` from a contract finds an
/// ordinary account with no code and returns success having done nothing, silently, which
/// is exactly the failure this replaced: an inner loop that appeared free because none of
/// its calls were reaching anything.
fn unreachable_stub(_input: &[u8], _gas_limit: u64, reservoir: u64) -> PrecompileResult {
    Ok(PrecompileOutput::revert(BASE_GAS, Bytes::new(), reservoir))
}

/// Ethereum's precompiles, plus both matmul forms, for one spec.
///
/// Leaked into a `'static` deliberately: a chain's precompile set is fixed for the life of
/// the process, so one initialisation per spec is its natural lifetime.
fn eth_plus_matmuls(spec: revm::primitives::hardfork::SpecId) -> &'static Precompiles {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static SETS: OnceLock<Mutex<HashMap<u8, &'static Precompiles>>> = OnceLock::new();
    let map = SETS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = map.lock().expect("precompile set registry poisoned");
    map.entry(spec as u8).or_insert_with(|| {
        let mut p = Precompiles::new(PrecompileSpecId::from_spec_id(spec)).clone();
        p.extend([
            crate::precompile(),
            crate::activation::precompile(),
            crate::norm::precompile(),
            crate::attention::attention(),
            crate::attention::rmsnorm(),
            Precompile::new(
                PrecompileId::Custom("tensor-classify".into()),
                crate::classify::CLASSIFY_ADDRESS,
                unreachable_stub,
            ),
            Precompile::new(
                PrecompileId::Custom("tensor-router".into()),
                crate::router::ROUTER_ADDRESS,
                unreachable_stub,
            ),
            Precompile::new(
                PrecompileId::Custom("tensor-ffn".into()),
                FFN_ADDRESS,
                unreachable_stub,
            ),
            Precompile::new(
                PrecompileId::Custom("tensor-attn-proj".into()),
                ATTN_PROJ_ADDRESS,
                unreachable_stub,
            ),
            Precompile::new(
                PrecompileId::Custom("tensor-layer".into()),
                LAYER_ADDRESS,
                unreachable_stub,
            ),
            Precompile::new(
                PrecompileId::Custom("tensor-matmul-state".into()),
                MATMUL_STATE_ADDRESS,
                unreachable_stub,
            ),
        ]);
        Box::leak(Box::new(p))
    })
}

impl<CTX: ContextTr> PrecompileProvider<CTX> for TensorPrecompiles {
    type Output = InterpreterResult;

    fn set_spec(&mut self, spec: <CTX::Cfg as Cfg>::Spec) -> bool {
        let spec = spec.into();
        if spec == self.inner.spec {
            return false;
        }
        self.inner = EthPrecompiles {
            precompiles: eth_plus_matmuls(spec),
            spec,
        };
        true
    }

    fn run(
        &mut self,
        context: &mut CTX,
        inputs: &CallInputs,
    ) -> Result<Option<InterpreterResult>, String> {
        if inputs.bytecode_address == crate::classify::CLASSIFY_ADDRESS {
            return run_classify(context, inputs);
        }
        if inputs.bytecode_address == crate::router::ROUTER_ADDRESS {
            return run_router(context, inputs);
        }
        if inputs.bytecode_address == FFN_ADDRESS {
            return run_host(context, inputs, ffn_state_with);
        }
        if inputs.bytecode_address == ATTN_PROJ_ADDRESS {
            return run_host(context, inputs, attn_proj_state_with);
        }
        if inputs.bytecode_address == LAYER_ADDRESS {
            return run_host(context, inputs, layer_state_with);
        }
        if inputs.bytecode_address != MATMUL_STATE_ADDRESS {
            return PrecompileProvider::<CTX>::run(&mut self.inner, context, inputs);
        }

        let gas_limit = inputs.gas_limit;
        let input = inputs.input.as_bytes(context).to_vec();

        let mut code_of = |a: Address| -> Result<(Bytes, bool), String> {
            let loaded = context
                .journal_mut()
                .code(a)
                .map_err(|_| "failed to load weights".to_string())?;
            Ok((loaded.data.clone(), loaded.is_cold))
        };
        match matmul_state_with(&mut code_of, &input, gas_limit)? {
            Outcome::Revert => Ok(Some(revert(gas_limit))),
            Outcome::OutOfGas => Ok(Some(oog(gas_limit))),
            Outcome::Return(bytes, used) => {
                let mut gas = Gas::new(gas_limit);
                if !gas.record_regular_cost(used) {
                    return Ok(Some(oog(gas_limit)));
                }
                Ok(Some(InterpreterResult::new(
                    InstructionResult::Return,
                    bytes.into(),
                    gas,
                )))
            }
        }
    }

    fn warm_addresses(&self) -> &revm::primitives::AddressSet {
        PrecompileProvider::<CTX>::warm_addresses(&self.inner)
    }

    fn contains(&self, address: &Address) -> bool {
        *address == MATMUL_STATE_ADDRESS
            || *address == crate::classify::CLASSIFY_ADDRESS
            || *address == crate::router::ROUTER_ADDRESS
            || *address == FFN_ADDRESS
            || *address == ATTN_PROJ_ADDRESS
            || *address == LAYER_ADDRESS
            || PrecompileProvider::<CTX>::contains(&self.inner, address)
    }
}

/// Encodes a call to the state-reading form.
pub fn encode_state(
    rows: usize,
    cols: usize,
    weights_at: Address,
    offset: u32,
    acts: &[i64],
) -> Vec<u8> {
    assert_eq!(acts.len(), cols, "activations must be cols");
    let mut v = Vec::with_capacity(HEADER + cols * 8);
    v.extend_from_slice(&(rows as u32).to_be_bytes());
    v.extend_from_slice(&(cols as u32).to_be_bytes());
    v.extend_from_slice(weights_at.as_slice());
    v.extend_from_slice(&offset.to_be_bytes());
    for a in acts {
        v.extend_from_slice(&a.to_be_bytes());
    }
    v
}

/// Lays out a weight blob for storage as contract code: scales, then row-major `i16`.
pub fn weight_blob(rows: usize, cols: usize, scales: &[i8], weights: &[i16]) -> Vec<u8> {
    assert_eq!(scales.len(), rows, "one scale per row");
    assert_eq!(weights.len(), rows * cols, "weights must be rows*cols");
    let mut v = Vec::with_capacity(rows + rows * cols * 2);
    v.extend(scales.iter().map(|&s| s as u8));
    for w in weights {
        v.extend_from_slice(&w.to_be_bytes());
    }
    v
}

/// The classifier reads its weights from state like the matmul does, and returns
/// only the winning row — a 151,936-entry logit vector is 1.2MB that the caller
/// would discard all but one of.
fn run_classify<CTX: ContextTr>(
    context: &mut CTX,
    inputs: &CallInputs,
) -> Result<Option<InterpreterResult>, String> {
    let gas_limit = inputs.gas_limit;
    let input = inputs.input.as_bytes(context).to_vec();
    if input.len() < 32 {
        return Ok(Some(revert(gas_limit)));
    }
    let rows = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
    let cols = u32::from_be_bytes(input[4..8].try_into().unwrap()) as u64;
    let at = Address::from_slice(&input[8..28]);
    let offset = u32::from_be_bytes(input[28..32].try_into().unwrap()) as usize;

    if rows == 0 || cols == 0 || cols > MAX_DIM || rows > (1 << 24) {
        return Ok(Some(revert(gas_limit)));
    }
    let Some(compute) = crate::classify::gas_cost(rows, cols) else {
        return Ok(Some(revert(gas_limit)));
    };
    if compute + WARM_ACCOUNT_GAS > gas_limit {
        return Ok(Some(oog(gas_limit)));
    }

    let cols_u = cols as usize;
    if input.len() != 32 + cols_u * 8 {
        return Ok(Some(revert(gas_limit)));
    }
    let x: Vec<i64> = input[32..]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();

    // Through the same resolver as the matmul, so a classifier uploaded in chunks behaves
    // exactly like one that fitted in a single deployment. At 151,936 rows it never fits,
    // so chunked is the normal case rather than the exception.
    let Some(span) = (rows as usize)
        .checked_mul(cols_u)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(rows as usize))
    else {
        return Ok(Some(revert(gas_limit)));
    };
    let Some((bytes, access)) = resolve(context, at, offset, span)? else {
        return Ok(Some(revert(gas_limit)));
    };

    let Some((row, logit)) = crate::classify::classify(&bytes, 0, rows as usize, cols_u, &x) else {
        return Ok(Some(revert(gas_limit)));
    };

    let mut gas = Gas::new(gas_limit);
    if !gas.record_regular_cost(compute + access) {
        return Ok(Some(oog(gas_limit)));
    }
    Ok(Some(InterpreterResult::new(
        InstructionResult::Return,
        crate::classify::pack(row, logit),
        gas,
    )))
}

/// Scores every expert against the residual and returns the chosen few.
///
/// The router tensor is small — one row an expert — so nothing here is expensive. What it
/// buys is expensive: the caller now knows which experts to read and, more to the point,
/// which not to. A layer of a 256-expert model is megabytes, and choosing eight of them
/// means never touching the other 97%.
fn run_router<CTX: ContextTr>(
    context: &mut CTX,
    inputs: &CallInputs,
) -> Result<Option<InterpreterResult>, String> {
    let gas_limit = inputs.gas_limit;
    let input = inputs.input.as_bytes(context).to_vec();

    // Parsed by the same function the standalone entry point uses, so the two cannot drift
    // about what a well-formed call is.
    let Some((experts, cols, at, offset, k)) = crate::router::parse(&input) else {
        return Ok(Some(revert(gas_limit)));
    };
    let Some(compute) = crate::router::gas_cost(experts, cols) else {
        return Ok(Some(revert(gas_limit)));
    };
    if compute + WARM_ACCOUNT_GAS > gas_limit {
        return Ok(Some(oog(gas_limit)));
    }

    let cols_u = cols as usize;
    let experts_u = experts as usize;
    let x: Vec<i64> = input[crate::router::HEADER..]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();

    let Some(span) = experts_u
        .checked_mul(cols_u)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(experts_u))
    else {
        return Ok(Some(revert(gas_limit)));
    };
    let Some((bytes, access)) = resolve(context, at, offset, span)? else {
        return Ok(Some(revert(gas_limit)));
    };

    let scales: Vec<i8> = bytes[..experts_u].iter().map(|&b| b as i8).collect();
    let weights: Vec<i16> = bytes[experts_u..]
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    let mut scores = vec![0i64; experts_u];
    matmul(&weights, &scales, &x, experts_u, cols_u, &mut scores);

    let chosen = crate::router::route(&scores, k);

    let mut gas = Gas::new(gas_limit);
    if !gas.record_regular_cost(compute + access) {
        return Ok(Some(oog(gas_limit)));
    }
    Ok(Some(InterpreterResult::new(
        InstructionResult::Return,
        crate::router::encode_output(&chosen).into(),
        gas,
    )))
}

/// What a precompile decided, before a host turns it into its own result type.
pub enum Outcome {
    Return(Vec<u8>, u64),
    Revert,
    OutOfGas,
}

/// The state-reading matmul, against any host that can produce an account's code.
///
/// Lifted out of the provider so reth can run the same function. A second implementation of
/// this would be a second opinion about what a model computes, which is the one thing a
/// chain cannot have.
pub fn matmul_state_with(
    code_of: CodeSource<'_>,
    input: &[u8],
    gas_limit: u64,
) -> Result<Outcome, String> {
    if input.len() < HEADER {
        return Ok(Outcome::Revert);
    }

    let rows = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
    let cols = u32::from_be_bytes(input[4..8].try_into().unwrap()) as u64;
    let weights_at = Address::from_slice(&input[8..28]);
    let offset = u32::from_be_bytes(input[28..32].try_into().unwrap()) as usize;

    if rows == 0 || cols == 0 || rows > MAX_DIM || cols > MAX_DIM {
        return Ok(Outcome::Revert);
    }

    let Some(compute) = gas_cost(rows, cols) else {
        return Ok(Outcome::Revert);
    };
    // Priced against the cheaper of the two access costs. The real figure depends on
    // whether the weights account is already warm, which is not known until it is
    // loaded; this is only the affordability precheck, so it must not reject a caller
    // who can in fact pay.
    let Some(floor) = compute.checked_add(WARM_ACCOUNT_GAS) else {
        return Ok(Outcome::Revert);
    };
    if floor > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let (rows_u, cols_u) = (rows as usize, cols as usize);

    // Activations arrive as calldata; only they scale with cols rather than rows*cols,
    // which is the entire reason this form is affordable.
    let Some(acts_len) = cols_u.checked_mul(8) else {
        return Ok(Outcome::Revert);
    };
    if input.len() != HEADER + acts_len {
        return Ok(Outcome::Revert);
    }
    let acts: Vec<i64> = input[HEADER..]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();

    let Some(weights_len) = rows_u.checked_mul(cols_u).and_then(|n| n.checked_mul(2)) else {
        return Ok(Outcome::Revert);
    };
    let Some(span) = rows_u.checked_add(weights_len) else {
        return Ok(Outcome::Revert);
    };

    // Warmth is what the access actually costs, and is only knowable from the load. A
    // forward pass hits the same weights on every call of every layer, so charging the
    // cold price unconditionally would overcharge by 26x on all but the first.
    let Some((bytes, access)) = resolve_with(code_of, weights_at, offset, span)? else {
        // Either the blob cannot hold the declared shape or the directory does not span
        // it. Either way it is the caller's error, not a reason to read past the end.
        return Ok(Outcome::Revert);
    };

    let Some(total) = compute.checked_add(access) else {
        return Ok(Outcome::Revert);
    };
    if total > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let scales: Vec<i8> = bytes[..rows_u].iter().map(|&b| b as i8).collect();
    let weights: Vec<i16> = bytes[rows_u..]
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();

    let mut out = vec![0i64; rows_u];
    matmul(&weights, &scales, &acts, rows_u, cols_u, &mut out);

    let mut packed = Vec::with_capacity(rows_u * 8);
    for v in out {
        packed.extend_from_slice(&v.to_be_bytes());
    }
    Ok(Outcome::Return(packed, total))
}

/// The classifier, against any host that can produce an account's code.
pub fn classify_state_with(
    code_of: CodeSource<'_>,
    input: &[u8],
    gas_limit: u64,
) -> Result<Outcome, String> {
    if input.len() < 32 {
        return Ok(Outcome::Revert);
    }
    let rows = u32::from_be_bytes(input[0..4].try_into().expect("four")) as u64;
    let cols = u32::from_be_bytes(input[4..8].try_into().expect("four")) as u64;
    let at = Address::from_slice(&input[8..28]);
    let offset = u32::from_be_bytes(input[28..32].try_into().expect("four")) as usize;

    if rows == 0 || cols == 0 || cols > MAX_DIM || rows > (1 << 24) {
        return Ok(Outcome::Revert);
    }
    let Some(compute) = crate::classify::gas_cost(rows, cols) else {
        return Ok(Outcome::Revert);
    };
    if compute + WARM_ACCOUNT_GAS > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let cols_u = cols as usize;
    if input.len() != 32 + cols_u * 8 {
        return Ok(Outcome::Revert);
    }
    let x: Vec<i64> = input[32..]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();

    let Some(span) = (rows as usize)
        .checked_mul(cols_u)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(rows as usize))
    else {
        return Ok(Outcome::Revert);
    };
    let Some((bytes, access)) = resolve_with(code_of, at, offset, span)? else {
        return Ok(Outcome::Revert);
    };
    let Some((row, logit)) = crate::classify::classify(&bytes, 0, rows as usize, cols_u, &x) else {
        return Ok(Outcome::Revert);
    };
    let Some(total) = compute.checked_add(access) else {
        return Ok(Outcome::Revert);
    };
    if total > gas_limit {
        return Ok(Outcome::OutOfGas);
    }
    Ok(Outcome::Return(
        crate::classify::pack(row, logit).to_vec(),
        total,
    ))
}

/// The router, against any host that can produce an account's code.
pub fn router_state_with(
    code_of: CodeSource<'_>,
    input: &[u8],
    gas_limit: u64,
) -> Result<Outcome, String> {
    let Some((experts, cols, at, offset, k)) = crate::router::parse(input) else {
        return Ok(Outcome::Revert);
    };
    let Some(compute) = crate::router::gas_cost(experts, cols) else {
        return Ok(Outcome::Revert);
    };
    if compute + WARM_ACCOUNT_GAS > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let (cols_u, experts_u) = (cols as usize, experts as usize);
    let x: Vec<i64> = input[crate::router::HEADER..]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();

    let Some(span) = experts_u
        .checked_mul(cols_u)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(experts_u))
    else {
        return Ok(Outcome::Revert);
    };
    let Some((bytes, access)) = resolve_with(code_of, at, offset, span)? else {
        return Ok(Outcome::Revert);
    };

    let scales: Vec<i8> = bytes[..experts_u].iter().map(|&b| b as i8).collect();
    let weights: Vec<i16> = bytes[experts_u..]
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();
    let mut scores = vec![0i64; experts_u];
    matmul(&weights, &scales, &x, experts_u, cols_u, &mut scores);

    let Some(total) = compute.checked_add(access) else {
        return Ok(Outcome::Revert);
    };
    if total > gas_limit {
        return Ok(Outcome::OutOfGas);
    }
    Ok(Outcome::Return(
        crate::router::encode_output(&crate::router::route(&scores, k)),
        total,
    ))
}

/// Address of the fused feed-forward.
pub const FFN_ADDRESS: Address = revm::precompile::u64_to_address(0x108);

/// ```text
/// [0..4)    hidden  u32
/// [4..8)    dim     u32
/// [8..28)   weights address holding all three tensors
/// [28..32)  gate    u32, byte offset within it
/// [32..36)  up      u32
/// [36..40)  down    u32
/// [40 ..)   x       i64 x dim
/// ```
pub const FFN_HEADER: usize = 40;

/// The whole feed-forward — gate, up, SwiGLU, down — in one call.
///
/// Three separate calls cost far more than the arithmetic in them. A layer's profile put the
/// precompiles at 2.8% of it and Solidity handing them their arguments at the rest, and the
/// feed-forward is where that hurts most: the intermediates are `hidden` wide, which is three
/// times the residual stream, and each one is packed into `bytes`, copied across a call
/// boundary, copied back, and paid for again in the EVM's quadratic memory price.
///
/// Fusing removes them. The caller passes `dim` activations and gets `dim` back; everything
/// of width `hidden` now exists only inside the VM, where it is a `Vec<i64>` and costs
/// nothing. Same arithmetic, in the same order, against the same weights.
pub fn ffn_state_with(
    code_of: CodeSource<'_>,
    input: &[u8],
    gas_limit: u64,
) -> Result<Outcome, String> {
    if input.len() < FFN_HEADER {
        return Ok(Outcome::Revert);
    }
    let hidden = u32::from_be_bytes(input[0..4].try_into().expect("four")) as u64;
    let dim = u32::from_be_bytes(input[4..8].try_into().expect("four")) as u64;
    let at = Address::from_slice(&input[8..28]);
    let g_off = u32::from_be_bytes(input[28..32].try_into().expect("four")) as usize;
    let u_off = u32::from_be_bytes(input[32..36].try_into().expect("four")) as usize;
    let d_off = u32::from_be_bytes(input[36..40].try_into().expect("four")) as usize;

    if hidden == 0 || dim == 0 || hidden > MAX_DIM || dim > MAX_DIM {
        return Ok(Outcome::Revert);
    }
    if input.len() != FFN_HEADER + (dim as usize) * 8 {
        return Ok(Outcome::Revert);
    }

    // Three matmuls and one elementwise pass, priced as the separate calls would have been.
    let Some(compute) = gas_cost(hidden, dim)
        .and_then(|a| a.checked_mul(2))
        .and_then(|a| a.checked_add(gas_cost(dim, hidden)?))
        .and_then(|a| a.checked_add(crate::activation::gas_cost(hidden)?))
    else {
        return Ok(Outcome::Revert);
    };
    if compute + WARM_ACCOUNT_GAS > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let (hidden_u, dim_u) = (hidden as usize, dim as usize);
    let x: Vec<i64> = input[FFN_HEADER..]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();

    let wide = hidden_u + hidden_u * dim_u * 2;
    let tall = dim_u + dim_u * hidden_u * 2;
    let mut access = 0u64;
    let mut read =
        |code_of: CodeSource<'_>, off: usize, span: usize| -> Result<Option<Bytes>, String> {
            match resolve_with(code_of, at, off, span)? {
                Some((bytes, charged)) => {
                    access = access.saturating_add(charged);
                    Ok(Some(bytes))
                }
                None => Ok(None),
            }
        };

    let (Some(g), Some(u), Some(d)) = (
        read(code_of, g_off, wide)?,
        read(code_of, u_off, wide)?,
        read(code_of, d_off, tall)?,
    ) else {
        return Ok(Outcome::Revert);
    };

    let split = |b: &[u8], rows: usize| -> (Vec<i8>, Vec<i16>) {
        (
            b[..rows].iter().map(|&v| v as i8).collect(),
            b[rows..]
                .chunks_exact(2)
                .map(|c| i16::from_be_bytes([c[0], c[1]]))
                .collect(),
        )
    };
    let (gs, gw) = split(&g, hidden_u);
    let (us, uw) = split(&u, hidden_u);
    let (ds, dw) = split(&d, dim_u);

    let mut gate = vec![0i64; hidden_u];
    let mut up = vec![0i64; hidden_u];
    matmul(&gw, &gs, &x, hidden_u, dim_u, &mut gate);
    matmul(&uw, &us, &x, hidden_u, dim_u, &mut up);
    tensor_kernel::swiglu_vec(&mut gate, &up);

    let mut out = vec![0i64; dim_u];
    matmul(&dw, &ds, &gate, dim_u, hidden_u, &mut out);

    let Some(total) = compute.checked_add(access) else {
        return Ok(Outcome::Revert);
    };
    if total > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let mut packed = Vec::with_capacity(dim_u * 8);
    for v in out {
        packed.extend_from_slice(&v.to_be_bytes());
    }
    Ok(Outcome::Return(packed, total))
}

/// Runs a host-agnostic precompile against revm's journal.
fn run_host<CTX: ContextTr>(
    context: &mut CTX,
    inputs: &CallInputs,
    run: fn(CodeSource<'_>, &[u8], u64) -> Result<Outcome, String>,
) -> Result<Option<InterpreterResult>, String> {
    let gas_limit = inputs.gas_limit;
    let input = inputs.input.as_bytes(context).to_vec();
    let mut code_of = |a: Address| -> Result<(Bytes, bool), String> {
        let loaded = context
            .journal_mut()
            .code(a)
            .map_err(|_| "failed to load weights".to_string())?;
        Ok((loaded.data.clone(), loaded.is_cold))
    };
    match run(&mut code_of, &input, gas_limit)? {
        Outcome::Revert => Ok(Some(revert(gas_limit))),
        Outcome::OutOfGas => Ok(Some(oog(gas_limit))),
        Outcome::Return(bytes, used) => {
            let mut gas = Gas::new(gas_limit);
            if !gas.record_regular_cost(used) {
                return Ok(Some(oog(gas_limit)));
            }
            Ok(Some(InterpreterResult::new(
                InstructionResult::Return,
                bytes.into(),
                gas,
            )))
        }
    }
}

/// Address of attention fused with its output projection.
pub const ATTN_PROJ_ADDRESS: Address = revm::precompile::u64_to_address(0x109);

/// ```text
/// [0..4)    heads     u32
/// [4..8)    n_kv      u32
/// [8..12)   head_dim  u32
/// [12..16)  positions u32
/// [16..20)  dim       u32, the residual width the projection returns
/// [20..40)  weights   address holding the output projection
/// [40..44)  offset    u32, where it sits in that blob
/// [44 ..)   q, then k, then v
/// ```
pub const ATTN_PROJ_HEADER: usize = 44;

/// Attention and the output projection that always follows it, in one call.
///
/// `o_proj` was the single most expensive line in a layer's profile — 184,368 gas of 673,643
/// — and almost none of it was arithmetic. Attention returns `heads * head_dim` values,
/// which in Qwen3-0.6B is twice the residual stream, and every one of them was packed into
/// `bytes`, handed back to Solidity, and handed straight into another precompile.
///
/// Nothing else ever consumes an attention result, so it need not leave the VM at all.
pub fn attn_proj_state_with(
    code_of: CodeSource<'_>,
    input: &[u8],
    gas_limit: u64,
) -> Result<Outcome, String> {
    if input.len() < ATTN_PROJ_HEADER {
        return Ok(Outcome::Revert);
    }
    let heads = u32::from_be_bytes(input[0..4].try_into().expect("four")) as u64;
    let n_kv = u32::from_be_bytes(input[4..8].try_into().expect("four")) as u64;
    let hs = u32::from_be_bytes(input[8..12].try_into().expect("four")) as u64;
    let positions = u32::from_be_bytes(input[12..16].try_into().expect("four")) as u64;
    let dim = u32::from_be_bytes(input[16..20].try_into().expect("four")) as u64;
    let at = Address::from_slice(&input[20..40]);
    let offset = u32::from_be_bytes(input[40..44].try_into().expect("four")) as usize;

    if heads == 0 || n_kv == 0 || hs == 0 || positions == 0 || dim == 0 {
        return Ok(Outcome::Revert);
    }
    if heads > MAX_DIM || n_kv > MAX_DIM || hs > MAX_DIM || positions > MAX_DIM || dim > MAX_DIM {
        return Ok(Outcome::Revert);
    }
    // Grouped-query attention shares each key head across an exact number of query heads.
    if !heads.is_multiple_of(n_kv) {
        return Ok(Outcome::Revert);
    }

    let (heads_u, n_kv_u, hs_u, pos_u, dim_u) = (
        heads as usize,
        n_kv as usize,
        hs as usize,
        positions as usize,
        dim as usize,
    );
    let q_len = heads_u * hs_u;
    let cache = pos_u * n_kv_u * hs_u;
    // The query is Q32 in eight bytes; the cache is Q16 in four, the width it is deposited
    // and read back at.
    if input.len() != ATTN_PROJ_HEADER + q_len * 8 + cache * 2 * 4 {
        return Ok(Outcome::Revert);
    }

    let Some(compute) = crate::attention::attention_gas(heads, hs, positions)
        .and_then(|a| a.checked_add(gas_cost(dim, q_len as u64)?))
    else {
        return Ok(Outcome::Revert);
    };
    if compute + WARM_ACCOUNT_GAS > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let read = |at_byte: usize, n: usize| -> Vec<i64> {
        input[at_byte..at_byte + n * 8]
            .chunks_exact(8)
            .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
            .collect()
    };
    // Widened back to Q32 on the way in, exactly as the reference does when it reads its
    // own cache, so attending to a narrowed cache and attending to the values it was made
    // from are the same computation.
    let read_kv = |at_byte: usize, n: usize| -> Vec<i64> {
        input[at_byte..at_byte + n * 4]
            .chunks_exact(4)
            .map(|c| {
                tensor_model::qwen3::widen_kv(i32::from_be_bytes(
                    c.try_into().expect("chunks_exact(4)"),
                ))
            })
            .collect()
    };
    let q = read(ATTN_PROJ_HEADER, q_len);
    let k = read_kv(ATTN_PROJ_HEADER + q_len * 8, cache);
    let v = read_kv(ATTN_PROJ_HEADER + q_len * 8 + cache * 4, cache);

    let attn = crate::attention::attend(q, k, v, heads_u, n_kv_u, hs_u, pos_u);

    let span = dim_u + dim_u * q_len * 2;
    let Some((bytes, access)) = resolve_with(code_of, at, offset, span)? else {
        return Ok(Outcome::Revert);
    };
    let scales: Vec<i8> = bytes[..dim_u].iter().map(|&b| b as i8).collect();
    let weights: Vec<i16> = bytes[dim_u..]
        .chunks_exact(2)
        .map(|c| i16::from_be_bytes([c[0], c[1]]))
        .collect();

    let mut out = vec![0i64; dim_u];
    matmul(&weights, &scales, &attn, dim_u, q_len, &mut out);

    let Some(total) = compute.checked_add(access) else {
        return Ok(Outcome::Revert);
    };
    if total > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let mut packed = Vec::with_capacity(dim_u * 8);
    for value in out {
        packed.extend_from_slice(&value.to_be_bytes());
    }
    Ok(Outcome::Return(packed, total))
}

/// Address of the whole-layer form.
pub const LAYER_ADDRESS: Address = revm::precompile::u64_to_address(0x10a);

/// ```text
/// [0..24)   dim, hidden, nHeads, nKv, headDim, positions — six u32
/// [24..44)  weights address
/// [44..88)  eleven u32 offsets: q k v o gate up down, then the four norms
/// [88..96)  rmsEps i64
/// [96 ..)   x (dim), cos (headDim/2), sin (headDim/2), k cache, v cache
/// ```
///
/// The caches hold `positions - 1` entries; this call appends the one it computes.
///
/// # Wiring this from Solidity
///
/// Two things are not obvious and are worth knowing before starting.
///
/// The header is ninety-six bytes, which is three words, and `Tensor._build` carries two.
/// Rather than widening it again and packing eleven offsets and an address across word
/// boundaries by hand, build the header with `abi.encodePacked` and pass it as the first
/// *part* with a head length of zero. The layout below is exactly what `encodePacked`
/// produces from six `uint32`, an `address`, eleven `uint32` and an `int64`, and there is no
/// bit shifting to get wrong.
///
/// The cache is the awkward one. A contract holds `kv.k` as a buffer covering positions
/// zero through `pos`, because `_extendCache` has already written this position's entry into
/// it before attention reads it. This call wants the entries *before* this position and
/// computes the current one itself, so a caller must either hand over a slice — which in
/// Solidity means a copy, and copying the cache is a good part of what fusing was meant to
/// avoid — or the order of operations has to change so the cache is extended after the layer
/// runs rather than inside it. The second is the right answer and it is a change to
/// `forwardLayer`, not to this.
pub const LAYER_HEADER: usize = 96;

/// A whole transformer block, in one call.
///
/// The two fused calls it replaces were 79% of a layer between them, and each was dominated
/// by moving a residual stream in and a residual stream out. Everything between them — the
/// query, the keys, the values, the attention result, the gate, the activation — existed
/// only to be handed back to Solidity and handed straight into the next precompile.
///
/// This is `tensor_model::qwen3::forward_layer` called directly, not a reimplementation of
/// it. That matters more than the gas: the reference is what every accuracy claim in this
/// project is measured against, and a fused layer that computed something subtly different
/// would invalidate all of them at once. Here it cannot differ, because it is the same code.
pub fn layer_state_with(
    code_of: CodeSource<'_>,
    input: &[u8],
    gas_limit: u64,
) -> Result<Outcome, String> {
    use tensor_model::qwen3::{Config, KvCache, LayerWeights, QTensor, Rope, Scratch};

    if input.len() < LAYER_HEADER {
        return Ok(Outcome::Revert);
    }
    let u32_at = |i: usize| u32::from_be_bytes(input[i..i + 4].try_into().expect("four")) as usize;
    let (dim, hidden, heads, n_kv, hs, positions) = (
        u32_at(0),
        u32_at(4),
        u32_at(8),
        u32_at(12),
        u32_at(16),
        u32_at(20),
    );
    let at = Address::from_slice(&input[24..44]);
    let off: Vec<usize> = (0..11).map(|i| u32_at(44 + i * 4)).collect();
    let eps = i64::from_be_bytes(input[88..96].try_into().expect("eight"));

    if dim == 0 || hidden == 0 || heads == 0 || n_kv == 0 || hs == 0 || positions == 0 {
        return Ok(Outcome::Revert);
    }
    if dim > MAX_DIM as usize || hidden > MAX_DIM as usize || positions > MAX_DIM as usize {
        return Ok(Outcome::Revert);
    }
    if !heads.is_multiple_of(n_kv) {
        return Ok(Outcome::Revert);
    }

    let cfg = Config {
        dim,
        hidden,
        n_heads: heads,
        n_kv,
        head_dim: hs,
        seq_len: positions,
        rms_eps: eps,
    };
    let (qw, kvw, half) = (cfg.q_width(), cfg.kv_width(), hs / 2);
    let past = positions - 1;
    // The caches arrive with room for the entry this call computes — `positions` slots, of
    // which the last is ignored on the way in and returned filled. That is the shape a
    // contract already holds, so taking it avoids a copy of the whole cache, which is a
    // good part of what fusing the layer was for.
    // The residual and the rotary row are Q32 in eight bytes; the cache is Q16 in four.
    // Halving the element halves what a position deposits and what every later position
    // reads back, and depositing this cache is two thirds of a step at position zero.
    let want = LAYER_HEADER + (dim + half * 2) * 8 + positions * kvw * 2 * 4;
    if input.len() != want {
        return Ok(Outcome::Revert);
    }

    // Priced as the calls it replaces would have been, so fusing changes what a layer costs
    // to run and not what it costs to buy.
    let Some(compute) = gas_cost(qw as u64, dim as u64)
        .and_then(|a| a.checked_add(gas_cost(kvw as u64, dim as u64)? * 2))
        .and_then(|a| a.checked_add(gas_cost(dim as u64, qw as u64)?))
        .and_then(|a| a.checked_add(gas_cost(hidden as u64, dim as u64)? * 2))
        .and_then(|a| a.checked_add(gas_cost(dim as u64, hidden as u64)?))
        .and_then(|a| {
            a.checked_add(crate::attention::attention_gas(
                heads as u64,
                hs as u64,
                positions as u64,
            )?)
        })
        .and_then(|a| a.checked_add(crate::activation::gas_cost(hidden as u64)?))
    else {
        return Ok(Outcome::Revert);
    };
    if compute + WARM_ACCOUNT_GAS > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    let read = |from: usize, n: usize| -> Vec<i64> {
        input[from..from + n * 8]
            .chunks_exact(8)
            .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
            .collect()
    };
    let mut x = read(LAYER_HEADER, dim);
    let cos = read(LAYER_HEADER + dim * 8, half);
    let sin = read(LAYER_HEADER + (dim + half) * 8, half);
    let read_kv = |from: usize, n: usize| -> Vec<i32> {
        input[from..from + n * 4]
            .chunks_exact(4)
            .map(|c| i32::from_be_bytes(c.try_into().expect("chunks_exact(4)")))
            .collect()
    };
    let kv_base = LAYER_HEADER + (dim + half * 2) * 8;
    let past_k = read_kv(kv_base, past * kvw);
    let past_v = read_kv(kv_base + positions * kvw * 4, past * kvw);

    // Every tensor out of one blob, through the same resolver everything else uses.
    let shapes = [
        (qw, dim),
        (kvw, dim),
        (kvw, dim),
        (dim, qw),
        (hidden, dim),
        (hidden, dim),
        (dim, hidden),
    ];
    let mut access = 0u64;
    let mut owned = Vec::with_capacity(7);
    for (i, (rows, cols)) in shapes.iter().enumerate() {
        let span = rows + rows * cols * 2;
        let Some((bytes, charged)) = resolve_with(code_of, at, off[i], span)? else {
            return Ok(Outcome::Revert);
        };
        access = access.saturating_add(charged);
        let scales: Vec<i8> = bytes[..*rows].iter().map(|&b| b as i8).collect();
        let w: Vec<i16> = bytes[*rows..]
            .chunks_exact(2)
            .map(|c| i16::from_be_bytes([c[0], c[1]]))
            .collect();
        owned.push((scales, w));
    }
    let mut norms = Vec::with_capacity(4);
    for (j, n) in [dim, hs, hs, dim].into_iter().enumerate() {
        let Some((bytes, charged)) = resolve_with(code_of, at, off[7 + j], n * 8)? else {
            return Ok(Outcome::Revert);
        };
        access = access.saturating_add(charged);
        norms.push(
            bytes
                .chunks_exact(8)
                .map(|c| i64::from_be_bytes(c.try_into().expect("eight")))
                .collect::<Vec<i64>>(),
        );
    }

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
    kv.k[..past * kvw].copy_from_slice(&past_k);
    kv.v[..past * kvw].copy_from_slice(&past_v);
    // `forward_layer` indexes the rotary table by position, and a caller only holds the row
    // for the position it is running. Placing that row where the reference will look for it
    // keeps the call small without teaching the reference a second convention.
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
    tensor_model::qwen3::forward_layer(&cfg, &w, &rope, &mut kv, &mut scratch, 0, past, &mut x);

    let Some(total) = compute.checked_add(access) else {
        return Ok(Outcome::Revert);
    };
    if total > gas_limit {
        return Ok(Outcome::OutOfGas);
    }

    // The residual stream at eight bytes a value, then the cache entry this position added
    // at four, matching the widths this call was handed.
    let mut packed = Vec::with_capacity(dim * 8 + kvw * 2 * 4);
    for v in x.iter() {
        packed.extend_from_slice(&v.to_be_bytes());
    }
    for e in kv.k[past * kvw..(past + 1) * kvw]
        .iter()
        .chain(&kv.v[past * kvw..(past + 1) * kvw])
    {
        packed.extend_from_slice(&e.to_be_bytes());
    }
    Ok(Outcome::Return(packed, total))
}
