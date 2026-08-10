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

/// Resolves `len` bytes at `offset` within a weight blob, following a directory if the
/// address is one.
///
/// A model arrives in chunks because no transaction carries 31.5MB, so a tensor is a byte
/// range spanning several contracts and something has to stitch it back. Doing it here
/// rather than in the contract keeps the caller's view of a tensor as an address and an
/// offset, whichever way the bytes were uploaded.
///
/// Chunk and directory are told apart by the first byte of code rather than by a marker in
/// the payload, because a blob's first byte is a quantisation scale and can be anything —
/// any in-band marker could collide with real weights.
/// How a host hands over an account's code, and whether this call was the first to touch it.
///
/// An indirection rather than a concrete type because there is more than one host. revm's
/// journal is one; a client that reaches state through its own abstraction is another, and
/// weight resolution is consensus-critical enough that the two must not each have a copy of
/// it.
pub type CodeSource<'a> = &'a mut dyn FnMut(Address) -> Result<(Vec<u8>, bool), String>;

fn resolve<CTX: ContextTr>(
    context: &mut CTX,
    at: Address,
    offset: usize,
    len: usize,
) -> Result<Option<(Vec<u8>, u64)>, String> {
    let mut code_of = |a: Address| -> Result<(Vec<u8>, bool), String> {
        let loaded = context
            .journal_mut()
            .code(a)
            .map_err(|_| "failed to load weights".to_string())?;
        Ok((loaded.data.to_vec(), loaded.is_cold))
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
) -> Result<Option<(Vec<u8>, u64)>, String> {
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
        return Ok(Some((code[offset + 1..end].to_vec(), charged)));
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
    Ok(Some((out, charged)))
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
        if inputs.bytecode_address != MATMUL_STATE_ADDRESS {
            return PrecompileProvider::<CTX>::run(&mut self.inner, context, inputs);
        }

        let gas_limit = inputs.gas_limit;
        let input = inputs.input.as_bytes(context).to_vec();

        if input.len() < HEADER {
            return Ok(Some(revert(gas_limit)));
        }

        let rows = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
        let cols = u32::from_be_bytes(input[4..8].try_into().unwrap()) as u64;
        let weights_at = Address::from_slice(&input[8..28]);
        let offset = u32::from_be_bytes(input[28..32].try_into().unwrap()) as usize;

        if rows == 0 || cols == 0 || rows > MAX_DIM || cols > MAX_DIM {
            return Ok(Some(revert(gas_limit)));
        }

        let Some(compute) = gas_cost(rows, cols) else {
            return Ok(Some(revert(gas_limit)));
        };
        // Priced against the cheaper of the two access costs. The real figure depends on
        // whether the weights account is already warm, which is not known until it is
        // loaded; this is only the affordability precheck, so it must not reject a caller
        // who can in fact pay.
        let Some(floor) = compute.checked_add(WARM_ACCOUNT_GAS) else {
            return Ok(Some(revert(gas_limit)));
        };
        if floor > gas_limit {
            return Ok(Some(oog(gas_limit)));
        }

        let (rows_u, cols_u) = (rows as usize, cols as usize);

        // Activations arrive as calldata; only they scale with cols rather than rows*cols,
        // which is the entire reason this form is affordable.
        let Some(acts_len) = cols_u.checked_mul(8) else {
            return Ok(Some(revert(gas_limit)));
        };
        if input.len() != HEADER + acts_len {
            return Ok(Some(revert(gas_limit)));
        }
        let acts: Vec<i64> = input[HEADER..]
            .chunks_exact(8)
            .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
            .collect();

        let Some(weights_len) = rows_u.checked_mul(cols_u).and_then(|n| n.checked_mul(2)) else {
            return Ok(Some(revert(gas_limit)));
        };
        let Some(span) = rows_u.checked_add(weights_len) else {
            return Ok(Some(revert(gas_limit)));
        };

        // Warmth is what the access actually costs, and is only knowable from the load. A
        // forward pass hits the same weights on every call of every layer, so charging the
        // cold price unconditionally would overcharge by 26x on all but the first.
        let Some((bytes, access)) = resolve(context, weights_at, offset, span)? else {
            // Either the blob cannot hold the declared shape or the directory does not span
            // it. Either way it is the caller's error, not a reason to read past the end.
            return Ok(Some(revert(gas_limit)));
        };

        let Some(total) = compute.checked_add(access) else {
            return Ok(Some(revert(gas_limit)));
        };
        if total > gas_limit {
            return Ok(Some(oog(gas_limit)));
        }

        let scales: Vec<i8> = bytes[..rows_u].iter().map(|&b| b as i8).collect();
        let weights: Vec<i16> = bytes[rows_u..]
            .chunks_exact(2)
            .map(|c| i16::from_be_bytes([c[0], c[1]]))
            .collect();

        let mut out = vec![0i64; rows_u];
        matmul(&weights, &scales, &acts, rows_u, cols_u, &mut out);

        let mut bytes = Vec::with_capacity(rows_u * 8);
        for v in out {
            bytes.extend_from_slice(&v.to_be_bytes());
        }

        let mut gas = Gas::new(gas_limit);
        // Affordability was established before the read; this cannot fail, and treating
        // it as OOG rather than asserting keeps a future refactor from turning a pricing
        // mistake into a node panic.
        if !gas.record_regular_cost(total) {
            return Ok(Some(oog(gas_limit)));
        }
        Ok(Some(InterpreterResult::new(
            InstructionResult::Return,
            Bytes::from(bytes),
            gas,
        )))
    }

    fn warm_addresses(&self) -> &revm::primitives::AddressSet {
        PrecompileProvider::<CTX>::warm_addresses(&self.inner)
    }

    fn contains(&self, address: &Address) -> bool {
        *address == MATMUL_STATE_ADDRESS
            || *address == crate::classify::CLASSIFY_ADDRESS
            || *address == crate::router::ROUTER_ADDRESS
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
