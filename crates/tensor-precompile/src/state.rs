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
use tensor_kernel::{activations_in_range, matmul};

use crate::{gas_cost, BASE_GAS, MAX_DIM};

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

        // Outside this bound the kernel's accumulation would wrap, so the boundary is
        // where it has to be refused.
        if !activations_in_range(&acts) {
            return Ok(Some(revert(gas_limit)));
        }

        // Warmth is what the access actually costs, and it is only knowable from the load
        // itself. A forward pass hits the same weights account on every call of every
        // layer, so charging the cold price unconditionally would overcharge by 26x on all
        // but the first — around a million gas per token at 32B, for nothing.
        let loaded = context
            .journal_mut()
            .code(weights_at)
            .map_err(|_| "failed to load weights account".to_string())?;
        let access = if loaded.is_cold {
            COLD_ACCOUNT_GAS
        } else {
            WARM_ACCOUNT_GAS
        };
        let code = loaded.data;

        let Some(total) = compute.checked_add(access) else {
            return Ok(Some(revert(gas_limit)));
        };
        if total > gas_limit {
            return Ok(Some(oog(gas_limit)));
        }

        let Some(weights_len) = rows_u.checked_mul(cols_u).and_then(|n| n.checked_mul(2)) else {
            return Ok(Some(revert(gas_limit)));
        };
        let Some(end) = offset
            .checked_add(rows_u)
            .and_then(|n| n.checked_add(weights_len))
        else {
            return Ok(Some(revert(gas_limit)));
        };
        // A blob too short for the declared shape is the caller's error, not a reason to
        // read whatever happens to follow in memory.
        if end > code.len() {
            return Ok(Some(revert(gas_limit)));
        }

        let scales: Vec<i8> = code[offset..offset + rows_u]
            .iter()
            .map(|&b| b as i8)
            .collect();
        let weights: Vec<i16> = code[offset + rows_u..end]
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
