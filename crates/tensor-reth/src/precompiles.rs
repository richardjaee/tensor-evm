//! The tensor precompiles, as reth's EVM wants them.
//!
//! reth's `EthEvmConfig` fixes the precompile set to `PrecompilesMap`, so the provider the
//! development node uses cannot be dropped in whole. That looks at first like a wall — a
//! `PrecompilesMap` is a table of functions, and a matmul has to read megabytes of weights
//! out of chain state.
//!
//! It is not. alloy-evm's dynamic precompiles are handed an `EvmInternals`, which can load
//! accounts, and that is the only thing weight resolution ever needed. So the adapter is
//! thin: turn `EvmInternals` into the code source the resolver already takes, and call the
//! same function the provider calls.
//!
//! What matters is that it *is* the same function. A second implementation of what a matmul
//! computes would be a second opinion about what a model says, which is the one thing a
//! chain cannot have two of.

use alloy_evm::precompiles::{DynPrecompile, PrecompileInput, PrecompilesMap};
use alloy_primitives::Address;
use revm::precompile::{PrecompileOutput, PrecompileResult};
use revm::primitives::Bytes;
use tensor_precompile::state::{
    classify_state_with, matmul_state_with, router_state_with, Outcome, MATMUL_STATE_ADDRESS,
};

/// Ethereum's precompiles, plus the tensor set.
pub fn tensor_precompiles(base: PrecompilesMap) -> PrecompilesMap {
    let mut map = base;
    map.extend_precompiles([
        (MATMUL_STATE_ADDRESS, state_reading(matmul_state_with)),
        (
            tensor_precompile::classify::CLASSIFY_ADDRESS,
            state_reading(classify_state_with),
        ),
        (
            tensor_precompile::router::ROUTER_ADDRESS,
            state_reading(router_state_with),
        ),
        (
            tensor_precompile::MATMUL_ADDRESS,
            stateless(tensor_precompile::matmul_precompile),
        ),
        (
            tensor_precompile::activation::SWIGLU_ADDRESS,
            stateless(tensor_precompile::activation::swiglu_precompile),
        ),
        (
            tensor_precompile::norm::NORM_ROPE_ADDRESS,
            stateless(tensor_precompile::norm::norm_rope_precompile),
        ),
        (
            tensor_precompile::attention::ATTENTION_ADDRESS,
            stateless(tensor_precompile::attention::attention_precompile),
        ),
        (
            tensor_precompile::attention::RMSNORM_ADDRESS,
            stateless(tensor_precompile::attention::rmsnorm_precompile),
        ),
    ]);
    map
}

/// A precompile that needs nothing but its calldata, wrapped for alloy-evm.
fn stateless(f: fn(&[u8], u64, u64) -> PrecompileResult) -> DynPrecompile {
    DynPrecompile::from(move |input: PrecompileInput<'_>| f(input.data, input.gas, 0))
}

/// A precompile whose operands live in chain state, wrapped for alloy-evm.
///
/// The three of them — matmul, classifier, router — differ only in which function they
/// call, so they share the adapter. Each is the same function the development node runs.
fn state_reading(
    run: fn(tensor_precompile::state::CodeSource<'_>, &[u8], u64) -> Result<Outcome, String>,
) -> DynPrecompile {
    DynPrecompile::from(move |mut input: PrecompileInput<'_>| -> PrecompileResult {
        let data = input.data.to_vec();
        let gas_limit = input.gas;

        // Whether a touch was cold is the difference between 2,600 gas and 100, and a
        // forward pass hits the same weights on every call of every layer — so it is taken
        // from the host rather than assumed.
        let mut code_of = |a: Address| -> Result<(Vec<u8>, bool), String> {
            // `load_account` alone is not enough: code is loaded lazily, so an account read
            // that way carries a code hash and an empty `code`. Weights would come back as
            // nothing and every call would revert — which an in-memory database hides,
            // because it keeps code inline in the account.
            let mut loaded = input
                .internals
                .load_account_code(a)
                .map_err(|_| "failed to load weights".to_string())?;
            let is_cold = loaded.is_cold;
            let code = loaded
                .data
                .load_code()
                .map_err(|_| "failed to load weights".to_string())?
                .original_bytes()
                .to_vec();
            Ok((code, is_cold))
        };

        match run(&mut code_of, &data, gas_limit) {
            Ok(Outcome::Return(bytes, used)) => {
                Ok(PrecompileOutput::new(used, Bytes::from(bytes), 0))
            }
            // A revert here is a caller error — a shape the blob cannot hold, or a
            // directory that does not span the range asked for.
            Ok(Outcome::Revert) | Err(_) => Ok(PrecompileOutput::revert(
                tensor_precompile::BASE_GAS,
                Bytes::new(),
                0,
            )),
            Ok(Outcome::OutOfGas) => Ok(PrecompileOutput::halt(
                revm::precompile::PrecompileHalt::OutOfGas,
                0,
            )),
        }
    })
}
