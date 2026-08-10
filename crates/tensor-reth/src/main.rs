//! The tensor precompiles inside reth.
//!
//! Everything else here runs revm directly. That proves a model executes correctly and
//! costs what the measurements say, and it cannot produce a chain: there is no peer-to-peer,
//! no consensus, and no ordering. Nodes agree because somebody handed them the same
//! transactions.
//!
//! reth supplies exactly the part that is missing, and asks for exactly the part this
//! project already has. Its `EvmFactory` names the precompile set as an associated type, so
//! carrying a custom one is a matter of saying which — the same `TensorPrecompiles` the
//! development node uses, unchanged.
//!
//! The version pin is the whole reason this is possible. reth's latest release still
//! carries revm 41 and these precompiles are written against 42; main moved to 42.0.1,
//! which is what this project builds against.

use alloy_evm::{EvmEnv, EvmFactory};
use reth_ethereum::evm::primitives::Database;
use reth_ethereum::evm::revm::context::result::{EVMError, HaltReason};
use reth_ethereum::evm::revm::context::{BlockEnv, Context, TxEnv};
use reth_ethereum::evm::revm::database_interface::DBErrorMarker;
use reth_ethereum::evm::revm::inspector::{Inspector, NoOpInspector};
use reth_ethereum::evm::revm::interpreter::interpreter::EthInterpreter;
use reth_ethereum::evm::revm::primitives::hardfork::SpecId;
use reth_ethereum::evm::revm::{MainBuilder, MainContext};
use alloy_evm::eth::EthEvmContext;
use reth_ethereum::evm::EthEvm;
use tensor_precompile::state::TensorPrecompiles;

/// An EVM that carries the tensor precompiles alongside Ethereum's own.
#[derive(Debug, Clone, Default)]
pub struct TensorEvmFactory;

impl EvmFactory for TensorEvmFactory {
    type Evm<DB: Database, I: Inspector<EthEvmContext<DB>, EthInterpreter>> =
        EthEvm<DB, I, TensorPrecompiles>;
    type Tx = TxEnv;
    type Error<DBError: DBErrorMarker> = EVMError<DBError>;
    type HaltReason = HaltReason;
    type Context<DB: Database> = EthEvmContext<DB>;
    type Spec = SpecId;
    type BlockEnv = BlockEnv;
    /// Not a `PrecompilesMap`, which is a table of stateless functions. A matmul reads its
    /// weights from chain state, and only a provider is handed the context to do that.
    type Precompiles = TensorPrecompiles;

    fn create_evm<DB: Database>(&self, db: DB, input: EvmEnv) -> Self::Evm<DB, NoOpInspector> {
        let evm = Context::mainnet()
            .with_db(db)
            .with_cfg(input.cfg_env)
            .with_block(input.block_env)
            .build_mainnet_with_inspector(NoOpInspector {})
            .with_precompiles(TensorPrecompiles::default());
        EthEvm::new(evm, false)
    }

    fn create_evm_with_inspector<DB: Database, I: Inspector<Self::Context<DB>, EthInterpreter>>(
        &self,
        db: DB,
        input: EvmEnv,
        inspector: I,
    ) -> Self::Evm<DB, I> {
        EthEvm::new(
            self.create_evm(db, input).into_inner().with_inspector(inspector),
            true,
        )
    }
}

fn main() -> eyre::Result<()> {
    println!("tensor-reth: precompile set builds against reth");
    Ok(())
}
