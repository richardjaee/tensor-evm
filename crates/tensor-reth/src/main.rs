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
use reth_ethereum::chainspec::ChainSpec;
use reth_ethereum::evm::EthEvmConfig;
use reth_ethereum::node::api::{FullNodeTypes, NodeTypes};
use reth_ethereum::node::builder::BuilderContext;
use reth_ethereum::node::builder::components::ExecutorBuilder;
use reth_ethereum::EthPrimitives;
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
mod precompiles;

use alloy_evm::precompiles::PrecompilesMap;
use reth_ethereum::evm::revm::handler::EthPrecompiles;

/// An EVM that carries the tensor precompiles alongside Ethereum's own.
#[derive(Debug, Clone, Default)]
pub struct TensorEvmFactory;

impl EvmFactory for TensorEvmFactory {
    type Evm<DB: Database, I: Inspector<EthEvmContext<DB>, EthInterpreter>> =
        EthEvm<DB, I, PrecompilesMap>;
    type Tx = TxEnv;
    type Error<DBError: DBErrorMarker> = EVMError<DBError>;
    type HaltReason = HaltReason;
    type Context<DB: Database> = EthEvmContext<DB>;
    type Spec = SpecId;
    type BlockEnv = BlockEnv;
    /// reth's `EthEvmConfig` fixes this to `PrecompilesMap`. That is not the obstacle it
    /// looks like: alloy-evm's dynamic precompiles are handed an `EvmInternals`, which can
    /// load accounts, and loading accounts is all weight resolution ever needed.
    type Precompiles = PrecompilesMap;

    fn create_evm<DB: Database>(&self, db: DB, input: EvmEnv) -> Self::Evm<DB, NoOpInspector> {
        // Two of Ethereum's limits have to move, and they are the same two the development
        // node raises: a token exceeds EIP-7825's 2^24 gas cap, and a layer of a real model
        // is tens of megabytes against EIP-170's 24KB. Both are configuration rather than
        // consensus changes, and both are stated here rather than assumed.
        let mut cfg = input.cfg_env;
        let spec = cfg.spec;
        cfg.tx_gas_limit_cap = Some(GAS_LIMIT);
        cfg.limit_contract_code_size = Some(MAX_CODE_SIZE);

        let evm = Context::mainnet()
            .with_db(db)
            .with_cfg(cfg)
            .with_block(input.block_env)
            .build_mainnet_with_inspector(NoOpInspector {})
            .with_precompiles(precompiles::tensor_precompiles(PrecompilesMap::from_static(
                EthPrecompiles::new(spec).precompiles,
            )));
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

/// Matching the development node, so a model that runs there runs here.
const GAS_LIMIT: u64 = 4_000_000_000;
const MAX_CODE_SIZE: usize = 64 * 1024 * 1024;

/// Executes a matmul through an EVM that reth built, against weights in chain state.
///
/// The node plumbing — network, pool, consensus — is reth's and is not what was in doubt.
/// What was in doubt is whether a precompile that reads megabytes out of state can live
/// inside reth's EVM at all, given its configuration fixes the precompile set to a table of
/// stateless functions. This answers that by running one.
fn main() -> eyre::Result<()> {
    use alloy_evm::Evm;
    use revm::database::{CacheDB, EmptyDB};
    use revm::primitives::{Address, Bytes, TxKind, U256};
    use revm::state::{AccountInfo, Bytecode};
    use tensor_precompile::state::{encode_state, weight_blob, MATMUL_STATE_ADDRESS};

    let caller = Address::new([0x11; 20]);
    let weights_at = Address::new([0x33; 20]);
    let (rows, cols) = (64usize, 96usize);

    let weights: Vec<i16> = (0..rows * cols)
        .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
        .collect();
    let scales: Vec<i8> = (0..rows).map(|r| (r % 5) as i8 + 12).collect();
    let acts: Vec<i64> = (0..cols)
        .map(|i| ((i as i64 * 40503 + 17) % 131071) - 65535)
        .collect();

    // The blob as an upload leaves it: a tag byte marking a chunk, then the weights.
    let mut blob = vec![0x00u8];
    blob.extend(weight_blob(rows, cols, &scales, &weights));

    let mut db = CacheDB::new(EmptyDB::default());
    db.insert_account_info(
        caller,
        AccountInfo {
            balance: U256::from(1u64) << 100,
            ..Default::default()
        },
    );
    db.insert_account_info(
        weights_at,
        AccountInfo {
            code: Some(Bytecode::new_raw(Bytes::from(blob))),
            ..Default::default()
        },
    );

    let mut evm = TensorEvmFactory.create_evm(db, EvmEnv::default());
    let out = evm.transact_raw(TxEnv {
        caller,
        kind: TxKind::Call(MATMUL_STATE_ADDRESS),
        data: encode_state(rows, cols, weights_at, 0, &acts).into(),
        gas_limit: 30_000_000,
        gas_price: 0,
        ..Default::default()
    })?;

    let returned = out.result.output().cloned().unwrap_or_default();
    let got: Vec<i64> = returned
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();

    let mut want = vec![0i64; rows];
    tensor_kernel::matmul_scalar(&weights, &scales, &acts, rows, cols, &mut want);

    println!("tensor-reth");
    println!("  precompiles   0x100-0x107, inside an EVM reth built");
    println!("  shape         {rows}x{cols}, weights read from chain state");
    println!("  gas           {}", out.result.gas_used());
    println!(
        "  result        {}",
        if got == want {
            "matches the kernel exactly"
        } else {
            "DIFFERS from the kernel"
        }
    );
    eyre::ensure!(got == want, "reth's EVM computed a different matmul");
    Ok(())
}

/// Ethereum's own components, with the executor swapped for one carrying the precompiles./// Ethereum's own components, with the executor swapped for one carrying the precompiles.
#[derive(Debug, Default, Clone, Copy)]
pub struct TensorExecutorBuilder;

impl<Node> ExecutorBuilder<Node> for TensorExecutorBuilder
where
    Node: FullNodeTypes<Types: NodeTypes<ChainSpec = ChainSpec, Primitives = EthPrimitives>>,
{
    type EVM = EthEvmConfig<ChainSpec, TensorEvmFactory>;

    async fn build_evm(self, ctx: &BuilderContext<Node>) -> eyre::Result<Self::EVM> {
        Ok(EthEvmConfig::new_with_evm_factory(
            ctx.chain_spec(),
            TensorEvmFactory,
        ))
    }
}

// The node launcher is not here yet. Wiring it needs reth's `test-utils` feature for an
// in-memory database, and that feature currently cannot resolve `arbitrary` against this
// toolchain — a dependency conflict rather than anything about this design. The executor
// builder above is what a node would be given.
