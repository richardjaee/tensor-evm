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
use reth_ethereum::node::builder::components::ExecutorBuilder;
use reth_ethereum::node::builder::{BuilderContext, NodeBuilder, NodeHandle};
use reth_ethereum::node::core::args::RpcServerArgs;
use reth_ethereum::node::core::node_config::NodeConfig;
use reth_ethereum::node::{EthereumAddOns, EthereumNode};
use reth_ethereum::tasks::Runtime;
use reth_ethereum::EthPrimitives;
use std::sync::Arc;
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
        // Two of Ethereum's limits have to move, and they are the same two the development
        // node raises: a token exceeds EIP-7825's 2^24 gas cap, and a layer of a real model
        // is tens of megabytes against EIP-170's 24KB. Both are configuration rather than
        // consensus changes, and both are stated here rather than assumed.
        let mut cfg = input.cfg_env;
        cfg.tx_gas_limit_cap = Some(GAS_LIMIT);
        cfg.limit_contract_code_size = Some(MAX_CODE_SIZE);

        let evm = Context::mainnet()
            .with_db(db)
            .with_cfg(cfg)
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

/// Matching the development node, so a model that runs there runs here.
const GAS_LIMIT: u64 = 4_000_000_000;
const MAX_CODE_SIZE: usize = 64 * 1024 * 1024;

/// Ethereum's own components, with the executor swapped for one carrying the precompiles.
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

/// A genesis that funds the account every development chain funds first, and sets a block
/// gas limit a token can actually fit inside.
fn chain_spec() -> Arc<ChainSpec> {
    let genesis: alloy_genesis::Genesis = serde_json::from_str(
        r#"{
            "nonce": "0x0",
            "timestamp": "0x0",
            "extraData": "0x74656e736f72",
            "gasLimit": "0xee6b2800",
            "difficulty": "0x0",
            "alloc": {
                "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266": {
                    "balance": "0x100000000000000000000000000000"
                }
            },
            "config": {
                "chainId": 4242,
                "homesteadBlock": 0, "eip150Block": 0, "eip155Block": 0, "eip158Block": 0,
                "byzantiumBlock": 0, "constantinopleBlock": 0, "petersburgBlock": 0,
                "istanbulBlock": 0, "berlinBlock": 0, "londonBlock": 0,
                "shanghaiTime": 0, "cancunTime": 0, "pragueTime": 0,
                "terminalTotalDifficulty": 0,
                "terminalTotalDifficultyPassed": true
            }
        }"#,
    )
    .expect("genesis");
    Arc::new(ChainSpec::from_genesis(genesis))
}

#[tokio::main]
async fn main() -> eyre::Result<()> {
    let _guard = reth_tracing::Tracer::init(&reth_tracing::RethTracer::new())?;

    let config = NodeConfig::test()
        .dev()
        .with_rpc(RpcServerArgs::default().with_http())
        .with_chain(chain_spec());

    let NodeHandle {
        node: _node,
        node_exit_future,
    } = NodeBuilder::new(config)
        .testing_node(Runtime::test())
        .with_types::<EthereumNode>()
        // Ethereum's components throughout, except the executor, which carries the tensor
        // precompiles. Everything that makes this a chain rather than a harness — the
        // network, the pool, the payload builder, consensus — is reth's, untouched.
        .with_components(EthereumNode::components().executor(TensorExecutorBuilder))
        .with_add_ons(EthereumAddOns::default())
        .launch()
        .await?;

    println!("tensor-reth: precompiles 0x100-0x107 live, chain 4242, RPC on 8545");
    node_exit_future.await
}
