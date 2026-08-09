//! The chain itself: state, blocks, and executing transactions against them.
//!
//! One block per transaction, which is what a development chain does — there is no mempool
//! to order and nothing to wait for. What matters here is not the consensus, which is
//! absent, but that transactions execute under the tensor precompiles and leave state
//! behind that a later transaction reads.

use revm::context::{Context, TxEnv};
use revm::context_interface::result::{ExecutionResult, Output};
use revm::database::{CacheDB, EmptyDB};
use revm::primitives::{Address, Bytes, TxKind, B256, U256};
use revm::state::AccountInfo;
use revm::{DatabaseCommit, ExecuteEvm, MainBuilder, MainContext};
use tensor_precompile::state::TensorPrecompiles;

/// A chain designed for one workload sets its own limits, and this one has to: a layer of
/// Qwen3-0.6B is 31.5MB against EIP-170's 24KB, and a token is over a hundred million gas
/// against EIP-7825's cap of 2^24. Both are configuration rather than consensus changes,
/// and naming them here is the point — a chain that quietly assumed them would be claiming
/// results Ethereum could not reproduce.
pub const CHAIN_ID: u64 = 4242;
pub const GAS_LIMIT: u64 = 4_000_000_000;
pub const MAX_CODE_SIZE: usize = 64 * 1024 * 1024;

pub struct Receipt {
    pub hash: B256,
    pub block: u64,
    pub from: Address,
    pub to: Option<Address>,
    pub created: Option<Address>,
    pub gas_used: u64,
    pub success: bool,
    /// @dev What the transaction returned. A receipt does not carry this on Ethereum —
    ///      callers use `eth_call` for return values — but a chain whose whole point is
    ///      auditing should keep what a transaction produced next to the record that it
    ///      ran, so it is stored and served by `tensor_getOutput`.
    pub output: Bytes,
    pub logs: Vec<(Address, Vec<B256>, Bytes)>,
}

pub struct Block {
    pub number: u64,
    pub hash: B256,
    pub parent: B256,
    pub timestamp: u64,
    pub gas_used: u64,
    pub transactions: Vec<B256>,
}

pub struct Chain {
    pub db: CacheDB<EmptyDB>,
    pub blocks: Vec<Block>,
    pub receipts: Vec<Receipt>,
}

impl Chain {
    /// A chain whose genesis funds the accounts a developer will actually use.
    pub fn new(prefund: &[Address]) -> Self {
        let mut db = CacheDB::new(EmptyDB::default());
        for a in prefund {
            db.insert_account_info(
                *a,
                AccountInfo {
                    balance: U256::from(1u64) << 100,
                    ..Default::default()
                },
            );
        }
        let genesis = Block {
            number: 0,
            hash: B256::with_last_byte(1),
            parent: B256::ZERO,
            timestamp: 0,
            gas_used: 0,
            transactions: Vec::new(),
        };
        Self {
            db,
            blocks: vec![genesis],
            receipts: Vec::new(),
        }
    }

    pub fn head(&self) -> &Block {
        self.blocks.last().expect("genesis always exists")
    }

    pub fn nonce(&mut self, who: Address) -> u64 {
        self.db.load_account(who).map(|a| a.info.nonce).unwrap_or(0)
    }

    pub fn balance(&mut self, who: Address) -> U256 {
        self.db
            .load_account(who)
            .map(|a| a.info.balance)
            .unwrap_or(U256::ZERO)
    }

    pub fn code(&mut self, who: Address) -> Bytes {
        self.db
            .load_account(who)
            .ok()
            .and_then(|a| a.info.code.clone())
            .map(|c| c.original_bytes())
            .unwrap_or_default()
    }

    /// Runs a transaction without committing, for `eth_call` and `eth_estimateGas`.
    pub fn simulate(&mut self, tx: TxEnv) -> Result<(Bytes, u64, bool), String> {
        let out = self.run(tx)?;
        Ok((output_of(&out), out.tx_gas_used(), out.is_success()))
    }

    /// Executes a transaction, commits it, and seals a block around it.
    pub fn apply(&mut self, tx: TxEnv, hash: B256) -> Result<usize, String> {
        let from = tx.caller;
        let to = match tx.kind {
            TxKind::Call(a) => Some(a),
            TxKind::Create => None,
        };

        let out = {
            let evm = Context::mainnet()
                .with_db(&mut self.db)
                .modify_cfg_chained(|c| {
                    c.chain_id = CHAIN_ID;
                    c.tx_gas_limit_cap = Some(GAS_LIMIT);
                    c.limit_contract_code_size = Some(MAX_CODE_SIZE);
                })
                .build_mainnet();
            let mut evm = evm.with_precompiles(TensorPrecompiles::default());
            evm.transact(tx).map_err(|e| format!("{e:?}"))?
        };
        let result = out.result.clone();
        self.db.commit(out.state);

        let created = match &result {
            ExecutionResult::Success {
                output: Output::Create(_, addr),
                ..
            } => *addr,
            _ => None,
        };
        let logs: Vec<(Address, Vec<B256>, Bytes)> = result
            .logs()
            .iter()
            .map(|l| (l.address, l.topics().to_vec(), l.data.data.clone()))
            .collect();

        let number = self.head().number + 1;
        let parent = self.head().hash;
        self.blocks.push(Block {
            number,
            hash: block_hash(number),
            parent,
            timestamp: 1_700_000_000 + number * 2,
            gas_used: result.tx_gas_used(),
            transactions: vec![hash],
        });
        self.receipts.push(Receipt {
            hash,
            block: number,
            from,
            to,
            created,
            gas_used: result.tx_gas_used(),
            success: result.is_success(),
            output: output_of(&result),
            logs,
        });
        Ok(self.receipts.len() - 1)
    }

    fn run(&mut self, tx: TxEnv) -> Result<ExecutionResult, String> {
        let evm = Context::mainnet()
            .with_db(&mut self.db)
            .modify_cfg_chained(|c| {
                c.chain_id = CHAIN_ID;
                c.tx_gas_limit_cap = Some(GAS_LIMIT);
                c.limit_contract_code_size = Some(MAX_CODE_SIZE);
                // A simulation is not a transaction and has no place in an ordering, so
                // the nonce it carries is whatever the caller happened to send — usually
                // nothing. Enforcing it makes `eth_call` fail as soon as the account has
                // ever sent anything, which is immediately.
                c.disable_nonce_check = true;
            })
            .build_mainnet();
        let mut evm = evm.with_precompiles(TensorPrecompiles::default());
        evm.transact(tx)
            .map(|o| o.result)
            .map_err(|e| format!("{e:?}"))
    }
}

fn output_of(r: &ExecutionResult) -> Bytes {
    match r {
        ExecutionResult::Success { output, .. } => match output {
            Output::Call(b) => b.clone(),
            Output::Create(b, _) => b.clone(),
        },
        ExecutionResult::Revert { output, .. } => output.clone(),
        ExecutionResult::Halt { .. } => Bytes::new(),
    }
}

/// Deterministic and unique, which is all a development chain needs of a block hash. A real
/// one commits to the block's contents; nothing here relies on that, and pretending
/// otherwise would invite someone to.
fn block_hash(number: u64) -> B256 {
    let mut b = [0u8; 32];
    b[24..].copy_from_slice(&number.to_be_bytes());
    b[0] = 0xb1;
    B256::from(b)
}
