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
    /// The signed transaction exactly as it arrived. Kept because every honest answer about
    /// a transaction — its nonce, its input, who signed it — is derivable from these bytes
    /// and from nothing else a receipt happens to remember.
    pub raw: Bytes,
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
    /// The state root after this block. Recorded per block rather than only at the tip so
    /// two nodes that diverge can be compared at the block where they first differed,
    /// which is the difference between knowing there is a bug and knowing where it is.
    pub state_root: B256,
    /// The Merkle root over this block's transactions, computed rather than asserted. A
    /// header that claims the empty root while carrying transactions is lying in a field
    /// clients are entitled to check.
    pub tx_root: B256,
    pub timestamp: u64,
    pub gas_used: u64,
    pub transactions: Vec<B256>,
}

pub struct Chain {
    pub db: CacheDB<EmptyDB>,
    pub blocks: Vec<Block>,
    pub receipts: Vec<Receipt>,
    /// The account trie, kept between blocks so a block pays for what it changed rather
    /// than for everything that has ever existed.
    pub trie: crate::trie::Trie,
    /// Whether receipts keep the signed transaction they came from.
    ///
    /// On by default, because it is what lets the node answer honestly about a transaction
    /// rather than from what a receipt happened to remember. It also means a bulk upload
    /// keeps a second copy of everything uploaded: 61GB of weights arrive as 61GB of
    /// transactions, and a machine that can hold the state cannot necessarily hold both.
    /// Turning it off trades that history for the room to hold a large model.
    pub archive_txs: bool,
    /// Where to append the raw signed transactions this node accepted, if it is recording.
    /// A journal is what lets somebody else re-execute the chain without having to trust
    /// this node's account of what happened.
    pub journal: Option<std::path::PathBuf>,
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
        let mut chain = Self {
            db,
            blocks: Vec::new(),
            receipts: Vec::new(),
            trie: crate::trie::Trie::new(),
            archive_txs: true,
            journal: None,
        };
        for a in prefund {
            chain.touch(*a);
        }
        let genesis_root = chain.state_root();
        chain.blocks.push(Block {
            number: 0,
            hash: B256::with_last_byte(1),
            parent: B256::ZERO,
            state_root: genesis_root,
            tx_root: alloy_trie::root::ordered_trie_root_with_encoder(
                &[] as &[&Bytes],
                |item: &&Bytes, buf: &mut Vec<u8>| buf.extend_from_slice(item),
            ),
            timestamp: 0,
            gas_used: 0,
            transactions: Vec::new(),
        });
        chain
    }

    /// Brings one account's leaf up to date, inserting or removing as its state requires.
    ///
    /// Called for every account a transaction touched, which is the set revm hands back —
    /// so the trie is updated exactly where the state changed and nowhere else.
    pub fn touch(&mut self, who: Address) {
        let key = revm::primitives::keccak256(who);
        let leaf = self.db.cache.accounts.get(&who).and_then(crate::root::leaf);
        match leaf {
            Some(v) => self.trie.insert(key, v),
            // An account that has become empty leaves the trie: EIP-161 says it is not in
            // the state, and leaving a stale leaf behind would put this node's root at odds
            // with every other client's.
            None => self.trie.remove(key),
        }
    }

    /// The state root, from the trie this chain maintains.
    pub fn state_root(&mut self) -> B256 {
        self.trie.root()
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
    pub fn apply(&mut self, tx: TxEnv, hash: B256, raw: Bytes) -> Result<usize, String> {
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
        // The addresses the transaction touched, taken before the state is consumed. This
        // is the whole reason a block can be cheap: it is the trie's dirty set.
        let touched: Vec<Address> = out.state.keys().copied().collect();
        self.db.commit(out.state);
        for a in touched {
            self.touch(a);
        }

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
        // After the commit above, so this is the state the block leaves behind rather than
        // the one it started from.
        let state_root = self.state_root();
        self.blocks.push(Block {
            number,
            hash: block_hash(number),
            parent,
            state_root,
            tx_root: alloy_trie::root::ordered_trie_root_with_encoder(&[&raw], |item, buf| {
                buf.extend_from_slice(item)
            }),
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
            raw: if self.archive_txs { raw } else { Bytes::new() },
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
