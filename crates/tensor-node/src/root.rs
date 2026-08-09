//! The state root: one hash committing to everything the chain holds.
//!
//! This is what makes "two nodes agree" a checkable claim rather than a hopeful one.
//! Determinism of the arithmetic was measured separately and holds across architectures,
//! but agreeing on arithmetic is not agreeing on a chain — nodes must also agree on which
//! accounts exist, what code they carry, what every storage slot contains, and how many
//! times each account has sent. A single root covers all of it, so two nodes that ran the
//! same transactions either produce the same 32 bytes or they have forked.
//!
//! It is a real Merkle-Patricia root, computed the way Ethereum computes it, rather than
//! a sorted digest that would have been easier. The easier thing would have been enough to
//! *detect* divergence, which is all this repository currently needs — but it would have
//! meant publishing a field called `stateRoot` over `eth_` JSON-RPC that no Ethereum tool
//! could check against, and a number that looks standard while meaning something else is
//! worse than an obviously custom one.

use alloy_trie::root::{state_root_unhashed, storage_root_unhashed};
use alloy_trie::TrieAccount;
use revm::primitives::{B256, KECCAK_EMPTY, U256};

impl super::chain::Chain {
    /// The Merkle-Patricia root over every account this chain holds.
    pub fn state_root(&self) -> B256 {
        state_root_unhashed(self.db.cache.accounts.iter().filter_map(|(addr, acct)| {
            // EIP-161: an account with no nonce, no balance and no code is not in the trie
            // at all. Including it would give a different root from any other client for a
            // state they would agree is identical — and the point of a real root is that
            // the comparison is meaningful outside this program.
            if is_empty(acct) {
                return None;
            }
            let storage = storage_root_unhashed(
                acct.storage
                    .iter()
                    // A slot set back to zero is a slot that does not exist. Ethereum
                    // prunes these; a cache remembers them, which would otherwise make the
                    // root depend on a node's history rather than on its state.
                    .filter(|(_, v)| !v.is_zero())
                    .map(|(k, v)| (B256::from(*k), *v)),
            );
            Some((
                *addr,
                TrieAccount {
                    nonce: acct.info.nonce,
                    balance: acct.info.balance,
                    storage_root: storage,
                    code_hash: acct.info.code_hash,
                },
            ))
        }))
    }
}

fn is_empty(acct: &revm::database::DbAccount) -> bool {
    acct.info.nonce == 0
        && acct.info.balance == U256::ZERO
        && (acct.info.code_hash == KECCAK_EMPTY || acct.info.code_hash == B256::ZERO)
}

#[cfg(test)]
mod tests {
    use crate::chain::Chain;
    use revm::primitives::Address;

    /// The root of an empty state: the hash of the RLP of the empty string. Not zero — a
    /// chain reporting zero for genesis would disagree with every other client at once.
    const EMPTY_ROOT: &str = "56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421";

    /// The well-known constant. If this is wrong, every root below it is wrong in a way
    /// that would only surface when compared against another client.
    #[test]
    fn empty_root_is_the_ethereum_constant() {
        assert_eq!(format!("{:x}", Chain::new(&[]).state_root()), EMPTY_ROOT);
    }

    /// Prefunding changes the root, and the same prefunding gives the same root. Together
    /// these say the root actually depends on state rather than being a constant.
    #[test]
    fn root_follows_state_and_only_state() {
        let a = Address::new([0x11; 20]);
        let b = Address::new([0x22; 20]);

        let empty = Chain::new(&[]).state_root();
        let one = Chain::new(&[a]).state_root();
        let one_again = Chain::new(&[a]).state_root();
        let two = Chain::new(&[a, b]).state_root();

        assert_eq!(one, one_again, "same state gave a different root");
        assert_ne!(one, empty, "funding an account did not change the root");
        assert_ne!(one, two, "a second account did not change the root");
    }

    /// How the cost of a root scales with the number of accounts.
    ///
    /// Not an assertion about speed — it prints. Uploading a real model creates one account
    /// per chunk, and Qwen3-0.6B is about 46,000 of them, so a root recomputed from scratch
    /// every block is the thing that decides whether a real upload is minutes or hours.
    #[test]
    #[ignore = "a measurement, not a check; run with --ignored"]
    fn root_cost_by_account_count() {
        use revm::state::AccountInfo;
        use std::time::Instant;
        for n in [1_000usize, 5_000, 20_000, 50_000] {
            let mut chain = Chain::new(&[]);
            for i in 0..n {
                let mut a = [0u8; 20];
                a[..8].copy_from_slice(&(i as u64).to_be_bytes());
                chain.db.insert_account_info(
                    Address::new(a),
                    AccountInfo {
                        nonce: 1,
                        ..Default::default()
                    },
                );
            }
            let t = Instant::now();
            let _ = chain.state_root();
            println!("{n:>7} accounts  {:?} per root", t.elapsed());
        }
    }

    /// Order of prefunding must not matter. It is the obvious way for two nodes to disagree
    /// while both being right, and the sort inside `state_root_unhashed` is what prevents
    /// it — this is the test that would catch its removal.
    #[test]
    fn root_is_independent_of_insertion_order() {
        let a = Address::new([0x11; 20]);
        let b = Address::new([0x22; 20]);
        assert_eq!(
            Chain::new(&[a, b]).state_root(),
            Chain::new(&[b, a]).state_root()
        );
    }
}
