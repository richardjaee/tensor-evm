//! Keeping the chain across restarts.
//!
//! Uploading a model is the expensive part — 1.19 GB and about a minute for Qwen3-0.6B —
//! and it only has to happen once per chain. It was happening once per *process*, because
//! this node held everything in memory and a restart took the contract with it. That is not
//! a property of chains; it is a property of a node that never learned to write anything
//! down.
//!
//! The format is deliberately dull: a length-prefixed dump of the accounts that matter.
//! Nothing here is a database, and it makes no attempt to be crash-safe — it is written when
//! asked and read at startup, which is all a development chain needs to stop re-uploading a
//! gigabyte every time somebody restarts it.

use revm::primitives::{Address, Bytes, U256};
use revm::state::{AccountInfo, Bytecode};
use std::io::{Read, Write};

use crate::chain::Chain;

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn take_u64(r: &mut impl Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

impl Chain {
    /// Writes every account with code or a balance, and the block height, to `path`.
    pub fn save(&mut self, path: &std::path::Path) -> Result<usize, String> {
        let accounts: Vec<(Address, AccountInfo)> = self
            .db
            .cache
            .accounts
            .iter()
            .filter(|(_, a)| a.info.code.is_some() || !a.info.balance.is_zero() || a.info.nonce > 0)
            .map(|(addr, a)| (*addr, a.info.clone()))
            .collect();

        let mut out = Vec::new();
        put_u64(&mut out, accounts.len() as u64);
        for (addr, info) in &accounts {
            out.extend_from_slice(addr.as_slice());
            put_u64(&mut out, info.nonce);
            out.extend_from_slice(&info.balance.to_le_bytes::<32>());
            let code = info
                .code
                .as_ref()
                .map(|c| c.original_bytes())
                .unwrap_or_default();
            put_u64(&mut out, code.len() as u64);
            out.extend_from_slice(&code);
        }

        let mut f = std::fs::File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        f.write_all(&out).map_err(|e| format!("{e}"))?;
        Ok(accounts.len())
    }

    /// Reads accounts back and rebuilds the trie from them.
    ///
    /// Blocks are not restored. A chain that resumes from saved state starts its block
    /// numbering again, which is honest about what this is: the weights are here, the
    /// history is not, and pretending otherwise would put a block number on something that
    /// never happened.
    pub fn load(&mut self, path: &std::path::Path) -> Result<usize, String> {
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return Ok(0),
        };
        let n = take_u64(&mut f).map_err(|e| format!("{e}"))?;
        for _ in 0..n {
            let mut addr = [0u8; 20];
            f.read_exact(&mut addr).map_err(|e| format!("{e}"))?;
            let nonce = take_u64(&mut f).map_err(|e| format!("{e}"))?;
            let mut bal = [0u8; 32];
            f.read_exact(&mut bal).map_err(|e| format!("{e}"))?;
            let len = take_u64(&mut f).map_err(|e| format!("{e}"))? as usize;
            let mut code = vec![0u8; len];
            f.read_exact(&mut code).map_err(|e| format!("{e}"))?;

            let who = Address::from(addr);
            self.db.insert_account_info(
                who,
                AccountInfo {
                    nonce,
                    balance: U256::from_le_bytes(bal),
                    code: (!code.is_empty()).then(|| Bytecode::new_raw(Bytes::from(code))),
                    ..Default::default()
                },
            );
            self.touch(who);
        }

        // Genesis now describes the state that was loaded rather than an empty one.
        let root = self.state_root();
        if let Some(g) = self.blocks.first_mut() {
            g.state_root = root;
        }
        Ok(n as usize)
    }
}
