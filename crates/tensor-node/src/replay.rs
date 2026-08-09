//! Re-execution: the thing this whole design exists to make possible.
//!
//! A chain's claim is not that its operator computed something correctly. It is that anyone
//! else can take the same transactions, run them independently, and get the same state — so
//! that a wrong answer is not a matter of trust but of arithmetic anyone can redo. Every
//! other measurement in this repository is a precondition for that claim. This is the claim
//! itself.
//!
//! So a node journals the raw signed transactions it accepted, and `--replay` takes such a
//! journal and executes it from genesis on a machine that has never spoken to the original
//! node. The journal carries signatures, so replay verifies each sender rather than being
//! told who they were, and it is the same code path as the live node rather than a
//! reimplementation that could agree by being wrong in the same way — the divergence being
//! looked for is between machines, so sharing the code is correct here.
//!
//! ```text
//! tensor-node --journal run.txt          # record while serving
//! tensor-node --replay  run.txt          # re-execute anywhere, print roots
//! ```

use alloy_consensus::{transaction::SignerRecoverable, Transaction, TxEnvelope};
use alloy_rlp::Decodable;
use revm::context::TxEnv;
use revm::primitives::B256;
use std::io::Write;
use std::path::Path;

use crate::chain::{Chain, CHAIN_ID};

/// Decodes a signed transaction, applies it, and journals it if this node is recording.
///
/// Shared by the RPC handler and by replay, so a replayed block is produced by exactly the
/// code that produced the original.
pub fn apply_raw(chain: &mut Chain, raw: &[u8]) -> Result<B256, String> {
    let env = TxEnvelope::decode(&mut &raw[..]).map_err(|e| format!("undecodable: {e}"))?;
    let from = env
        .recover_signer()
        .map_err(|e| format!("unrecoverable sender: {e}"))?;
    let hash = *env.hash();

    // The envelope implements `Transaction` across all of its variants, so a legacy, 2930,
    // 1559 or 4844 transaction is read the same way.
    let tx = TxEnv {
        caller: from,
        kind: env.kind(),
        data: env.input().clone(),
        value: env.value(),
        gas_limit: env.gas_limit(),
        gas_price: 0,
        nonce: env.nonce(),
        chain_id: Some(CHAIN_ID),
        ..Default::default()
    };
    chain.apply(tx, hash, raw.to_vec().into())?;

    // Journalled only after it succeeded, so a replay of the journal reproduces the chain
    // rather than a chain plus the transactions this node rejected.
    if let Some(path) = &chain.journal {
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", hex(raw));
        }
    }
    Ok(hash)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

/// Parses a journal into the raw signed transactions it records.
pub fn read_journal(path: &Path) -> Result<Vec<Vec<u8>>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| unhex(l).ok_or_else(|| format!("not hex: {l}")))
        .collect()
}

/// Executes a journal from genesis, silently, and hands back the chain it produced.
pub fn execute(txs: &[Vec<u8>], prefund: &[revm::primitives::Address]) -> Result<Chain, String> {
    let mut chain = Chain::new(prefund);
    for (i, raw) in txs.iter().enumerate() {
        apply_raw(&mut chain, raw).map_err(|e| format!("transaction {i}: {e}"))?;
    }
    Ok(chain)
}

/// Executes a journal from genesis and reports the state root after every block.
///
/// Prints per block rather than only at the end because a divergence is far cheaper to
/// diagnose when you know which transaction introduced it — and if two machines ever do
/// disagree, that first differing line is the entire investigation.
pub fn replay(path: &Path, prefund: &[revm::primitives::Address]) -> Result<(), String> {
    let txs = read_journal(path)?;
    println!(
        "replaying {} transactions from {}",
        txs.len(),
        path.display()
    );
    println!("  arch            {}", std::env::consts::ARCH);
    println!("  os              {}", std::env::consts::OS);

    // Executed through the same function the tests call, then printed from the blocks it
    // left behind. A second loop here could quietly diverge from the one under test.
    let chain = execute(&txs, prefund)?;

    println!("\n{:>5}  {:>12}  state root", "block", "gas");
    println!("{}", "-".repeat(74));
    for b in &chain.blocks {
        println!("{:>5}  {:>12}  {:#x}", b.number, b.gas_used, b.state_root);
    }

    // One line to compare between machines, for the same reason the determinism report has
    // one: agreement should be checkable across two terminals without reading a table.
    println!("\nFINAL {:#x}", chain.head().state_root);
    let failed = chain.receipts.iter().filter(|r| !r.success).count();
    println!(
        "{} transactions, {} reverted, {} blocks",
        chain.receipts.len(),
        failed,
        chain.head().number
    );
    println!(
        "\nTwo nodes that executed this journal and report different roots have forked.\n\
         The first differing line above is the block where they parted."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use revm::primitives::Address;
    use std::path::PathBuf;

    /// The account `demo.sh` signs with, which is every development chain's first key.
    const DEV: Address = Address::new([
        0xf3, 0x9f, 0xd6, 0xe5, 0x1a, 0xad, 0x88, 0xf6, 0xf4, 0xce, 0x6a, 0xb8, 0x82, 0x72, 0x79,
        0xcf, 0xff, 0xb9, 0x22, 0x66,
    ]);

    /// The state root after replaying `testdata/demo.journal`, measured on aarch64/macOS,
    /// x86_64 Intel and x86_64 AMD alike.
    ///
    /// Pinned rather than merely recomputed, for the same reason the determinism digest is:
    /// anything that moves it — the arithmetic, gas accounting, how an account is encoded —
    /// is by definition something that would fork a chain, and should cost somebody a
    /// deliberate edit rather than pass unnoticed.
    const EXPECTED_ROOT: &str =
        "0x5618b7f89cf47864ee58611808c9263d2d2d006b1df06aadd574c586c2810839";

    fn journal() -> Vec<Vec<u8>> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root")
            .join("testdata/demo.journal");
        read_journal(&path).expect("journal must parse")
    }

    /// Re-executing the recorded transactions reproduces the chain they came from. This is
    /// the whole claim of the design, reduced to an assertion.
    #[test]
    fn replaying_the_journal_reproduces_the_state() {
        let chain = execute(&journal(), &[DEV]).expect("replay must succeed");
        assert_eq!(format!("{:#x}", chain.head().state_root), EXPECTED_ROOT);
        assert!(
            chain.receipts.iter().all(|r| r.success),
            "a recorded transaction reverted on replay"
        );
    }

    /// Two independently constructed chains agree. Within one process this only rules out
    /// state leaking between runs — the cross-machine version of this check lives in
    /// docs/results.md, because it needs two machines.
    #[test]
    fn two_independent_runs_agree() {
        let txs = journal();
        let a = execute(&txs, &[DEV]).expect("first run");
        let b = execute(&txs, &[DEV]).expect("second run");
        assert_eq!(a.head().state_root, b.head().state_root);
        for (x, y) in a.blocks.iter().zip(&b.blocks) {
            assert_eq!(x.state_root, y.state_root, "diverged at block {}", x.number);
        }
    }

    /// A journal is only worth replaying if a changed transaction changes the result.
    /// Without this, the test above would pass against a root that ignored the input.
    #[test]
    fn a_tampered_journal_does_not_reproduce_it() {
        let mut txs = journal();
        let last = txs.last_mut().expect("journal is not empty");
        let n = last.len();
        last[n - 20] ^= 0x01;
        // Either the signature no longer recovers, or it recovers to somebody else whose
        // transaction lands differently. Both are refusals to reproduce; neither may be a
        // silent match.
        match execute(&txs, &[DEV]) {
            Err(_) => {}
            Ok(c) => assert_ne!(format!("{:#x}", c.head().state_root), EXPECTED_ROOT),
        }
    }
}
