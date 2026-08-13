//! The signatures callers hash to reach `Generator`, in one place.
//!
//! These were copied by hand into three examples and the CLI. Then routing grew `Config` by
//! `nExperts`, `topK` and `expertStride` and `Layer` by `router`; the CLI was updated and
//! the examples were not, so they hashed a selector no longer on the contract, hit the
//! fallback, and reverted with no data. That reads as a broken model rather than a stale
//! ABI, and it survived because nothing compared a caller's signature against the contract.
//!
//! A constant does not fix that on its own — it makes all four wrong together instead of
//! separately. What fixes it is [`crate::abi::tests`], which rebuilds these strings from the
//! compiled artifact and fails when they drift. The constant is what gives that check one
//! thing to test rather than four.

/// A `Config` tuple: the widths, `rmsEps`, then the three routing fields.
///
/// The routing fields sit after `rmsEps` rather than among the widths so a dense model's
/// encoding is unchanged up to the point it stops caring. Dense callers pass zero.
pub const CONFIG: &str =
    "(uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,uint32,uint32,uint32)";

/// A `Layer` tuple: where the tensors live, and the offset of each within that blob.
pub const LAYER: &str =
    "(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32)";

/// Advance a session through as many layers as the gas allows.
pub fn step() -> String {
    format!("step(bytes32,({CONFIG},{LAYER}[],(bytes,bytes)),uint16)")
}

/// Classify the residual stream and record the token.
pub fn emit_token() -> String {
    format!("emitToken(bytes32,{CONFIG},bytes,address,uint32,uint32)")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    /// The compiled artifact, or `None` when the contracts have not been built.
    ///
    /// Skipping rather than failing when it is absent, because `cargo test` has to pass
    /// without `forge` installed. `verify.sh` builds the contracts, so the check has teeth
    /// where it counts and stays out of the way where it cannot run.
    fn artifact() -> Option<String> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()?
            .parent()?
            .join("out/Generator.sol/Generator.json");
        std::fs::read_to_string(root).ok()
    }

    /// Rebuild a function's canonical signature from the artifact's ABI.
    ///
    /// Walks the ABI directly rather than deriving types: the shape wanted here is narrow —
    /// the entry for `name`, then its inputs, each flattened to its type, with tuples
    /// spelled out from `components`. Anything unexpected returns `None` and fails loudly.
    fn signature_of(json: &str, name: &str) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        let abi = v.get("abi")?.as_array()?;
        let f = abi.iter().find(|e| {
            e.get("type").and_then(|t| t.as_str()) == Some("function")
                && e.get("name").and_then(|n| n.as_str()) == Some(name)
        })?;
        fn ty(p: &serde_json::Value) -> Option<String> {
            let t = p.get("type")?.as_str()?;
            // A tuple's own type is "tuple" or "tuple[]"; the members are in `components`,
            // and the canonical form spells them out in place.
            if let Some(rest) = t.strip_prefix("tuple") {
                let inner: Option<Vec<String>> =
                    p.get("components")?.as_array()?.iter().map(ty).collect();
                return Some(format!("({}){rest}", inner?.join(",")));
            }
            Some(t.to_string())
        }
        let args: Option<Vec<String>> = f.get("inputs")?.as_array()?.iter().map(ty).collect();
        Some(format!("{name}({})", args?.join(",")))
    }

    /// What the callers hash must be what the contract answers to.
    ///
    /// This is the check that was missing. A struct can gain a field without any signature
    /// being compared to anything, and the only symptom is an empty revert from a fallback
    /// — which looks like broken arithmetic, not a stale ABI, and cost an afternoon.
    #[test]
    fn signatures_match_the_compiled_contract() {
        let Some(json) = artifact() else {
            eprintln!("out/Generator.sol/Generator.json missing — run forge build");
            return;
        };
        for (name, ours) in [("step", super::step()), ("emitToken", super::emit_token())] {
            let theirs = signature_of(&json, name)
                .unwrap_or_else(|| panic!("no {name} in the compiled ABI"));
            assert_eq!(
                ours, theirs,
                "\n{name} drifted from the contract.\n  callers hash: {ours}\n  contract has: {theirs}\n"
            );
        }
    }
}
