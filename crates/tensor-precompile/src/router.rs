//! The mixture-of-experts router.
//!
//! Selection belongs in the VM rather than in a contract, and not for speed. A router
//! decides which parameters a token reads, so two nodes that break a tie differently do not
//! round differently — they run different weights and emit different text. Putting the
//! comparison here means every node runs the same comparison, written once, rather than
//! whatever a contract author's sort happened to do.
//!
//! It also has to be cheap in the one way that matters. The router itself is a tiny matmul —
//! `n_experts` rows against the model width — so its arithmetic is nothing. Its value is
//! that it lets the caller skip reading the experts it did not choose, and those are the
//! megabytes.

use revm::precompile::{
    u64_to_address, Precompile, PrecompileHalt, PrecompileId, PrecompileOutput, PrecompileResult,
};
use revm::primitives::{Address, Bytes};
use tensor_kernel::fixed::{exp_q32, top_k, F, ONE};

use crate::{BASE_GAS, MACS_PER_GAS, MAX_DIM};

/// Address of the router.
pub const ROUTER_ADDRESS: Address = u64_to_address(0x107);

/// ```text
/// [0..4)    experts u32, how many the router scores
/// [4..8)    cols    u32, the model width
/// [8..28)   weights address whose code holds the router tensor
/// [28..32)  offset  u32, byte offset within that code
/// [32..36)  k       u32, how many experts to choose
/// [36 ..)   x       i64 x cols
/// ```
///
/// Output is `k` pairs of `(u32 index, i64 weight)`, best-scoring first — 12 bytes each.
/// Weights are Q32 and sum to one across the chosen.
pub const HEADER: usize = 36;

/// The most experts a single router call may score.
///
/// Well above any published model — DeepSeek-V3 has 256 — and present so that a malformed
/// header cannot ask for an allocation proportional to a 32-bit field.
pub const MAX_EXPERTS: u64 = 1 << 16;

pub fn gas_cost(experts: u64, cols: u64) -> Option<u64> {
    let macs = experts.checked_mul(cols)?;
    BASE_GAS.checked_add(macs.div_ceil(MACS_PER_GAS))
}

/// Chooses `k` experts from `scores` and returns them with their routing weights.
///
/// The softmax is over the chosen scores alone rather than over all of them. Normalising
/// across every expert and then dropping the unchosen leaves the kept weights summing to
/// less than one, which silently scales the whole feed-forward down — self-consistent,
/// plausible, and wrong.
pub fn route(scores: &[i64], k: usize) -> Vec<(u32, i64)> {
    let mut chosen = Vec::with_capacity(k);
    top_k(scores, k, &mut chosen);

    // Shifted by the maximum before exponentiating: scores are logits and can be large and
    // positive, where `exp_q32` would overflow. Subtracting the maximum is exact here.
    let max = chosen.iter().map(|&(_, v)| v).max().unwrap_or(0);
    let mut exps = Vec::with_capacity(chosen.len());
    let mut sum: u128 = 0;
    for &(_, v) in &chosen {
        let e = exp_q32(v - max) as u128;
        exps.push(e);
        sum += e;
    }

    // Every exponent underflowing means the scores are further apart than the arithmetic
    // can express. Giving the whole weight to the best-scoring expert is deterministic,
    // which dividing by zero is not.
    let collapsed = sum == 0;
    chosen
        .iter()
        .enumerate()
        .map(|(i, &(idx, _))| {
            let w = if collapsed {
                if i == 0 {
                    ONE
                } else {
                    0
                }
            } else {
                ((exps[i] << F) / sum) as i64
            };
            (idx as u32, w)
        })
        .collect()
}

/// Packs the routing decision as the precompile returns it.
pub fn encode_output(chosen: &[(u32, i64)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(chosen.len() * 12);
    for &(i, w) in chosen {
        out.extend_from_slice(&i.to_be_bytes());
        out.extend_from_slice(&w.to_be_bytes());
    }
    out
}

/// Reads a routing decision back.
pub fn decode_output(b: &[u8]) -> Option<Vec<(u32, i64)>> {
    if !b.len().is_multiple_of(12) {
        return None;
    }
    Some(
        b.chunks_exact(12)
            .map(|c| {
                (
                    u32::from_be_bytes(c[0..4].try_into().expect("four bytes")),
                    i64::from_be_bytes(c[4..12].try_into().expect("eight bytes")),
                )
            })
            .collect(),
    )
}

/// Encodes a call.
pub fn encode(
    experts: u32,
    cols: u32,
    weights: Address,
    offset: u32,
    k: u32,
    x: &[i64],
) -> Vec<u8> {
    let mut v = Vec::with_capacity(HEADER + x.len() * 8);
    v.extend_from_slice(&experts.to_be_bytes());
    v.extend_from_slice(&cols.to_be_bytes());
    v.extend_from_slice(weights.as_slice());
    v.extend_from_slice(&offset.to_be_bytes());
    v.extend_from_slice(&k.to_be_bytes());
    for &a in x {
        v.extend_from_slice(&a.to_be_bytes());
    }
    v
}

fn bad_input(reservoir: u64) -> PrecompileResult {
    Ok(PrecompileOutput::revert(BASE_GAS, Bytes::new(), reservoir))
}

/// Validates a header and reports what it asked for, so the provider and this entry point
/// agree about what is well-formed.
pub fn parse(input: &[u8]) -> Option<(u64, u64, Address, usize, usize)> {
    if input.len() < HEADER {
        return None;
    }
    let experts = u32::from_be_bytes(input[0..4].try_into().ok()?) as u64;
    let cols = u32::from_be_bytes(input[4..8].try_into().ok()?) as u64;
    let at = Address::from_slice(&input[8..28]);
    let offset = u32::from_be_bytes(input[28..32].try_into().ok()?) as usize;
    let k = u32::from_be_bytes(input[32..36].try_into().ok()?) as usize;

    if experts == 0 || cols == 0 || cols > MAX_DIM || experts > MAX_EXPERTS {
        return None;
    }
    // Choosing more experts than exist is a caller error rather than something to clamp:
    // silently returning fewer than asked would leave the caller's routing weights short.
    if k == 0 || k as u64 > experts {
        return None;
    }
    if input.len() != HEADER + (cols as usize) * 8 {
        return None;
    }
    Some((experts, cols, at, offset, k))
}

pub fn router_precompile(input: &[u8], gas_limit: u64, reservoir: u64) -> PrecompileResult {
    let Some((experts, cols, _, _, _)) = parse(input) else {
        return bad_input(reservoir);
    };
    let Some(gas_used) = gas_cost(experts, cols) else {
        return bad_input(reservoir);
    };
    if gas_used > gas_limit {
        return Ok(PrecompileOutput::halt(PrecompileHalt::OutOfGas, reservoir));
    }
    // The router tensor lives in chain state, which only the provider can reach. This entry
    // point exists so the address is a member of the precompile set — membership is what
    // makes the interpreter treat it as one — and the provider intercepts it before this
    // runs.
    bad_input(reservoir)
}

pub fn precompile() -> Precompile {
    Precompile::new(
        PrecompileId::Custom("tensor-router".into()),
        ROUTER_ADDRESS,
        router_precompile,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one() {
        for spread in [1i64, 1 << 20, 1 << 32] {
            let scores: Vec<i64> = (0..8).map(|i| i as i64 * spread).collect();
            let got = route(&scores, 3);
            assert_eq!(got.len(), 3);
            let sum: i64 = got.iter().map(|&(_, w)| w).sum();
            assert!((sum - ONE).abs() < 16, "spread {spread}: summed to {sum}");
        }
    }

    /// The property consensus rests on: equal scores must resolve the same way everywhere.
    #[test]
    fn ties_resolve_to_the_lower_index() {
        let scores = vec![7i64; 6];
        let got = route(&scores, 3);
        assert_eq!(
            got.iter().map(|&(i, _)| i).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert!(got.iter().all(|&(_, w)| w == got[0].1), "{got:?}");
    }

    #[test]
    fn best_scoring_expert_comes_first() {
        let scores = vec![1i64 << 33, 9i64 << 33, 3i64 << 33];
        let got = route(&scores, 2);
        assert_eq!(got[0].0, 1);
        assert_eq!(got[1].0, 2);
        assert!(got[0].1 > got[1].1, "the better expert must weigh more");
    }

    #[test]
    fn output_round_trips() {
        let chosen = vec![(3u32, 1234i64), (0, -5), (9, i64::MAX)];
        assert_eq!(decode_output(&encode_output(&chosen)).unwrap(), chosen);
        assert!(decode_output(&[0u8; 13]).is_none());
    }

    /// Headers that ask for the impossible must be refused rather than clamped.
    #[test]
    fn malformed_headers_are_refused() {
        let at = Address::new([0x33; 20]);
        let x = vec![0i64; 4];
        // k larger than the expert count
        assert!(parse(&encode(2, 4, at, 0, 3, &x)).is_none());
        // no experts, no width, no choice
        assert!(parse(&encode(0, 4, at, 0, 1, &x)).is_none());
        assert!(parse(&encode(2, 4, at, 0, 0, &x)).is_none());
        // activation vector that does not match the stated width
        assert!(parse(&encode(2, 5, at, 0, 1, &x)).is_none());
        // and a well-formed one, so the test cannot pass by rejecting everything
        assert!(parse(&encode(2, 4, at, 0, 2, &x)).is_some());
    }
}
