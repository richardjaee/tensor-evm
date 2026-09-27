//! Emit routing vectors for the companion project's Solidity to match.
//!
//! The mixture's arithmetic has to be reproducible like everything else here, but its
//! *selection* has to be reproducible too, and that is the sharper requirement: a tie
//! broken differently does not round differently, it sends the token through different
//! parameters. Two honest nodes then emit different text.
//!
//! So the cases below are chosen for their ties rather than for their coverage — every
//! score equal, a tie at the top, scores far enough apart that every exponent underflows.
//! A comparator that merely looks right passes on random input and fails on these.
//!
//! ```text
//! cargo run --release --bin moevec > ../on-chain-llm/artifacts/moe_vectors.json
//! ```

use tensor_kernel::fixed::{exp_q32, top_k, F, ONE};

/// Softmax over the chosen scores alone, in Q32 — `moe.rs`'s `route_weights`, copied
/// rather than imported because it is private there and this must match what ships.
fn route_weights(chosen: &[(usize, i64)], out: &mut [i64]) {
    let max = chosen.iter().map(|&(_, v)| v).max().unwrap_or(0);
    let mut sum: u128 = 0;
    for (i, &(_, v)) in chosen.iter().enumerate() {
        let e = exp_q32(v - max) as u128;
        out[i] = e as i64;
        sum += e;
    }
    if sum == 0 {
        out[..chosen.len()].fill(0);
        if !chosen.is_empty() {
            out[0] = ONE;
        }
        return;
    }
    for slot in out.iter_mut().take(chosen.len()) {
        *slot = (((*slot as u128) << F) / sum) as i64;
    }
}

fn main() {
    let cases: Vec<Vec<i64>> = vec![
        (0..8).map(|i| ((i as i64 * 2654435761) % 4001) - 2000).collect(),
        vec![5, 5, 5, 5, 1, 2, 3, 4],
        vec![-(1 << 40), 1 << 40, 0, 7, 7, -3, 1 << 35, 2],
        (0..16).map(|i| ((i as i64 * 7919 + 13) % 997) - 500).collect(),
        vec![0; 12],
        vec![1 << 50, -(1 << 50), 0, 0, 0, 0],
    ];

    // Shaped for the companion project's test helpers: a top-level object, signed values
    // as strings. `stdJson` reads unsigned arrays directly but has no signed equivalent,
    // so the existing vectors there quote their integers and parse them back.
    let mut rows: Vec<String> = Vec::new();
    let mut n = 0usize;
    for (_case, v) in cases.iter().enumerate() {
        for &k in &[1usize, 2, 4] {
            if k > v.len() {
                continue;
            }
            let mut chosen = Vec::new();
            top_k(v, k, &mut chosen);
            let mut w = vec![0i64; k];
            route_weights(&chosen, &mut w);
            let mut order: Vec<usize> = (0..chosen.len()).collect();
            order.sort_by_key(|&i| chosen[i].0);

            let nums = |xs: Vec<String>| xs.join(",");
            let strs = |xs: Vec<String>| {
                xs.iter()
                    .map(|x| format!("\"{x}\""))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            rows.push(format!(
                "  \"c{}\": {{\"k\":{},\"scores\":[{}],\"idx\":[{}],\"val\":[{}],\"w\":[{}],\"order\":[{}]}}",
                n,
                k,
                strs(v.iter().map(|x| x.to_string()).collect()),
                nums(chosen.iter().map(|c| c.0.to_string()).collect()),
                strs(chosen.iter().map(|c| c.1.to_string()).collect()),
                strs(w.iter().map(|x| x.to_string()).collect()),
                nums(order.iter().map(|x| x.to_string()).collect()),
            ));
            n += 1;
        }
    }
    println!("{{\n  \"n\": {},\n{}\n}}", n, rows.join(",\n"));
}
