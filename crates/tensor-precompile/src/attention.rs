//! The attention and RMSNorm precompiles.
//!
//! What was left after matmul, SwiGLU, and QK-norm went native: attention at 19% of a
//! layer and the two residual normalisations at 14%. Neither is expensive arithmetic.
//! Attention at one position is sixteen dot products of 128 values and a softmax over one
//! entry; RMSNorm is a square root and 1024 divides. They are expensive because the EVM
//! charges for the loop, not the work.
//!
//! # Attention takes the whole cache
//!
//! Rather than a score at a time. The cache is the argument that grows — `positions x
//! kvWidth` — and at long context it will dominate what crosses the boundary. That is a
//! real limit and it is the same one the companion project hit: keys and values are the
//! state that grows with generation, and compressing them is the thing to do before pushing
//! this to long sequences. Passing them per call rather than per score at least keeps the
//! count of crossings proportional to layers rather than to positions squared.

use revm::precompile::{
    u64_to_address, Precompile, PrecompileHalt, PrecompileId, PrecompileOutput, PrecompileResult,
};
use revm::primitives::{Address, Bytes};
use tensor_kernel::fixed::{exp_q32, isqrt, F, ONE};

use crate::{BASE_GAS, MAX_DIM};

/// Grouped-query attention over a cache.
pub const ATTENTION_ADDRESS: Address = u64_to_address(0x104);

/// RMSNorm over one vector.
pub const RMSNORM_ADDRESS: Address = u64_to_address(0x105);

/// Multiply-accumulates purchased by one unit of gas, matching the matmul's price: the
/// inner loop is the same operation.
pub const MACS_PER_GAS: u64 = 1024;

/// Elements purchased by one unit of gas in [`rmsnorm_precompile`].
pub const ELEMENTS_PER_GAS: u64 = 4;

/// ```text
/// [0..4)    heads      u32
/// [4..8)    nKv        u32
/// [8..12)   headDim    u32
/// [12..16)  positions  u32, the cache depth including this one
/// [16 ..)   q          i64 x heads*headDim
/// [.. ..)   k          i64 x positions*nKv*headDim
/// [.. ..)   v          i64 x positions*nKv*headDim
/// ```
///
/// Output is `heads * headDim` big-endian `i64` values.
const ATTN_HEADER: usize = 16;

/// ```text
/// [0..4)   n       u32
/// [4..12)  eps     i64, Q32
/// [12 ..)  weight  i64 x n
/// [.. ..)  x       i64 x n
/// ```
const NORM_HEADER: usize = 12;

pub fn attention_gas(heads: u64, head_dim: u64, positions: u64) -> Option<u64> {
    // Two passes over the cache per head: scores, then the weighted sum.
    let macs = heads
        .checked_mul(head_dim)?
        .checked_mul(positions)?
        .checked_mul(2)?;
    BASE_GAS.checked_add(macs.div_ceil(MACS_PER_GAS))
}

pub fn rmsnorm_gas(n: u64) -> Option<u64> {
    BASE_GAS.checked_add(n.div_ceil(ELEMENTS_PER_GAS))
}

fn bad_input(reservoir: u64) -> PrecompileResult {
    Ok(PrecompileOutput::revert(BASE_GAS, Bytes::new(), reservoir))
}

fn take(input: &[u8], at: usize, n: usize) -> Vec<i64> {
    input[at..at + n * 8]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect()
}

fn pack(v: &[i64]) -> Bytes {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v {
        b.extend_from_slice(&x.to_be_bytes());
    }
    Bytes::from(b)
}

pub fn attention_precompile(input: &[u8], gas_limit: u64, reservoir: u64) -> PrecompileResult {
    if input.len() < ATTN_HEADER {
        return bad_input(reservoir);
    }
    let heads = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
    let n_kv = u32::from_be_bytes(input[4..8].try_into().unwrap()) as u64;
    let hs = u32::from_be_bytes(input[8..12].try_into().unwrap()) as u64;
    let positions = u32::from_be_bytes(input[12..16].try_into().unwrap()) as u64;

    if heads == 0 || n_kv == 0 || hs == 0 || positions == 0 {
        return bad_input(reservoir);
    }
    if heads > MAX_DIM || n_kv > MAX_DIM || hs > MAX_DIM || positions > MAX_DIM {
        return bad_input(reservoir);
    }
    // Grouped-query attention shares each key head across an exact number of query heads.
    if !heads.is_multiple_of(n_kv) {
        return bad_input(reservoir);
    }

    let Some(gas_used) = attention_gas(heads, hs, positions) else {
        return bad_input(reservoir);
    };
    if gas_used > gas_limit {
        return Ok(PrecompileOutput::halt(PrecompileHalt::OutOfGas, reservoir));
    }

    let (heads, n_kv, hs, positions) = (
        heads as usize,
        n_kv as usize,
        hs as usize,
        positions as usize,
    );
    let kvw = n_kv * hs;
    let q_len = heads * hs;
    let Some(cache) = positions.checked_mul(kvw) else {
        return bad_input(reservoir);
    };
    let Some(want) = q_len
        .checked_add(cache)
        .and_then(|n| n.checked_add(cache))
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(ATTN_HEADER))
    else {
        return bad_input(reservoir);
    };
    if input.len() != want {
        return bad_input(reservoir);
    }

    let q = take(input, ATTN_HEADER, q_len);
    let k = take(input, ATTN_HEADER + q_len * 8, cache);
    let v = take(input, ATTN_HEADER + (q_len + cache) * 8, cache);

    // No range check, deliberately, and this is where the bound's purpose becomes clear.
    // `MAX_ABS_ACTIVATION` exists for one reason: the matmul kernel sums products in `i64`
    // within a chunk, and an unbounded operand would wrap. Everything here accumulates in
    // `i128` unconditionally, so no bound is needed — and imposing one would refuse the
    // real model, because queries and keys arrive after QK-norm and Qwen3's `k_norm` scales
    // them by up to 96.5.
    //
    // The invariant that actually matters is narrower than "activations are small": it is
    // that whatever reaches a *matmul* is bounded. That holds here, since attention returns
    // a convex combination of values — softmax weights sum to one — so its output cannot
    // exceed the values it mixes, which came from a projection.

    let out = attend(q, k, v, heads, n_kv, hs, positions);

    Ok(PrecompileOutput::new(gas_used, pack(&out), reservoir))
}

pub fn rmsnorm_precompile(input: &[u8], gas_limit: u64, reservoir: u64) -> PrecompileResult {
    if input.len() < NORM_HEADER {
        return bad_input(reservoir);
    }
    let n = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
    let eps = i64::from_be_bytes(input[4..12].try_into().unwrap());
    if n == 0 || n > MAX_DIM || eps < 0 {
        return bad_input(reservoir);
    }

    let Some(gas_used) = rmsnorm_gas(n) else {
        return bad_input(reservoir);
    };
    if gas_used > gas_limit {
        return Ok(PrecompileOutput::halt(PrecompileHalt::OutOfGas, reservoir));
    }

    let n = n as usize;
    let Some(want) = n.checked_mul(16).and_then(|b| b.checked_add(NORM_HEADER)) else {
        return bad_input(reservoir);
    };
    if input.len() != want {
        return bad_input(reservoir);
    }

    let weight = take(input, NORM_HEADER, n);
    let x = take(input, NORM_HEADER + n * 8, n);

    let mut ss: i128 = 0;
    for &v in &x {
        ss += (v as i128) * (v as i128); // Q64
    }
    let mean = ss / n as i128 + ((eps as i128) << F);
    let rms = {
        let r = isqrt(mean as u128) as i128;
        if r == 0 {
            1
        } else {
            r
        }
    };

    let out: Vec<i64> = (0..n)
        .map(|i| {
            let normed = ((x[i] as i128) << F) / rms; // Q32
            ((normed * weight[i] as i128) >> F) as i64
        })
        .collect();

    Ok(PrecompileOutput::new(gas_used, pack(&out), reservoir))
}

pub fn attention() -> Precompile {
    Precompile::new(
        PrecompileId::Custom("tensor-attention".into()),
        ATTENTION_ADDRESS,
        attention_precompile,
    )
}

pub fn rmsnorm() -> Precompile {
    Precompile::new(
        PrecompileId::Custom("tensor-rmsnorm".into()),
        RMSNORM_ADDRESS,
        rmsnorm_precompile,
    )
}

pub fn encode_attention(
    heads: usize,
    n_kv: usize,
    head_dim: usize,
    positions: usize,
    q: &[i64],
    k: &[i64],
    v: &[i64],
) -> Vec<u8> {
    let kvw = n_kv * head_dim;
    assert_eq!(q.len(), heads * head_dim, "q width");
    assert_eq!(k.len(), positions * kvw, "k cache");
    assert_eq!(v.len(), positions * kvw, "v cache");

    let mut b = Vec::with_capacity(ATTN_HEADER + (q.len() + k.len() + v.len()) * 8);
    for f in [heads, n_kv, head_dim, positions] {
        b.extend_from_slice(&(f as u32).to_be_bytes());
    }
    for slice in [q, k, v] {
        for x in slice {
            b.extend_from_slice(&x.to_be_bytes());
        }
    }
    b
}

pub fn encode_rmsnorm(eps: i64, weight: &[i64], x: &[i64]) -> Vec<u8> {
    assert_eq!(weight.len(), x.len(), "weight width");
    let mut b = Vec::with_capacity(NORM_HEADER + x.len() * 16);
    b.extend_from_slice(&(x.len() as u32).to_be_bytes());
    b.extend_from_slice(&eps.to_be_bytes());
    for slice in [weight, x] {
        for v in slice {
            b.extend_from_slice(&v.to_be_bytes());
        }
    }
    b
}

/// Grouped-query attention over a cache, returning `heads * head_dim` values.
///
/// Lifted out of the precompile so the fused attention-and-projection form runs this and
/// not a copy of it. Two implementations of attention would be two opinions about what a
/// layer computes.
pub fn attend(
    q: Vec<i64>,
    k: Vec<i64>,
    v: Vec<i64>,
    heads: usize,
    n_kv: usize,
    hs: usize,
    positions: usize,
) -> Vec<i64> {
    let kvw = n_kv * hs;
    let q_len = heads * hs;
    // 1/sqrt(headDim) in Q32, by integer square root — no float provenance, and identical
    // to what the Solidity and the Rust reference derive.
    let scale = {
        let root = isqrt((hs as u128) << 64);
        (((ONE as u128) << 32) / root) as i128
    };

    let group = heads / n_kv;
    let mut out = vec![0i64; q_len];
    let mut scores = vec![0i64; positions];

    for h in 0..heads {
        let kvh = h / group;
        let qh = &q[h * hs..(h + 1) * hs];

        for (t, s) in scores.iter_mut().enumerate() {
            let base = t * kvw + kvh * hs;
            let mut acc: i128 = 0;
            for j in 0..hs {
                acc += (qh[j] as i128) * (k[base + j] as i128); // Q64
            }
            *s = (((acc >> F) * scale) >> F) as i64;
        }

        // Softmax. The maximum is subtracted first because the exponential is defined only
        // for non-positive arguments, so this is what makes every call legal.
        let max = *scores.iter().max().expect("positions > 0");
        let mut sum: i128 = 0;
        let mut e = vec![0u64; positions];
        for t in 0..positions {
            e[t] = exp_q32(scores[t] - max);
            sum += e[t] as i128;
        }
        if sum == 0 {
            // Only reachable if the spread exceeds what Q32 represents. Uniform is the
            // honest answer; leaving zeros would silently drop the whole row.
            let u = ONE / positions as i64;
            scores.iter_mut().for_each(|s| *s = u);
        } else {
            for t in 0..positions {
                scores[t] = (((e[t] as i128) << F) / sum) as i64;
            }
        }

        let dst = &mut out[h * hs..(h + 1) * hs];
        for (t, &w) in scores.iter().enumerate() {
            let base = t * kvw + kvh * hs;
            let w = w as i128;
            for (o, vv) in dst.iter_mut().zip(&v[base..base + hs]) {
                *o += ((w * *vv as i128) >> F) as i64;
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_model::ops::rmsnorm as ref_rmsnorm;

    const PLENTY: u64 = 1_000_000_000;
    const EPS: i64 = 4_295;

    fn noise(n: usize, seed: i64) -> Vec<i64> {
        (0..n)
            .map(|i| ((i as i64 * 40503 + seed) % 262_144) - 131_072)
            .collect()
    }

    #[test]
    fn rmsnorm_matches_the_reference() {
        let n = 129; // deliberately odd and not a lane multiple
        let w = noise(n, 5);
        let x = noise(n, 9);
        let mut want = vec![0i64; n];
        ref_rmsnorm(&x, &w, EPS, &mut want);

        let out = rmsnorm_precompile(&encode_rmsnorm(EPS, &w, &x), PLENTY, 0).unwrap();
        assert_eq!(crate::decode(&out.bytes).unwrap(), want);
    }

    /// Against the same attention the Rust model performs, through a real cache.
    #[test]
    fn attention_matches_the_reference() {
        let (heads, n_kv, hs, positions) = (4usize, 2usize, 8usize, 3usize);
        let kvw = n_kv * hs;
        let q = noise(heads * hs, 1);
        let k = noise(positions * kvw, 2);
        let v = noise(positions * kvw, 3);

        // Recomputed here rather than shared, so this checks the specification rather than
        // one implementation of it.
        let scale = {
            let root = isqrt((hs as u128) << 64);
            (((ONE as u128) << 32) / root) as i128
        };
        let group = heads / n_kv;
        let mut want = vec![0i64; heads * hs];
        for h in 0..heads {
            let kvh = h / group;
            let mut sc = vec![0i64; positions];
            for (t, s) in sc.iter_mut().enumerate() {
                let base = t * kvw + kvh * hs;
                let mut acc: i128 = 0;
                for j in 0..hs {
                    acc += (q[h * hs + j] as i128) * (k[base + j] as i128);
                }
                *s = (((acc >> F) * scale) >> F) as i64;
            }
            let max = *sc.iter().max().unwrap();
            let e: Vec<u64> = sc.iter().map(|s| exp_q32(s - max)).collect();
            let sum: i128 = e.iter().map(|x| *x as i128).sum();
            for (t, ee) in e.iter().enumerate() {
                let w = ((*ee as i128) << F) / sum;
                let base = t * kvw + kvh * hs;
                for j in 0..hs {
                    want[h * hs + j] += ((w * v[base + j] as i128) >> F) as i64;
                }
            }
        }

        let input = encode_attention(heads, n_kv, hs, positions, &q, &k, &v);
        let out = attention_precompile(&input, PLENTY, 0).unwrap();
        assert_eq!(crate::decode(&out.bytes).unwrap(), want);
    }

    /// Query heads sharing a key head must see the same keys, and heads in different
    /// groups must not. A mistake here degrades output without erroring.
    #[test]
    fn grouped_queries_share_the_right_heads() {
        let (heads, n_kv, hs, positions) = (4usize, 2usize, 4usize, 1usize);
        let kvw = n_kv * hs;
        // Identical queries across all heads, so any difference in output comes from which
        // key head each was paired with.
        let q: Vec<i64> = std::iter::repeat_n(1i64 << 30, heads * hs).collect();
        let mut k = vec![0i64; positions * kvw];
        let mut v = vec![0i64; positions * kvw];
        for j in 0..hs {
            k[j] = 1 << 30;
            v[j] = 1 << 31; // kv head 0
            k[hs + j] = 1 << 29;
            v[hs + j] = 1 << 28; // kv head 1
        }

        let out = attention_precompile(
            &encode_attention(heads, n_kv, hs, positions, &q, &k, &v),
            PLENTY,
            0,
        )
        .unwrap();
        let got = crate::decode(&out.bytes).unwrap();

        assert_eq!(got[..hs], got[hs..2 * hs], "heads 0,1 share kv head 0");
        assert_eq!(
            got[2 * hs..3 * hs],
            got[3 * hs..],
            "heads 2,3 share kv head 1"
        );
        assert_ne!(got[..hs], got[2 * hs..3 * hs], "groups must differ");
    }

    #[test]
    fn malformed_input_never_panics() {
        let good = encode_attention(2, 1, 4, 2, &noise(8, 1), &noise(8, 2), &noise(8, 3));
        let hdr = |h: u32, kv: u32, d: u32, p: u32| {
            let mut v = good.clone();
            v[0..4].copy_from_slice(&h.to_be_bytes());
            v[4..8].copy_from_slice(&kv.to_be_bytes());
            v[8..12].copy_from_slice(&d.to_be_bytes());
            v[12..16].copy_from_slice(&p.to_be_bytes());
            v
        };
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", vec![]),
            ("truncated header", vec![0u8; ATTN_HEADER - 1]),
            ("zero heads", hdr(0, 1, 4, 2)),
            ("zero kv", hdr(2, 0, 4, 2)),
            ("zero positions", hdr(2, 1, 4, 0)),
            ("heads not a multiple of kv", hdr(3, 2, 4, 2)),
            ("oversized", hdr(u32::MAX, 1, 4, 2)),
            ("truncated body", good[..good.len() - 1].to_vec()),
            ("trailing bytes", [good.clone(), vec![0]].concat()),
        ];
        for (name, input) in cases {
            let out = attention_precompile(&input, PLENTY, 0)
                .unwrap_or_else(|e| panic!("{name}: fatal {e:?}"));
            assert!(
                out.status.is_revert_or_halt(),
                "{name}: expected failure, got {:?}",
                out.status
            );
        }

        let good = encode_rmsnorm(EPS, &noise(4, 1), &noise(4, 2));
        for (name, input) in [
            ("empty", vec![]),
            ("header only", vec![0u8; NORM_HEADER]),
            ("truncated body", good[..good.len() - 1].to_vec()),
            ("trailing bytes", [good.clone(), vec![0]].concat()),
        ] {
            let out = rmsnorm_precompile(&input, PLENTY, 0)
                .unwrap_or_else(|e| panic!("rmsnorm {name}: fatal {e:?}"));
            assert!(out.status.is_revert_or_halt(), "rmsnorm {name}");
        }
    }
}
