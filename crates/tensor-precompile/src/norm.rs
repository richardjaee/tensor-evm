//! The QK-norm and rotary precompile.
//!
//! # Why this one
//!
//! Because measuring the layer said so. With matmul and SwiGLU already native, a real
//! Qwen3-0.6B block spends **43%** of its gas normalising and rotating query and key heads
//! — 4.37M on the query side alone, against 432k for the query projection it follows. The
//! operation is trivial arithmetic on 128 values; what makes it expensive is doing it
//! sixteen times through a Solidity loop that allocates an array per head.
//!
//! This is the third time the same pattern has appeared. Accelerate the largest cost and
//! the next one is revealed to be larger than expected; the layer went from 99% matmul, to
//! 99% SwiGLU, to 43% this. The lesson is not that any particular operation is special. It
//! is that a chain running transformers needs the whole tensor vocabulary as primitives,
//! and picking them off one measurement at a time is how you find out which.
//!
//! # Why the two are fused
//!
//! RMSNorm and rotary are separate operations, and separating them here would cost a second
//! call per head plus a round trip for the intermediate. They are also always adjacent and
//! always in this order — normalising a rotated vector rescales its components unequally
//! and destroys the angle the embedding encodes — so the fusion cannot be wrong for a
//! caller that wanted them apart.

use revm::precompile::{
    u64_to_address, Precompile, PrecompileHalt, PrecompileId, PrecompileOutput, PrecompileResult,
};
use revm::primitives::{Address, Bytes};
use tensor_kernel::activations_in_range;
use tensor_kernel::fixed::{isqrt, F};

use crate::{BASE_GAS, MAX_DIM};

/// Address of the fused QK-norm and rotary precompile.
pub const NORM_ROPE_ADDRESS: Address = u64_to_address(0x103);

/// Elements purchased by one unit of gas.
///
/// The work per element is a multiply, a shift, and a share of one square root — cheaper
/// than SwiGLU's exponential, so this is priced above it rather than below. Being generous
/// is the safe direction.
pub const ELEMENTS_PER_GAS: u64 = 4;

/// ```text
/// [0..4)    heads     u32
/// [4..8)    headDim   u32
/// [8..16)   eps       i64, Q32
/// [16 ..)   weight    i64 x headDim
/// [.. ..)   cos       i64 x headDim/2
/// [.. ..)   sin       i64 x headDim/2
/// [.. ..)   vec       i64 x heads*headDim
/// ```
///
/// Output is `heads * headDim` big-endian `i64` values.
const HEADER: usize = 16;

pub fn gas_cost(heads: u64, head_dim: u64) -> Option<u64> {
    let n = heads.checked_mul(head_dim)?;
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

pub fn norm_rope_precompile(input: &[u8], gas_limit: u64, reservoir: u64) -> PrecompileResult {
    if input.len() < HEADER {
        return bad_input(reservoir);
    }
    let heads = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
    let hs = u32::from_be_bytes(input[4..8].try_into().unwrap()) as u64;
    let eps = i64::from_be_bytes(input[8..16].try_into().unwrap());

    // Rotary pairs elements `j` with `j + hs/2`, so an odd width has no meaning here.
    if heads == 0 || hs == 0 || !hs.is_multiple_of(2) || heads > MAX_DIM || hs > MAX_DIM {
        return bad_input(reservoir);
    }
    if eps < 0 {
        return bad_input(reservoir);
    }

    let Some(gas_used) = gas_cost(heads, hs) else {
        return bad_input(reservoir);
    };
    if gas_used > gas_limit {
        return Ok(PrecompileOutput::halt(PrecompileHalt::OutOfGas, reservoir));
    }

    let (heads, hs) = (heads as usize, hs as usize);
    let half = hs / 2;
    let Some(total) = heads.checked_mul(hs) else {
        return bad_input(reservoir);
    };
    let Some(want) = hs
        .checked_add(half * 2)
        .and_then(|n| n.checked_add(total))
        .and_then(|n| n.checked_mul(8))
        .and_then(|n| n.checked_add(HEADER))
    else {
        return bad_input(reservoir);
    };
    if input.len() != want {
        return bad_input(reservoir);
    }

    let mut at = HEADER;
    let weight = take(input, at, hs);
    at += hs * 8;
    let cos = take(input, at, half);
    at += half * 8;
    let sin = take(input, at, half);
    at += half * 8;
    let mut vec = take(input, at, total);

    // The input to this operation comes from a projection and is genuinely bounded, so the
    // check is kept as a cheap guard against nonsense. The weight is not checked: it is a
    // scale factor rather than an activation, and Qwen3's `k_norm` reaches 96.5 — six times
    // the activation bound — so checking it as one would refuse the real model.
    //
    // Nothing here overflows either way, since every product is formed in `i128`. The bound
    // exists for the matmul kernel alone, which sums in `i64` within a chunk. That is also
    // why this precompile's *output* is not bounded and need not be: it feeds attention,
    // not a matmul.
    if !activations_in_range(&vec) {
        return bad_input(reservoir);
    }

    for h in 0..heads {
        let seg = &mut vec[h * hs..(h + 1) * hs];

        // RMSNorm reads every element before writing any, so the normalised values are
        // formed first and the rotation reads them from a scratch buffer.
        let mut ss: i128 = 0;
        for &v in seg.iter() {
            ss += (v as i128) * (v as i128); // Q64
        }
        let mean = ss / hs as i128 + ((eps as i128) << F);
        let rms = {
            let r = isqrt(mean as u128) as i128;
            if r == 0 {
                1
            } else {
                r
            }
        };

        let mut normed = vec![0i64; hs];
        for j in 0..hs {
            let n = ((seg[j] as i128) << F) / rms; // Q32
            normed[j] = ((n * weight[j] as i128) >> F) as i64;
        }

        for j in 0..half {
            let a = normed[j] as i128;
            let b = normed[j + half] as i128;
            let c = cos[j] as i128;
            let s = sin[j] as i128;
            seg[j] = ((a * c - b * s) >> F) as i64;
            seg[j + half] = ((a * s + b * c) >> F) as i64;
        }
    }

    let mut bytes = Vec::with_capacity(total * 8);
    for v in vec {
        bytes.extend_from_slice(&v.to_be_bytes());
    }
    Ok(PrecompileOutput::new(
        gas_used,
        Bytes::from(bytes),
        reservoir,
    ))
}

pub fn precompile() -> Precompile {
    Precompile::new(
        PrecompileId::Custom("tensor-norm-rope".into()),
        NORM_ROPE_ADDRESS,
        norm_rope_precompile,
    )
}

/// Encodes a call.
pub fn encode(
    heads: usize,
    head_dim: usize,
    eps: i64,
    weight: &[i64],
    cos: &[i64],
    sin: &[i64],
    vec: &[i64],
) -> Vec<u8> {
    assert_eq!(weight.len(), head_dim, "weight must be headDim");
    assert_eq!(cos.len(), head_dim / 2, "cos must be headDim/2");
    assert_eq!(sin.len(), head_dim / 2, "sin must be headDim/2");
    assert_eq!(vec.len(), heads * head_dim, "vec must be heads*headDim");

    let mut v = Vec::with_capacity(HEADER + (head_dim + head_dim + heads * head_dim) * 8);
    v.extend_from_slice(&(heads as u32).to_be_bytes());
    v.extend_from_slice(&(head_dim as u32).to_be_bytes());
    v.extend_from_slice(&eps.to_be_bytes());
    for slice in [weight, cos, sin, vec] {
        for x in slice {
            v.extend_from_slice(&x.to_be_bytes());
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_kernel::fixed::ONE;
    use tensor_model::ops::{rmsnorm, rope};

    const PLENTY: u64 = 1_000_000_000;
    const EPS: i64 = 4_295;

    fn noise(n: usize, seed: i64) -> Vec<i64> {
        (0..n)
            .map(|i| ((i as i64 * 40503 + seed) % 262_144) - 131_072)
            .collect()
    }

    /// Norm weights are scale factors rather than activations, and real ones are large:
    /// Qwen3's `k_norm` peaks at 96.5, well outside the bound that applies to activations.
    /// Refusing those would refuse the model.
    #[test]
    fn accepts_norm_weights_beyond_the_activation_bound() {
        let (heads, hs) = (2usize, 8usize);
        let half = hs / 2;
        // 96.5 in Q32, as k_norm actually contains.
        let big = (96.5 * (1u64 << 32) as f64) as i64;
        assert!(big > tensor_kernel::MAX_ABS_ACTIVATION);

        let weight = vec![big; hs];
        let out = norm_rope_precompile(
            &encode(
                heads,
                hs,
                EPS,
                &weight,
                &noise(half, 1),
                &noise(half, 2),
                &noise(heads * hs, 3),
            ),
            PLENTY,
            0,
        )
        .unwrap();
        assert!(out.status.is_success(), "refused a real norm weight");
    }

    /// Against the same operations composed separately, which is what the Solidity did
    /// before this existed.
    #[test]
    fn matches_rmsnorm_then_rope() {
        let (heads, hs) = (4usize, 16usize);
        let half = hs / 2;
        let weight = noise(hs, 7);
        let cos = noise(half, 11);
        let sin = noise(half, 13);
        let vec = noise(heads * hs, 17);

        let mut want = vec.clone();
        for h in 0..heads {
            let seg = &mut want[h * hs..(h + 1) * hs];
            let mut tmp = vec![0i64; hs];
            rmsnorm(seg, &weight, EPS, &mut tmp);
            seg.copy_from_slice(&tmp);
            rope(seg, &cos, &sin);
        }

        let input = encode(heads, hs, EPS, &weight, &cos, &sin, &vec);
        let out = norm_rope_precompile(&input, PLENTY, 0).expect("no error");
        assert_eq!(crate::decode(&out.bytes).unwrap(), want);
    }

    /// Heads must be independent. Changing one must leave the others untouched, or the
    /// fusion is silently mixing them.
    #[test]
    fn heads_do_not_interfere() {
        let (heads, hs) = (3usize, 8usize);
        let half = hs / 2;
        let weight = vec![ONE; hs];
        let cos = vec![ONE; half];
        let sin = vec![0i64; half];
        let mut vec = noise(heads * hs, 3);

        let a = norm_rope_precompile(
            &encode(heads, hs, EPS, &weight, &cos, &sin, &vec),
            PLENTY,
            0,
        )
        .unwrap();
        let first = crate::decode(&a.bytes).unwrap();

        vec[2 * hs] += 1 << 30; // perturb the third head only
        let b = norm_rope_precompile(
            &encode(heads, hs, EPS, &weight, &cos, &sin, &vec),
            PLENTY,
            0,
        )
        .unwrap();
        let second = crate::decode(&b.bytes).unwrap();

        assert_eq!(first[..2 * hs], second[..2 * hs], "untouched heads changed");
        assert_ne!(first[2 * hs..], second[2 * hs..], "perturbed head did not");
    }

    #[test]
    fn malformed_input_never_panics() {
        let (heads, hs) = (2usize, 8usize);
        let good = encode(
            heads,
            hs,
            EPS,
            &noise(hs, 1),
            &noise(hs / 2, 2),
            &noise(hs / 2, 3),
            &noise(heads * hs, 4),
        );
        let with = |h: u32, d: u32| {
            let mut v = good.clone();
            v[0..4].copy_from_slice(&h.to_be_bytes());
            v[4..8].copy_from_slice(&d.to_be_bytes());
            v
        };
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", vec![]),
            ("truncated header", vec![0u8; HEADER - 1]),
            ("header only", vec![0u8; HEADER]),
            ("zero heads", with(0, hs as u32)),
            ("zero headDim", with(heads as u32, 0)),
            ("odd headDim", with(heads as u32, 7)),
            ("oversized heads", with(u32::MAX, hs as u32)),
            ("oversized headDim", with(heads as u32, u32::MAX)),
            ("truncated body", good[..good.len() - 1].to_vec()),
            ("trailing bytes", [good.clone(), vec![0]].concat()),
        ];
        for (name, input) in cases {
            let out = norm_rope_precompile(&input, PLENTY, 0)
                .unwrap_or_else(|e| panic!("{name}: fatal error {e:?}"));
            assert!(
                out.status.is_revert_or_halt(),
                "{name}: expected failure, got {:?}",
                out.status
            );
        }
    }

    #[test]
    fn price_scales_with_the_work() {
        let small = gas_cost(8, 128).unwrap() - BASE_GAS;
        let large = gas_cost(16, 128).unwrap() - BASE_GAS;
        assert_eq!(large, small * 2);
    }
}
