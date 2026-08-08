//! The SwiGLU precompile.
//!
//! # Why this exists, when the design said one narrow precompile
//!
//! Because the measurement disagreed. In the EVM a single SiLU costs 6,235 gas, so a
//! Qwen3-0.6B layer's 3,072 of them come to 19.1M — against 15,360 for the same layer's
//! matrix multiplication once that is a precompile. Accelerating only matmul does not
//! remove the bottleneck, it relocates it: a layer goes from 99% matmul to 99% SwiGLU, and
//! the thing left behind is **1,247x** the thing that was fixed.
//!
//! The original instinct was right about interface *shape* and wrong about *sufficiency*.
//! A precompile that ran a whole transformer block would need revising for every
//! architecture variant, and that objection still stands. But `matmul` alone is not enough
//! to make a chain fast, and no amount of narrowing changes that.
//!
//! # Where the line is drawn
//!
//! Vectorised, one call per layer rather than one per element. At 3,072 elements the
//! calling overhead would otherwise be the whole cost of what it replaced.
//!
//! SwiGLU rather than the more general `exp`, and that is a real trade. `exp` is universal
//! — SiLU, softmax, and GELU-through-tanh all reduce to it — but leaving the surrounding
//! arithmetic in the EVM keeps a division and a multiply per element, which measurement
//! says is still an order of magnitude above the matmul it sits beside. SwiGLU covers the
//! Llama, Qwen, and Mistral families as they are. GeGLU models would need a sibling, and
//! adding one is additive rather than a revision — which is the property the narrow-
//! interface argument was actually protecting.

use revm::precompile::{
    u64_to_address, Precompile, PrecompileId, PrecompileOutput, PrecompileResult,
};
use revm::primitives::{Address, Bytes};
use tensor_kernel::swiglu_vec;

use crate::{BASE_GAS, MAX_DIM};

/// Address of the SwiGLU precompile.
pub const SWIGLU_ADDRESS: Address = u64_to_address(0x102);

/// Elements purchased by one unit of gas.
///
/// Derived the same way the matmul price is. One SiLU takes about 40ns natively, so a
/// second buys roughly 25 million; against Ethereum's implied 3M gas per second of node
/// work that is around 8 elements per gas. This sets it at 4 — priced above cost, which is
/// the safe direction, and still leaves a layer's SwiGLU at 768 gas against 19.1M in the
/// EVM.
pub const ELEMENTS_PER_GAS: u64 = 4;

/// ```text
/// [0..4)   n      u32, elements in each vector
/// [4 ..)   gate   i64 x n
/// [.. ..)  up     i64 x n
/// ```
///
/// Output is `n` big-endian `i64` values: `silu(gate) * up`.
const HEADER: usize = 4;

pub fn gas_cost(n: u64) -> Option<u64> {
    BASE_GAS.checked_add(n.div_ceil(ELEMENTS_PER_GAS))
}

fn bad_input(reservoir: u64) -> PrecompileResult {
    Ok(PrecompileOutput::revert(BASE_GAS, Bytes::new(), reservoir))
}

pub fn swiglu_precompile(input: &[u8], gas_limit: u64, reservoir: u64) -> PrecompileResult {
    if input.len() < HEADER {
        return bad_input(reservoir);
    }
    let n = u32::from_be_bytes(input[0..4].try_into().unwrap()) as u64;
    if n == 0 || n > MAX_DIM {
        return bad_input(reservoir);
    }

    let Some(gas_used) = gas_cost(n) else {
        return bad_input(reservoir);
    };
    if gas_used > gas_limit {
        return Ok(PrecompileOutput::halt(
            revm::precompile::PrecompileHalt::OutOfGas,
            reservoir,
        ));
    }

    let n = n as usize;
    let Some(body) = n.checked_mul(16) else {
        return bad_input(reservoir);
    };
    // Exact: trailing bytes mean the caller and the precompile disagree about the layout.
    if input.len() != HEADER + body {
        return bad_input(reservoir);
    }

    let split = HEADER + n * 8;
    let mut gate: Vec<i64> = input[HEADER..split]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();
    let up: Vec<i64> = input[split..]
        .chunks_exact(8)
        .map(|c| i64::from_be_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect();

    swiglu_vec(&mut gate, &up);

    let mut bytes = Vec::with_capacity(n * 8);
    for v in gate {
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
        PrecompileId::Custom("tensor-swiglu".into()),
        SWIGLU_ADDRESS,
        swiglu_precompile,
    )
}

/// Encodes a call.
pub fn encode(gate: &[i64], up: &[i64]) -> Vec<u8> {
    assert_eq!(gate.len(), up.len(), "gate and up must match");
    let mut v = Vec::with_capacity(HEADER + gate.len() * 16);
    v.extend_from_slice(&(gate.len() as u32).to_be_bytes());
    for g in gate {
        v.extend_from_slice(&g.to_be_bytes());
    }
    for u in up {
        v.extend_from_slice(&u.to_be_bytes());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_kernel::fixed::{silu, F, ONE};

    const PLENTY: u64 = 1_000_000_000;

    fn vecs(n: usize) -> (Vec<i64>, Vec<i64>) {
        let g = (0..n)
            .map(|i| ((i as i64 * 40503 + 17) % 262_144) - 131_072)
            .collect();
        let u = (0..n)
            .map(|i| ((i as i64 * 2654435 + 7) % 262_144) - 131_072)
            .collect();
        (g, u)
    }

    #[test]
    fn matches_the_kernel() {
        let (g, u) = vecs(257);
        let mut want = g.clone();
        for i in 0..want.len() {
            want[i] = ((silu(g[i]) as i128 * u[i] as i128) >> F) as i64;
        }

        let out = swiglu_precompile(&encode(&g, &u), PLENTY, 0).expect("no error");
        let got = crate::decode(&out.bytes).expect("decodes");
        assert_eq!(got, want);
    }

    /// The property SiLU exists for: it is not monotonic, and a sign error would keep the
    /// magnitudes plausible while destroying the shape.
    #[test]
    fn reproduces_silus_dip() {
        let at = |z: f64| {
            let q = (z * ONE as f64) as i64;
            let out = swiglu_precompile(&encode(&[q], &[ONE]), PLENTY, 0).unwrap();
            crate::decode(&out.bytes).unwrap()[0] as f64 / ONE as f64
        };
        assert!(at(-1.278) < at(-3.0));
        assert!(at(-1.278) < at(0.0));
    }

    #[test]
    fn charges_for_the_work_and_scales_with_it() {
        let small = gas_cost(1024).unwrap() - BASE_GAS;
        let large = gas_cost(4096).unwrap() - BASE_GAS;
        assert_eq!(large, small * 4);
    }

    #[test]
    fn malformed_input_never_panics() {
        let (g, u) = vecs(4);
        let good = encode(&g, &u);
        let with_n = |n: u32| {
            let mut v = good.clone();
            v[0..4].copy_from_slice(&n.to_be_bytes());
            v
        };
        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", vec![]),
            ("header only", vec![0u8; HEADER]),
            ("truncated header", vec![0u8; HEADER - 1]),
            ("zero length", with_n(0)),
            ("oversized", with_n(u32::MAX)),
            ("truncated body", good[..good.len() - 1].to_vec()),
            ("trailing bytes", [good.clone(), vec![0]].concat()),
            ("length disagrees with body", with_n(8)),
        ];
        for (name, input) in cases {
            let out = swiglu_precompile(&input, PLENTY, 0)
                .unwrap_or_else(|e| panic!("{name}: fatal error {e:?}"));
            assert!(
                out.status.is_revert_or_halt(),
                "{name}: expected failure, got {:?}",
                out.status
            );
        }
    }

    /// Large operands must be handled rather than refused. The residual stream of a real
    /// model reaches magnitudes that every earlier bound in this project rejected, and the
    /// pairing forms its product in `i128`, so there is nothing left to refuse them for.
    #[test]
    fn accepts_operands_of_any_magnitude() {
        let big = 1i64 << 50;
        let out = swiglu_precompile(&encode(&[big], &[ONE]), PLENTY, 0).unwrap();
        assert!(
            out.status.is_success(),
            "refused a magnitude a real model produces"
        );
    }
}
