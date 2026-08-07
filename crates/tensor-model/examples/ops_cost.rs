//! Where a layer's time actually goes.
//!
//! The design so far rests on one narrow precompile: matrix multiplication, and nothing
//! else. That is the right shape for an interface — it serves every architecture and never
//! needs revising — but it is only the right shape for a *chain* if the matmuls are what
//! the work consists of.
//!
//! Measuring a real Qwen3-0.6B layer said otherwise: about two milliseconds per position,
//! against roughly nine hundred microseconds of matmul at the throughput those shapes
//! achieve. This attributes the rest.
//!
//! Run with `cargo run --release --example ops_cost`.

use std::time::Instant;
use tensor_kernel::matmul;
use tensor_model::fixed::silu;
use tensor_model::ops::{rmsnorm, rope, scaled_dot, softmax, swiglu};

/// Qwen3-0.6B.
const DIM: usize = 1024;
const HIDDEN: usize = 3072;
const HEADS: usize = 16;
const KV: usize = 8;
const HS: usize = 128;

fn noise(n: usize, seed: u64) -> Vec<i64> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as i64) - (1 << 23)
        })
        .collect()
}

/// Repeats until the measurement is well clear of timer noise, then reports per-call cost.
fn time<F: FnMut()>(mut f: F) -> f64 {
    f();
    let start = Instant::now();
    let mut n = 0u32;
    loop {
        f();
        n += 1;
        let e = start.elapsed().as_secs_f64();
        if e >= 0.25 {
            return e / n as f64;
        }
    }
}

fn main() {
    let x = noise(DIM, 1);
    let w = noise(DIM, 2);
    let mut out = vec![0i64; DIM];

    let hidden_a = noise(HIDDEN, 3);
    let hidden_b = noise(HIDDEN, 4);
    let mut hidden_out = vec![0i64; HIDDEN];

    let head = noise(HS, 5);
    let cos = noise(HS / 2, 6);
    let sin = noise(HS / 2, 7);

    println!("one Qwen3-0.6B layer, per position\n");
    println!("{:<34} {:>12} {:>10}", "operation", "microseconds", "count");
    println!("{}", "-".repeat(58));

    let mut total = 0.0;
    let mut report = |name: &str, per_call: f64, count: usize| {
        let us = per_call * 1e6 * count as f64;
        total += us;
        println!("{name:<34} {us:>12.1} {count:>10}");
        us
    };

    // ---- matmuls ----
    let mut matmul_us = 0.0;
    for (name, rows, cols) in [
        ("matmul q_proj  2048x1024", 2048usize, 1024usize),
        ("matmul k_proj  1024x1024", 1024, 1024),
        ("matmul v_proj  1024x1024", 1024, 1024),
        ("matmul o_proj  1024x2048", 1024, 2048),
        ("matmul gate    3072x1024", 3072, 1024),
        ("matmul up      3072x1024", 3072, 1024),
        ("matmul down    1024x3072", 1024, 3072),
    ] {
        let weights: Vec<i16> = (0..rows * cols)
            .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
            .collect();
        let scales = vec![20i8; rows];
        let acts = noise(cols, 11);
        let mut o = vec![0i64; rows];
        let t = time(|| matmul(&weights, &scales, &acts, rows, cols, &mut o));
        matmul_us += report(name, t, 1);
    }

    // ---- everything else ----
    let mut other_us = 0.0;
    other_us += report("rmsnorm (residual)", time(|| rmsnorm(&x, &w, &mut out)), 2);
    other_us += report(
        "rmsnorm (QK-norm, per head)",
        time(|| {
            let mut o = vec![0i64; HS];
            rmsnorm(&head, &cos.repeat(2), &mut o)
        }),
        HEADS + KV,
    );
    other_us += report(
        "rope (per head)",
        time(|| {
            let mut v = head.clone();
            rope(&mut v, &cos, &sin)
        }),
        HEADS + KV,
    );
    other_us += report(
        "attention scores + weighted sum",
        time(|| {
            let _ = scaled_dot(&head, &head, 1 << 28);
        }),
        HEADS * 2,
    );
    other_us += report(
        "softmax (per head, short context)",
        time(|| {
            let mut s = vec![1i64 << 30, 2 << 30, 3 << 30, 4 << 30];
            softmax(&mut s)
        }),
        HEADS,
    );
    other_us += report(
        "swiglu (3072 SiLU)",
        time(|| swiglu(&hidden_a, &hidden_b, &mut hidden_out)),
        1,
    );

    println!("{}", "-".repeat(58));
    println!("{:<34} {total:>12.1}", "total");
    println!(
        "\n  matmul          {:>8.1} us  ({:.0}%)",
        matmul_us,
        100.0 * matmul_us / total
    );
    println!(
        "  everything else {:>8.1} us  ({:.0}%)",
        other_us,
        100.0 * other_us / total
    );

    // The single most expensive non-matmul operation, isolated.
    let one_silu = time(|| {
        std::hint::black_box(silu(std::hint::black_box(1 << 31)));
    });
    println!(
        "\n  one SiLU is {:.0} ns, and a layer performs {HIDDEN} of them",
        one_silu * 1e9
    );
    println!(
        "  which is {:.0} us, or {:.0}% of the layer",
        one_silu * 1e6 * HIDDEN as f64,
        100.0 * one_silu * 1e6 * HIDDEN as f64 / total
    );

    println!(
        "\nA chain computing matmuls natively and the rest in the EVM would be bounded by\n\
         the rest. Whether that matters depends on how much slower the EVM is at these\n\
         operations than at matrix multiplication, which is the next thing to measure."
    );
}
