//! Measures the kernel at real transformer shapes and projects what it implies.
//!
//! The question this exists to answer: if the matmul moves out of the EVM interpreter and
//! into native code, what is the resulting ceiling on model size for a chain where every
//! validator re-executes every token?
//!
//! The answer is a bandwidth figure, not a FLOP figure. A forward pass reads every weight
//! exactly once and does two operations with it, so past a certain size the arithmetic is
//! free and the memory system is the whole story. Reporting GB/s rather than GOP/s keeps
//! that visible.

use std::time::Instant;
use tensor_kernel::{matmul, matmul_scalar, weight_bytes};

/// Repeat until at least this much time has passed, so short shapes are not measured
/// against timer noise.
const MIN_SECS: f64 = 0.35;

fn bench<F: FnMut()>(mut f: F) -> f64 {
    // One untimed pass so the weights are resident and we measure steady state rather
    // than the first-touch page faults.
    f();
    let start = Instant::now();
    let mut iters = 0u32;
    loop {
        f();
        iters += 1;
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= MIN_SECS {
            return elapsed / iters as f64;
        }
    }
}

fn synth(rows: usize, cols: usize) -> (Vec<i16>, Vec<i8>, Vec<i64>) {
    let w = (0..rows * cols)
        .map(|i| (((i as i64 * 2654435761) % 4001) - 2000) as i16)
        .collect();
    let scales = vec![30i8; rows];
    let x = (0..cols)
        .map(|i| ((i as i64 * 40503 + 17) % 131071) - 65535)
        .collect();
    (w, scales, x)
}

fn main() {
    let threads = rayon::current_num_threads();
    println!("threads: {threads}\n");

    // Shapes drawn from real models: a small dense projection, SmolLM2-135M's width,
    // Qwen3-32B's, and a slab large enough to leave cache entirely.
    let shapes: &[(&str, usize, usize)] = &[
        ("SmolLM2-135M  dim 576", 576, 576),
        ("SmolLM2-135M  ffn", 1536, 576),
        ("Qwen3-4B      dim 2560", 2560, 2560),
        ("Qwen3-32B     dim 5120", 5120, 5120),
        ("Qwen3-32B     ffn", 25600, 5120),
        // The shapes a real Qwen3-0.6B layer actually issues. Peak throughput is measured
        // on slabs far larger than these, and the gap between the two is the difference
        // between a projection and a measurement.
        ("Qwen3-0.6B    q_proj", 2048, 1024),
        ("Qwen3-0.6B    k_proj", 1024, 1024),
        ("Qwen3-0.6B    gate", 3072, 1024),
        ("Qwen3-0.6B    down", 1024, 3072),
    ];

    println!(
        "{:<24} {:>10} {:>12} {:>12} {:>10}",
        "shape", "MiB", "scalar GB/s", "par GB/s", "speedup"
    );
    println!("{}", "-".repeat(72));

    let mut best_parallel = 0.0f64;

    for &(name, rows, cols) in shapes {
        let (w, scales, x) = synth(rows, cols);
        let mut out = vec![0i64; rows];
        let bytes = weight_bytes(rows, cols) as f64;

        // Only benchmark the scalar path on shapes where it will not dominate the run.
        let scalar_gbs = if rows * cols <= 8_000_000 {
            let t = bench(|| matmul_scalar(&w, &scales, &x, rows, cols, &mut out));
            bytes / t / 1e9
        } else {
            f64::NAN
        };

        let t = bench(|| matmul(&w, &scales, &x, rows, cols, &mut out));
        let par_gbs = bytes / t / 1e9;
        best_parallel = best_parallel.max(par_gbs);

        if scalar_gbs.is_nan() {
            println!(
                "{:<24} {:>10.1} {:>12} {:>12.1} {:>10}",
                name,
                bytes / 1048576.0,
                "-",
                par_gbs,
                "-"
            );
        } else {
            println!(
                "{:<24} {:>10.1} {:>12.1} {:>12.1} {:>9.1}x",
                name,
                bytes / 1048576.0,
                scalar_gbs,
                par_gbs,
                par_gbs / scalar_gbs
            );
        }
    }

    // A forward pass reads every weight once per token, so time per token is simply the
    // model's size divided by achieved bandwidth. For mixture-of-experts models only the
    // active parameters are read, which is the entire reason they are tractable at all.
    println!("\nprojected seconds per token at {best_parallel:.1} GB/s (int16 weights)\n");
    println!("{:<28} {:>12} {:>14}", "model", "read/token", "sec/token");
    println!("{}", "-".repeat(56));

    let models: &[(&str, f64)] = &[
        ("SmolLM2-135M", 135e6),
        ("Qwen3-4B", 4e9),
        ("Qwen3-32B", 32e9),
        ("DeepSeek-V3 (37B active)", 37e9),
        ("dense 1T", 1e12),
    ];
    for &(name, params) in models {
        let gb = params * 2.0 / 1e9;
        println!("{:<28} {:>10.1} GB {:>14.2}", name, gb, gb / best_parallel);
    }

    println!(
        "\nNote: these are upper bounds on throughput. They assume weights stream from RAM\n\
         at the measured rate and count nothing for attention, softmax, KV cache traffic,\n\
         or reading weights that do not fit in memory. Treat them as the ceiling a real\n\
         implementation is measured against, not as a prediction."
    );
}
