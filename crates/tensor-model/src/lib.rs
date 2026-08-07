//! Deterministic integer transformer inference.
//!
//! The matmul lives in `tensor-kernel` and is exposed to a chain as a precompile. This
//! crate is everything around it: normalisation, softmax, rotary embedding, attention, and
//! the layer that composes them — all in Q32 fixed point, all reproducible bit for bit on
//! any machine.
//!
//! Each operation is checked against float64 rather than against another integer
//! implementation. That distinction is not pedantry: an earlier version of the kernel and
//! its reference generator shared a requantisation mistake, agreed with each other on every
//! test, and were wrong by more than a thousand percent.

/// The fixed-point primitives live beside the matmul, in `tensor-kernel`: they are the
/// same layer of the stack, and a chain exposing one as a precompile has to expose the
/// others for the same reason.
pub use tensor_kernel::fixed;

pub mod ops;
pub mod qwen3;
