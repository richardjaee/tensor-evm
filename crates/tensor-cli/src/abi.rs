//! Just enough ABI encoding to call the contracts, written out by hand.
//!
//! A full ABI library would be the obvious choice, and it is deliberately not used here.
//! The point of this client is that it is *outside* the node — it shares no code with the
//! thing it is checking, so agreement between them means something. Pulling in the same
//! encoder the tests use would quietly reintroduce a shared assumption, and the encodings
//! involved are a dozen lines.

use alloy_primitives::{keccak256, Address};

/// A 32-byte word holding an unsigned value.
pub fn word(v: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&v.to_be_bytes());
    w
}

/// A 32-byte word holding a signed value, sign-extended as the ABI requires.
pub fn signed(v: i64) -> [u8; 32] {
    let mut w = [0u8; 32];
    if v < 0 {
        w[..24].fill(0xff);
    }
    w[24..].copy_from_slice(&v.to_be_bytes());
    w
}

pub fn addr_word(a: Address) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(a.as_slice());
    w
}

/// Activations as the contracts pass them: big-endian `int64`, packed, no per-element
/// padding. Padding each to a word would quadruple the calldata for no gain.
pub fn packed(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_be_bytes()).collect()
}

/// A dynamic `bytes`: length, contents, then padding to a word boundary.
pub fn dyn_bytes(b: &[u8]) -> Vec<u8> {
    let mut out = word(b.len() as u64).to_vec();
    out.extend_from_slice(b);
    out.extend(std::iter::repeat_n(0u8, (32 - b.len() % 32) % 32));
    out
}

/// A tuple of dynamic `bytes` members: offsets first, then the members themselves.
pub fn bytes_tuple(parts: &[&[u8]]) -> Vec<u8> {
    let tails: Vec<Vec<u8>> = parts.iter().map(|p| dyn_bytes(p)).collect();
    let mut out = Vec::new();
    let mut at = parts.len() * 32;
    for t in &tails {
        out.extend_from_slice(&word(at as u64));
        at += t.len();
    }
    for t in tails {
        out.extend_from_slice(&t);
    }
    out
}

/// The first four bytes of the hash of a signature, which is how the EVM dispatches.
pub fn selector(sig: &str) -> Vec<u8> {
    keccak256(sig.as_bytes())[..4].to_vec()
}

/// A `bytes[]` argument: the array's length, then one offset per member relative to the
/// start of the array, then the members themselves.
pub fn bytes_array(parts: &[&[u8]]) -> Vec<u8> {
    let tails: Vec<Vec<u8>> = parts.iter().map(|p| dyn_bytes(p)).collect();
    let mut out = word(parts.len() as u64).to_vec();
    let mut at = parts.len() * 32;
    for t in &tails {
        out.extend_from_slice(&word(at as u64));
        at += t.len();
    }
    for t in tails {
        out.extend_from_slice(&t);
    }
    out
}
