//! One real mixture layer, uploaded to a chain and run over JSON-RPC.
//!
//! The whole Qwen3-30B-A3B is 61GB, which this node cannot hold: every chunk is an account
//! and the state root is recomputed over all of them each block, so two and a half million
//! accounts would spend a day inside the trie rather than the upload. A layer is 1.25GB —
//! about what Qwen3-0.6B's entire upload was — and it is the piece that carries the claim
//! anyway.
//!
//! The claim being: a token pays for the experts it visits rather than the experts the model
//! holds. Eight of a hundred and twenty-eight, so 76MB of a layer's 1,208MB, decided inside
//! the VM and paid for in gas that anyone can read off the receipt.

use crate::abi::{addr_word, bytes_tuple, dyn_bytes, packed, selector, signed, word};
use crate::rpc::Client;
use crate::{meta, read, read_f64, read_i64, say, upload};
use alloy_primitives::Address;
use std::path::Path;

const Q32: f64 = 4_294_967_296.0;

const CFG_TY: &str =
    "(uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,uint32,uint32,uint32)";
const LAYER_TY: &str =
    "(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32)";

pub fn run(
    c: &mut Client,
    dir: &Path,
    uploader: Address,
    generator: Address,
) -> Result<(), String> {
    let mj = std::fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| format!("{}: {e}", dir.join("meta.json").display()))?;

    let dim = meta(&mj, "dim")?;
    let hidden = meta(&mj, "hidden")?;
    let n_heads = meta(&mj, "n_heads")?;
    let n_kv = meta(&mj, "n_kv")?;
    let head_dim = meta(&mj, "head_dim")?;
    let positions = meta(&mj, "positions")?;
    let n_experts = meta(&mj, "n_experts")?;
    let top_k = meta(&mj, "top_k")?;
    let (q_width, kv_width, half) = (n_heads * head_dim, n_kv * head_dim, head_dim / 2);

    say("layer");
    println!("  dim {dim}, heads {n_heads}, kv {n_kv}, head_dim {head_dim}");
    println!("  mixture: {n_experts} experts of {hidden} wide, top {top_k}");

    // ---- the blob, in the layout the chain reads ----
    let gate_len = hidden + hidden * dim * 2;
    let down_len = dim + dim * hidden * 2;
    let stride = gate_len * 2 + down_len;

    let mut blob = Vec::new();
    let mut off = [0u64; 12];
    for (i, (name, rows, cols)) in [
        ("q", q_width, dim),
        ("k", kv_width, dim),
        ("v", kv_width, dim),
        ("o", dim, q_width),
    ]
    .iter()
    .enumerate()
    {
        let raw = read(dir, &format!("{name}.bin"))?;
        if raw.len() != rows + rows * cols * 2 {
            return Err(format!("{name}: expected {rows}x{cols}"));
        }
        off[i] = blob.len() as u64;
        blob.extend_from_slice(&raw);
    }

    let experts = read(dir, "experts.bin")?;
    if experts.len() != n_experts * stride {
        return Err(format!(
            "experts.bin is {} bytes, expected {n_experts} x {stride}",
            experts.len()
        ));
    }
    // Expert zero's three tensors; the rest are reached by stride.
    let at = blob.len();
    off[4] = at as u64;
    off[5] = (at + gate_len) as u64;
    off[6] = (at + gate_len * 2) as u64;
    blob.extend_from_slice(&experts);

    for (j, name) in ["input_norm", "q_norm", "k_norm", "post_norm"]
        .iter()
        .enumerate()
    {
        off[7 + j] = blob.len() as u64;
        blob.extend(packed(&read_i64(dir, &format!("{name}.bin"))?));
    }
    off[11] = blob.len() as u64;
    blob.extend_from_slice(&read(dir, "router.bin")?);

    say("uploading the layer as contract code");
    println!("  {:.2} GB", blob.len() as f64 / 1e9);
    let before = c.gas_used;
    let layer_at = upload(c, uploader, &blob, "layer")?;
    println!("  {layer_at:#x}");
    println!("  {} transactions, {} gas", c.sent, c.gas_used - before);

    // ---- run it ----
    let xs = read_i64(dir, "x.bin")?;
    let expected = read_f64(dir, "expected.bin")?;
    let cos = read_i64(dir, "cos.bin")?;
    let sin = read_i64(dir, "sin.bin")?;
    let id = [0x30u8; 32];

    let mut model = Vec::new();
    for f in [dim, hidden, n_heads, n_kv, head_dim, positions, 1] {
        model.extend_from_slice(&word(f as u64));
    }
    model.extend_from_slice(&signed(4_295)); // Qwen3's epsilon in Q32
    for f in [n_experts as u64, top_k as u64, stride as u64] {
        model.extend_from_slice(&word(f));
    }

    let mut layers = word(1).to_vec();
    layers.extend_from_slice(&addr_word(layer_at));
    for o in &off {
        layers.extend_from_slice(&word(*o));
    }

    let step_sig = format!("step(bytes32,({CFG_TY},{LAYER_TY}[],(bytes,bytes)),uint16)");

    say("running the layer, on chain");
    println!("  a token reads {top_k}/{n_experts} of the feed-forward\n");

    let mut worst = 0.0f64;
    for p in 0..positions {
        let spent = c.gas_used;

        let mut d = selector(if p == 0 {
            "start(bytes32,bytes)"
        } else {
            "feed(bytes32,bytes)"
        });
        d.extend_from_slice(&id);
        d.extend_from_slice(&word(2 * 32));
        d.extend_from_slice(&dyn_bytes(&packed(&xs[p * dim..(p + 1) * dim])));
        c.send(Some(generator), d, "seed")?;

        let a = p * half;
        let rope = bytes_tuple(&[&packed(&cos[a..a + half]), &packed(&sin[a..a + half])]);
        let head = 13 * 32;
        let mut m = model.clone();
        m.extend_from_slice(&word(head as u64));
        m.extend_from_slice(&word((head + layers.len()) as u64));
        m.extend_from_slice(&layers);
        m.extend_from_slice(&rope);

        let mut d = selector(&step_sig);
        d.extend_from_slice(&id);
        d.extend_from_slice(&word(3 * 32));
        d.extend_from_slice(&word(1));
        d.extend_from_slice(&m);
        c.send(Some(generator), d, "step")?;

        let mut d = selector("residual(bytes32,uint32)");
        d.extend_from_slice(&id);
        d.extend_from_slice(&word(dim as u64));
        let ret = c.call(generator, d)?;
        let got: Vec<i64> = ret[64..]
            .chunks_exact(32)
            .take(dim)
            .map(|w| i64::from_be_bytes(w[24..].try_into().expect("eight bytes")))
            .collect();

        let want = &expected[p * dim..(p + 1) * dim];
        let scale: f64 = want.iter().map(|v| v.abs()).sum::<f64>() / dim as f64;
        let err = got
            .iter()
            .zip(want)
            .map(|(g, t)| ((*g as f64 / Q32) - t).abs())
            .fold(0.0f64, f64::max)
            / scale;
        worst = worst.max(err);

        println!(
            "  position {p}: {:>12} gas   rel err {err:.2e}",
            c.gas_used - spent
        );

        // Advancing to the next position needs `emitToken`, which is what resets the layer
        // counter and moves the cache on. A single layer has no classifier, so the router
        // stands in as one: the token it produces is meaningless and ignored, and the call
        // is made only for its effect on the session. It does not touch the residual — the
        // comparison above already happened, and `emitToken` reads into scratch.
        if p + 1 < positions {
            let emit_sig = format!("emitToken(bytes32,{CFG_TY},bytes,address,uint32,uint32)");
            let mut d = selector(&emit_sig);
            d.extend_from_slice(&id);
            d.extend_from_slice(&model);
            d.extend_from_slice(&word(16 * 32));
            d.extend_from_slice(&addr_word(layer_at));
            d.extend_from_slice(&word(off[11]));
            d.extend_from_slice(&word(n_experts as u64));
            d.extend_from_slice(&dyn_bytes(&packed(&read_i64(dir, "post_norm.bin")?)));
            c.send(Some(generator), d, "advance")?;
        }
    }

    say("against float64");
    println!("  worst relative error  {worst:.2e}");
    println!("  state root            {}", c.state_root()?);
    if worst > 1e-2 {
        return Err(format!("the chain's layer diverged: {worst:.2e}"));
    }
    println!("\n\x1b[1m  match: yes\x1b[0m — a real 30B mixture layer, on chain\n");
    Ok(())
}
