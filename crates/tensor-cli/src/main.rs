//! Generating tokens on a running chain, from outside it.
//!
//! Every other check in this repository executes the model in the same process that holds
//! the state. That proves the arithmetic and the contracts, but not the chain: a mistake
//! shared between the executor and the state would be invisible, and nothing about it looks
//! like a system anyone could join.
//!
//! This uploads a model to a live node as contract code, then drives a generation through
//! `start`, `step` and `emitToken` with signed transactions over JSON-RPC, and compares the
//! tokens the chain emits against what float64 says the model produces. The client shares
//! no memory with the node and no code with the tests — it encodes calls by hand for that
//! reason — so agreement means the chain did the work.
//!
//! ```text
//! cargo run --release -p tensor-node -- --journal run.txt &
//! cargo run --release -p tensor-cli
//! ```

mod abi;
mod layer;
mod rpc;

use abi::{addr_word, bytes_array, bytes_tuple, dyn_bytes, packed, selector, signed, word};
use alloy_primitives::Address;
use rpc::Client;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The account every development chain funds first; the key is public by design.
const KEY: &str = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
/// Gas offered per transaction. Above Ethereum's cap by design — a token exceeds
/// EIP-7825's 2^24 — and high enough to carry an 8MB upload batch at 226 gas a byte.
pub const GAS: u64 = 3_500_000_000;

/// Gas offered per transaction, overridable.
///
/// Raising the EVM's cap is not enough on a real client: reth's transaction pool enforces
/// EIP-7825's 2^24 limit itself, before execution, so a transaction asking for more is
/// rejected before any EVM sees it. A chain that meant to serve models would raise both;
/// against a stock pool, the granularity knob is what keeps a token inside the cap.
pub fn gas() -> u64 {
    std::env::var("GAS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(GAS)
}
/// Qwen3's epsilon in Q32, which is what the fixture was generated against.
const RMS_EPS: i64 = 4_295;

pub fn read(dir: &Path, name: &str) -> Result<Vec<u8>, String> {
    std::fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()))
}

pub fn read_i64(dir: &Path, name: &str) -> Result<Vec<i64>, String> {
    Ok(read(dir, name)?
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect())
}

/// The addresses an earlier run recorded: the classifier, then one per layer.
fn parse_placed(text: &str) -> Option<(Vec<Address>, Address)> {
    let head: Address = text
        .split("\"lm_head\":\"")
        .nth(1)?
        .split('"')
        .next()?
        .parse()
        .ok()?;
    let layers: Vec<Address> = text
        .split("\"layers\":[")
        .nth(1)?
        .split(']')
        .next()?
        .split(',')
        .filter_map(|t| t.trim().trim_matches('"').parse().ok())
        .collect();
    (!layers.is_empty()).then_some((layers, head))
}

pub fn read_f64(dir: &Path, name: &str) -> Result<Vec<f64>, String> {
    Ok(read(dir, name)?
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().expect("chunks_exact(8)")))
        .collect())
}

pub fn meta(s: &str, key: &str) -> Result<usize, String> {
    let at = s
        .find(&format!("\"{key}\":"))
        .ok_or_else(|| format!("meta.json has no {key}"))?;
    s[at + key.len() + 3..]
        .split([',', '\n', '}'])
        .next()
        .ok_or("truncated")?
        .trim()
        .parse()
        .map_err(|e| format!("{key}: {e}"))
}

/// Compiles a contract and returns its deployment bytecode.
fn solc_bin(src: &Path, name: &str) -> Result<Vec<u8>, String> {
    let out = Command::new("solc")
        .args(["--optimize", "--optimize-runs", "200", "--bin"])
        .arg(src)
        .output()
        .map_err(|e| format!("solc: {e}"))?;
    if !out.status.success() {
        return Err(format!("solc: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let at = text
        .find(&format!(":{name} ======="))
        .ok_or_else(|| format!("{name} not in solc output"))?;
    let hex = text[at..]
        .lines()
        .find(|l| l.starts_with("60"))
        .ok_or("no bytecode")?
        .trim();
    rpc::unhex(hex).ok_or_else(|| "bytecode is not hex".to_string())
}

/// Bytes per chunk, below `WeightStore`'s 24,000-byte limit.
///
/// Overridable because the two useful settings are opposite. A real model wants chunks as
/// large as the contract allows, since a Qwen3-0.6B layer is 31.5MB and every chunk is an
/// account. A fixture wants them small, so tensors straddle chunk boundaries and the
/// directory path is genuinely exercised — that path is the one every real model takes, and
/// leaving it unexercised is how it came to be broken.
fn chunk_bytes() -> usize {
    std::env::var("CHUNK_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(23_552)
}

/// Bytes of chunk payload to put in one `storeMany` transaction.
///
/// Batching matters more than it looks. Every chunk becomes an account, and the node
/// recomputes the state root over every account each block — measured at 0.53µs an account,
/// of which building the trie is about 70% and preparing leaves the rest. The block count is
/// therefore the lever: an upload spends time proportional to blocks times accounts, and
/// only the first of those is under a client's control.
///
/// Eight megabytes is what the gas cap allows. Code deposit is 200 gas a byte and calldata
/// 16, so a batch costs about 226 gas a byte: 8MB is 1.81 billion against this client's
/// 3.5 billion and the node's 4. Sixteen would exceed both.
/// Overridable so the trade-off can be measured rather than assumed: a bigger batch means
/// fewer blocks and less trie work, but the EVM charges for memory quadratically, so the
/// gas per byte rises with it.
fn batch_bytes() -> usize {
    std::env::var("BATCH_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8 << 20)
}

/// Stores a blob as contract code and returns the address to address it by.
///
/// One chunk is stored directly; anything larger becomes several chunks and a directory
/// listing them. The precompile resolves either, so a caller holds one address whichever
/// happened.
pub fn upload(
    c: &mut Client,
    uploader: Address,
    blob: &[u8],
    what: &str,
) -> Result<Address, String> {
    let size = chunk_bytes();
    let parts: Vec<&[u8]> = blob.chunks(size).collect();

    if parts.len() == 1 {
        let mut d = selector("store(bytes)");
        d.extend_from_slice(&word(32));
        d.extend_from_slice(&dyn_bytes(parts[0]));
        return rpc::first_topic(&c.send(Some(uploader), d, what)?);
    }

    let mut chunks = Vec::with_capacity(parts.len());
    let per_batch = (batch_bytes() / size).max(1);
    for (i, batch) in parts.chunks(per_batch).enumerate() {
        let mut d = selector("storeMany(bytes[])");
        d.extend_from_slice(&word(32));
        d.extend_from_slice(&bytes_array(batch));
        let r = c.send(Some(uploader), d, &format!("{what} batch {i}"))?;
        let got = rpc::all_topics(&r)?;
        if got.len() != batch.len() {
            return Err(format!(
                "{what} batch {i}: stored {} chunks but heard about {}",
                batch.len(),
                got.len()
            ));
        }
        chunks.extend(got);
    }

    let mut d = selector("finish(address[])");
    d.extend_from_slice(&word(32));
    d.extend_from_slice(&word(chunks.len() as u64));
    for a in &chunks {
        d.extend_from_slice(&addr_word(*a));
    }
    rpc::first_topic(&c.send(Some(uploader), d, &format!("{what} directory"))?)
}

pub fn say(s: &str) {
    println!("\n\x1b[1m{s}\x1b[0m");
}

fn main() {
    if let Err(e) = run() {
        eprintln!("\n\x1b[31mfailed:\x1b[0m {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let url = std::env::var("RPC").unwrap_or_else(|_| "http://127.0.0.1:8545".into());
    let dir: PathBuf = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent())
                .expect("workspace root")
                .join("testdata/tiny")
                .to_string_lossy()
                .into_owned()
        })
        .into();

    // A directory holding one exported layer rather than a whole model. Told apart by what
    // is in it: a model has a classifier, a layer does not.
    let one_layer = dir.join("experts.bin").exists() && !dir.join("lm_head.bin").exists();

    let mut c = Client::new(&url, KEY, gas())?;
    say("chain");
    println!("  rpc         {url}");
    println!("  sender      {:#x}", c.address());
    println!("  block       {}", c.block_number()?);

    let contracts = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .join("contracts/src");

    // A single layer takes a different path: it has no classifier and no vocabulary, so the
    // model fields below would not be there to read.
    if one_layer {
        say("deploying");
        let up = solc_bin(&contracts.join("WeightStore.sol"), "Uploader")?;
        let uploader = rpc::created(&c.send(None, up, "deploy Uploader")?)?;
        println!("  uploader    {uploader:#x}");
        let gen_bin = solc_bin(&contracts.join("Generator.sol"), "Generator")?;
        let generator = rpc::created(&c.send(None, gen_bin, "deploy Generator")?)?;
        println!("  generator   {generator:#x}");
        return layer::run(&mut c, &dir, uploader, generator);
    }

    // ---- the model ----
    let mj = std::fs::read_to_string(dir.join("meta.json"))
        .map_err(|e| format!("{}: {e}", dir.join("meta.json").display()))?;
    let dim = meta(&mj, "dim")?;
    let hidden = meta(&mj, "hidden")?;
    let n_layers = meta(&mj, "n_layers")?;
    let n_heads = meta(&mj, "n_heads")?;
    let n_kv = meta(&mj, "n_kv")?;
    let head_dim = meta(&mj, "head_dim")?;
    let vocab = meta(&mj, "vocab")?;
    let positions = meta(&mj, "positions")?;
    // Zero experts is a dense model, which is what `Transformer` branches on.
    let n_experts = meta(&mj, "n_experts").unwrap_or(0);
    let top_k = meta(&mj, "top_k").unwrap_or(0);
    let (q_width, kv_width, half) = (n_heads * head_dim, n_kv * head_dim, head_dim / 2);

    say("model");
    println!("  fixture     {}", dir.display());
    println!("  layers      {n_layers}   dim {dim}   hidden {hidden}");
    println!("  heads       {n_heads} query, {n_kv} kv, head_dim {head_dim}");
    println!("  vocab       {vocab}   positions {positions}");

    // ---- deploy the runtime ----
    say("deploying");
    let up = solc_bin(&contracts.join("WeightStore.sol"), "Uploader")?;
    let uploader = rpc::created(&c.send(None, up, "deploy Uploader")?)?;
    println!("  uploader    {uploader:#x}");
    let gen_bin = solc_bin(&contracts.join("Generator.sol"), "Generator")?;
    let generator = rpc::created(&c.send(None, gen_bin, "deploy Generator")?)?;
    println!("  generator   {generator:#x}");

    // ---- upload the weights ----
    //
    // One blob per layer: seven tensors then four norm vectors, laid out contiguously, with
    // the offset of each recorded so the contract can address into it. This is the layout a
    // 31.5MB layer uses; nothing about it is special to a fixture this small.
    let attention: [(&str, usize, usize); 4] = [
        ("q", q_width, dim),
        ("k", kv_width, dim),
        ("v", kv_width, dim),
        ("o", dim, q_width),
    ];
    let gate_len = hidden + hidden * dim * 2;
    let down_len = dim + dim * hidden * 2;
    let stride = gate_len * 2 + down_len;

    // Where a layer's tensors sit inside its blob is arithmetic — the shapes decide it —
    // so it can be worked out without reading a gigabyte back off disk. That is what lets a
    // second run skip the upload entirely.
    let layout = |off: &mut [u64; 12]| {
        let mut at = 0u64;
        for (i, (_, rows, cols)) in attention.iter().enumerate() {
            off[i] = at;
            at += (rows + rows * cols * 2) as u64;
        }
        if n_experts == 0 {
            for (i, (rows, cols)) in [(hidden, dim), (hidden, dim), (dim, hidden)]
                .iter()
                .enumerate()
            {
                off[4 + i] = at;
                at += (rows + rows * cols * 2) as u64;
            }
        } else {
            off[4] = at;
            off[5] = at + gate_len as u64;
            off[6] = at + (gate_len * 2) as u64;
            at += (n_experts * stride) as u64;
        }
        for (j, n) in [dim, head_dim, head_dim, dim].into_iter().enumerate() {
            off[7 + j] = at;
            at += (n * 8) as u64;
        }
        if n_experts != 0 {
            off[11] = at;
        }
    };

    // A model already on this chain does not need uploading again. Where it landed is
    // written beside the weights, and a later run reads it — which is the difference
    // between a demonstration and something somebody would sit in front of, since the
    // upload takes minutes and answering a prompt takes seconds.
    let placed = dir.join("placed.json");
    let mut layer_at: Vec<Address> = Vec::with_capacity(n_layers);
    let mut offsets = vec![[0u64; 12]; n_layers];
    let mut head = Address::ZERO;
    let mut reused = false;

    if let Some((saved_layers, saved_head)) = std::fs::read_to_string(&placed)
        .ok()
        .and_then(|t| parse_placed(&t))
    {
        if saved_layers.len() == n_layers && c.has_code(saved_head)? {
            say("model already on this chain");
            println!("  layers      {}", saved_layers.len());
            println!("  lm_head     {saved_head:#x}");
            layer_at = saved_layers;
            head = saved_head;
            for off in offsets.iter_mut() {
                layout(off);
            }
            reused = true;
        }
    }

    if reused {
        // Nothing to upload.
    } else {
        say("uploading weights as contract code");
        let mut uploaded = 0usize;
        for (l, off) in offsets.iter_mut().enumerate() {
            let mut blob = Vec::new();
            for (i, (name, rows, cols)) in attention.iter().enumerate() {
                let raw = read(&dir, &format!("l{l}_{name}.bin"))?;
                if raw.len() != rows + rows * cols * 2 {
                    return Err(format!("l{l}_{name}: expected {rows}x{cols}"));
                }
                off[i] = blob.len() as u64;
                blob.extend_from_slice(&raw);
            }

            // A dense layer carries gate, up and down directly. A mixture carries every expert
            // laid out expert by expert, so expert e's sit one stride past expert zero's.
            let ffn_at = blob.len();
            if n_experts == 0 {
                for (i, (name, rows, cols)) in [
                    ("gate", hidden, dim),
                    ("up", hidden, dim),
                    ("down", dim, hidden),
                ]
                .iter()
                .enumerate()
                {
                    let raw = read(&dir, &format!("l{l}_{name}.bin"))?;
                    if raw.len() != rows + rows * cols * 2 {
                        return Err(format!("l{l}_{name}: expected {rows}x{cols}"));
                    }
                    off[4 + i] = blob.len() as u64;
                    blob.extend_from_slice(&raw);
                }
            } else {
                let raw = read(&dir, &format!("l{l}_experts.bin"))?;
                if raw.len() != n_experts * stride {
                    return Err(format!("l{l}_experts: expected {n_experts} x {stride}"));
                }
                off[4] = ffn_at as u64;
                off[5] = (ffn_at + gate_len) as u64;
                off[6] = (ffn_at + gate_len * 2) as u64;
                blob.extend_from_slice(&raw);
            }

            for (j, name) in ["input_norm", "q_norm", "k_norm", "post_norm"]
                .iter()
                .enumerate()
            {
                off[7 + j] = blob.len() as u64;
                blob.extend(packed(&read_i64(&dir, &format!("l{l}_{name}.bin"))?));
            }
            if n_experts != 0 {
                off[11] = blob.len() as u64;
                blob.extend_from_slice(&read(&dir, &format!("l{l}_router.bin"))?);
            }
            uploaded += blob.len();

            let parts = blob.len().div_ceil(chunk_bytes());
            let at = upload(&mut c, uploader, &blob, &format!("layer {l}"))?;
            println!(
                "  layer {l}     {at:#x}  {} bytes in {parts} chunk{}",
                blob.len(),
                if parts == 1 { "" } else { "s" }
            );
            layer_at.push(at);
        }

        let head_blob = read(&dir, "lm_head.bin")?;
        head = upload(&mut c, uploader, &head_blob, "lm_head")?;
        uploaded += head_blob.len();
        println!(
            "  lm_head     {head:#x}  {} bytes in {} chunks",
            head_blob.len(),
            head_blob.len().div_ceil(chunk_bytes())
        );
        println!("  total       {uploaded} bytes on chain");

        // Written after the fact, so a run that failed part way through does not leave a note
        // claiming a model is somewhere it is not.
        let mut note = String::from("{\"lm_head\":\"");
        note.push_str(&format!("{head:#x}"));
        note.push_str("\",\"layers\":[");
        for (i, a) in layer_at.iter().enumerate() {
            if i > 0 {
                note.push(',');
            }
            note.push_str(&format!("\"{a:#x}\""));
        }
        note.push_str("]}");
        let _ = std::fs::write(&placed, note);
    }

    // ---- generate ----
    let final_norm = read_i64(&dir, "final_norm.bin")?;

    // A prompt given as token ids turns this into real generation: the model is fed its own
    // output and keeps going. Without one it replays the positions the exporter baked in,
    // which is what every measurement above used.
    let prompt: Vec<usize> = std::env::var("PROMPT_IDS")
        .ok()
        .map(|v| v.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_default();
    let generate: usize = std::env::var("GENERATE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // The embedding table, which for a model with tied weights is the classifier — already
    // on chain, and read here only to look up a row. That is a table lookup rather than a
    // multiply, which is why the row is supplied to the contract rather than fetched by it.
    let embed_table = if prompt.is_empty() {
        Vec::new()
    } else {
        read(&dir, "lm_head.bin")?
    };
    let embed_row = |t: usize| -> Result<Vec<i64>, String> {
        if t >= vocab {
            return Err(format!("token {t} is outside a vocabulary of {vocab}"));
        }
        let scale = embed_table[t] as i8;
        let at = vocab + t * dim * 2;
        // `q * 2^-scale` as a Q32 activation is `q << (32 - scale)`, exactly.
        Ok(embed_table[at..at + dim * 2]
            .chunks_exact(2)
            .map(|c| (i16::from_be_bytes([c[0], c[1]]) as i64) << (32 - scale as i64))
            .collect())
    };

    let embed = read_i64(&dir, "embed.bin")?;
    let cos = read_i64(&dir, "cos.bin")?;
    let sin = read_i64(&dir, "sin.bin")?;

    // Seven widths, the epsilon, then the mixture fields — zero here, which is what selects
    // the dense feed-forward. They sit after `rmsEps` so a dense encoding is unchanged up to
    // the point it stops caring.
    let config = |out: &mut Vec<u8>| {
        for f in [dim, hidden, n_heads, n_kv, head_dim, positions, n_layers] {
            out.extend_from_slice(&word(f as u64));
        }
        out.extend_from_slice(&signed(RMS_EPS));
        for f in [n_experts as u64, top_k as u64, stride as u64] {
            out.extend_from_slice(&word(f));
        }
    };

    let cfg_ty = "(uint32,uint32,uint32,uint32,uint32,uint32,uint32,int256,uint32,uint32,uint32)";
    let layer_ty = "(address,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,uint32,\
uint32,uint32,uint32)";
    let step_sig = format!("step(bytes32,({cfg_ty},{layer_ty}[],(bytes,bytes)),uint16)");
    let emit_sig = format!("emitToken(bytes32,{cfg_ty},bytes,address,uint32,uint32)");
    // A session id nobody else is using. Sessions are owned and keyed, so two generations
    // can run at once without one destroying the other.
    let id = [0x7au8; 32];

    say("generating");
    println!("  each position: seed, run every layer, classify\n");
    let steps = if prompt.is_empty() {
        positions
    } else {
        prompt.len() + generate
    };
    if steps > positions {
        return Err(format!(
            "asked for {steps} positions but the rotary tables cover {positions}; \
             export with --positions {steps}"
        ));
    }

    let mut tokens: Vec<u32> = Vec::with_capacity(steps);
    for pos in 0..steps {
        let before = c.gas_used;

        let mut d = selector(if pos == 0 {
            "start(bytes32,bytes)"
        } else {
            "feed(bytes32,bytes)"
        });
        d.extend_from_slice(&id);
        d.extend_from_slice(&word(2 * 32));
        // The row for this position: a prompt token while the prompt lasts, and after that
        // the token the chain itself just produced. That feedback is what makes this
        // generation rather than replay.
        let row: Vec<i64> = if prompt.is_empty() {
            embed[pos * dim..(pos + 1) * dim].to_vec()
        } else if pos < prompt.len() {
            embed_row(prompt[pos])?
        } else {
            embed_row(tokens[pos - 1] as usize)?
        };
        d.extend_from_slice(&dyn_bytes(&packed(&row)));
        c.send(Some(generator), d, "seed")?;

        // The model, re-encoded per position: the rotary table is the only part that
        // depends on where in the sequence we are.
        let mut model = Vec::new();
        config(&mut model);
        let mut layers = word(n_layers as u64).to_vec();
        for (l, off) in offsets.iter().enumerate() {
            layers.extend_from_slice(&addr_word(layer_at[l]));
            for o in off {
                layers.extend_from_slice(&word(*o));
            }
        }
        let a = pos * half;
        let rope = bytes_tuple(&[&packed(&cos[a..a + half]), &packed(&sin[a..a + half])]);
        // Eleven config words then the two offsets that follow it.
        let head_off = 13 * 32;
        model.extend_from_slice(&word(head_off as u64));
        model.extend_from_slice(&word((head_off + layers.len()) as u64));
        model.extend_from_slice(&layers);
        model.extend_from_slice(&rope);

        // How many layers one transaction attempts. A whole token of a 48-layer mixture is
        // about 6.6 billion gas, past any single transaction's cap, so a token is split —
        // which is what the granularity knob is for. Splitting must not change the token,
        // and `granularity_does_not_change_the_output` is what says it does not.
        let max_layers: usize = std::env::var("MAX_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(n_layers)
            .clamp(1, n_layers);

        let mut done = 0usize;
        let mut steps = 0;
        while done < n_layers {
            let mut d = selector(&step_sig);
            d.extend_from_slice(&id);
            d.extend_from_slice(&word(3 * 32));
            d.extend_from_slice(&word(max_layers as u64));
            d.extend_from_slice(&model);
            let out = c.send_reading(generator, d, &format!("step from layer {done}"))?;
            let next = rpc::word_at(&out, 0).ok_or("step returned no layer")? as usize;
            if next <= done {
                return Err(format!("step did not advance past layer {done}"));
            }
            done = next;
            steps += 1;
        }

        let mut d = selector(&emit_sig);
        d.extend_from_slice(&id);
        config(&mut d);
        // id, eleven config words, this offset, the head address and two more: sixteen.
        d.extend_from_slice(&word(16 * 32));
        d.extend_from_slice(&addr_word(head));
        d.extend_from_slice(&word(0));
        d.extend_from_slice(&word(vocab as u64));
        d.extend_from_slice(&dyn_bytes(&packed(&final_norm)));
        let out = c.send_reading(generator, d, "emitToken")?;
        let token = rpc::word_at(&out, 0).ok_or("emitToken returned no token")? as u32;
        tokens.push(token);

        println!(
            "  position {pos}: token {token:>5}   {} transactions   {} gas",
            steps + 2,
            c.gas_used - before
        );
    }

    if !prompt.is_empty() {
        say("generated");
        println!("  prompt      {prompt:?}");
        println!("  continued   {:?}", &tokens[prompt.len()..]);
        println!("  all         {tokens:?}");
        say("chain state");
        println!("  blocks      {}", c.block_number()?);
        println!("  sent        {} transactions", c.sent);
        println!("  gas         {}", c.gas_used);
        println!("  state root  {}", c.state_root()?);
        return Ok(());
    }

    // ---- against what float64 says ----
    say("what float64 says");
    let want: Vec<u32> = read(&dir, "tokens.bin")?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact(4)")))
        .collect();
    println!("  expected    {want:?}");
    println!("  chain       {tokens:?}");

    say("chain state");
    println!("  blocks      {}", c.block_number()?);
    println!("  sent        {} transactions", c.sent);
    println!("  gas         {}", c.gas_used);
    println!("  state root  {}", c.state_root()?);

    if tokens != want {
        return Err("the chain emitted different tokens".into());
    }
    println!("\n\x1b[1m  match: yes\x1b[0m — every token, generated on chain\n");
    Ok(())
}
