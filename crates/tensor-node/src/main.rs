//! A development chain that carries the tensor precompiles.
//!
//! Everything measured until now ran inside a test harness: revm executing transactions in
//! a process, with no blocks, no RPC, and nothing anyone else could look at. This is the
//! smallest thing that is actually a chain — it seals blocks, keeps state between them, and
//! answers Ethereum's JSON-RPC, so `cast`, `forge`, MetaMask and a block explorer all work
//! against it unchanged.
//!
//! What it is not is a client. There is no peer-to-peer, no consensus, and no sync, so
//! nobody can join it; a real deployment means the same precompiles inside reth. The claim
//! this supports is narrower and worth having on its own: that a model runs under an
//! ordinary Ethereum interface, and that what it did is a matter of public record
//! afterwards rather than something that happened inside one process.
//!
//! ```text
//! cargo run --release -p tensor-node
//! cast block-number --rpc-url http://127.0.0.1:8545
//! ```

mod chain;
mod replay;
mod root;
mod rpc;
mod trie;

use chain::{Chain, CHAIN_ID, GAS_LIMIT, MAX_CODE_SIZE};
use revm::primitives::Address;
use serde_json::Value;
use std::sync::{Arc, Mutex};

/// The account `cast` and `forge` use by default, and the one Anvil funds first. Prefunding
/// it means the usual commands work without anyone being told a new key.
const DEV_ACCOUNT: [u8; 20] = [
    0xf3, 0x9f, 0xd6, 0xe5, 0x1a, 0xad, 0x88, 0xf6, 0xf4, 0xce, 0x6a, 0xb8, 0x82, 0x72, 0x79, 0xcf,
    0xff, 0xb9, 0x22, 0x66,
];

fn main() {
    let mut args = std::env::args().skip(1);
    let mut port: u16 = 8545;
    let mut journal = None;
    let mut replay_from = None;
    let mut archive = true;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--journal" => journal = args.next().map(std::path::PathBuf::from),
            // A bulk upload arrives as as many bytes as it stores, so keeping every
            // transaction doubles what the node holds.
            "--no-tx-archive" => archive = false,
            "--replay" => replay_from = args.next().map(std::path::PathBuf::from),
            other => port = other.parse().unwrap_or(port),
        }
    }

    // Replay is a whole mode rather than a flag on the server: it must not bind a port, so
    // that re-executing a chain never depends on being able to serve one.
    if let Some(path) = replay_from {
        if let Err(e) = replay::replay(&path, &[Address::from(DEV_ACCOUNT)]) {
            eprintln!("replay failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    let mut c = Chain::new(&[Address::from(DEV_ACCOUNT)]);
    c.journal = journal.clone();
    c.archive_txs = archive;
    let chain = Arc::new(Mutex::new(c));
    let server = tiny_http::Server::http(("127.0.0.1", port)).expect("could not bind");

    println!("tensor-node on http://127.0.0.1:{port}");
    println!("  chain id           {CHAIN_ID}");
    println!("  gas limit          {GAS_LIMIT}");
    println!("  max code size      {} MB", MAX_CODE_SIZE / (1024 * 1024));
    println!("  prefunded          {:#x}", Address::from(DEV_ACCOUNT));
    if let Some(j) = &journal {
        println!("  journal            {}", j.display());
    }
    if !archive {
        println!("  transaction archive  off — receipts keep no signed bytes");
    }
    println!("  precompiles        0x101 matmul   0x102 swiglu   0x103 norm+rope");
    println!("                     0x104 attention 0x105 rmsnorm 0x106 classify");
    println!(
        "\n  The gas limit and code size are raised from Ethereum's: a layer of Qwen3-0.6B\n\
         \x20 is 31.5MB against EIP-170's 24KB, and a token exceeds EIP-7825's 2^24 cap.\n\
         \x20 Both are configuration, and both are stated rather than assumed.\n"
    );

    for mut request in server.incoming_requests() {
        let mut body = String::new();
        if request.as_reader().read_to_string(&mut body).is_err() {
            let _ = request.respond(tiny_http::Response::empty(400));
            continue;
        }

        let reply = match serde_json::from_str::<Value>(&body) {
            // Batched calls are a single JSON array, which every wallet and explorer uses
            // on connect — answering only single calls looks like an unreachable node.
            Ok(Value::Array(calls)) => {
                let mut c = chain.lock().expect("chain lock");
                Value::Array(calls.iter().map(|r| rpc::handle(&mut c, r)).collect())
            }
            Ok(one) => {
                let mut c = chain.lock().expect("chain lock");
                rpc::handle(&mut c, &one)
            }
            Err(e) => serde_json::json!({
                "jsonrpc":"2.0","id":Value::Null,
                "error":{"code":-32700,"message":format!("{e}")}
            }),
        };

        let text = reply.to_string();
        let response = tiny_http::Response::from_string(text).with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                .expect("static header"),
        );
        let _ = request.respond(response);
    }
}
