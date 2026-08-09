//! Enough JSON-RPC for the standard tools to work.
//!
//! The point of speaking Ethereum's RPC rather than something bespoke is that nothing has
//! to be written twice. `cast`, `forge`, MetaMask and a block explorer all work against
//! this because only the precompile set differs from any other EVM chain — the interface a
//! chain presents is where compatibility lives, not the interpreter behind it.

use crate::chain::{Chain, CHAIN_ID, GAS_LIMIT};
use alloy_consensus::{transaction::SignerRecoverable, Transaction, TxEnvelope};
use alloy_rlp::Decodable;
use revm::context::TxEnv;
use revm::primitives::{Address, Bytes, TxKind, B256, U256};
use serde_json::{json, Value};

fn hex_u64(v: u64) -> String {
    format!("0x{v:x}")
}

fn hex_u256(v: U256) -> String {
    format!("0x{v:x}")
}

fn parse_u64(v: &Value) -> Option<u64> {
    let s = v.as_str()?;
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

fn parse_addr(v: &Value) -> Option<Address> {
    let s = v.as_str()?.trim_start_matches("0x");
    let b = hex_bytes(s)?;
    (b.len() == 20).then(|| Address::from_slice(&b))
}

fn hex_bytes(s: &str) -> Option<Vec<u8>> {
    let s = s.trim_start_matches("0x");
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

fn err(id: &Value, code: i64, msg: &str) -> Value {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":msg}})
}

fn ok(id: &Value, result: Value) -> Value {
    json!({"jsonrpc":"2.0","id":id,"result":result})
}

/// Builds the transaction environment an `eth_call` object describes.
///
/// Defaults are generous rather than realistic: a caller probing a view function should
/// not have to know this chain's gas limit, and nothing is being paid for.
fn call_env(o: &Value) -> TxEnv {
    TxEnv {
        caller: o
            .get("from")
            .and_then(parse_addr)
            .unwrap_or(Address::with_last_byte(1)),
        kind: o
            .get("to")
            .and_then(parse_addr)
            .map(TxKind::Call)
            .unwrap_or(TxKind::Create),
        data: o
            .get("data")
            .or_else(|| o.get("input"))
            .and_then(|d| d.as_str())
            .and_then(hex_bytes)
            .map(Bytes::from)
            .unwrap_or_default(),
        value: o
            .get("value")
            .and_then(parse_u64)
            .map(U256::from)
            .unwrap_or(U256::ZERO),
        gas_limit: o.get("gas").and_then(parse_u64).unwrap_or(GAS_LIMIT),
        gas_price: 0,
        chain_id: Some(CHAIN_ID),
        ..Default::default()
    }
}

fn block_json(chain: &Chain, n: u64, full: bool) -> Value {
    let b = &chain.blocks[n as usize];
    let txs: Vec<Value> = if full {
        b.transactions
            .iter()
            .map(|h| json!({"hash": format!("{h:#x}")}))
            .collect()
    } else {
        b.transactions
            .iter()
            .map(|h| json!(format!("{h:#x}")))
            .collect()
    };
    // The whole header, including the roots. A client deserialises a block into a typed
    // struct and refuses one with fields missing, so an incomplete header reads as a
    // malformed node rather than as a simplified one. The roots are zero because nothing
    // here builds tries — which is exactly the kind of shortcut that makes this a
    // development chain rather than a client.
    const EMPTY_ROOT: &str = "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421";
    json!({
        "number": hex_u64(b.number),
        "hash": format!("{:#x}", b.hash),
        "parentHash": format!("{:#x}", b.parent),
        "nonce": "0x0000000000000000",
        "mixHash": format!("0x{}", "0".repeat(64)),
        "stateRoot": format!("0x{}", "0".repeat(64)),
        "transactionsRoot": EMPTY_ROOT,
        "receiptsRoot": EMPTY_ROOT,
        "timestamp": hex_u64(b.timestamp),
        "gasUsed": hex_u64(b.gas_used),
        "gasLimit": hex_u64(GAS_LIMIT),
        "miner": "0x0000000000000000000000000000000000000000",
        "difficulty": "0x0",
        "totalDifficulty": "0x0",
        "extraData": "0x",
        "size": "0x200",
        "baseFeePerGas": "0x0",
        "logsBloom": format!("0x{}", "0".repeat(512)),
        "sha3Uncles": "0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347",
        "transactions": txs,
        "uncles": [],
    })
}

pub fn handle(chain: &mut Chain, req: &Value) -> Value {
    let id = req.get("id").cloned().unwrap_or(json!(1));
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = req
        .get("params")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();

    match method {
        "eth_chainId" => ok(&id, json!(hex_u64(CHAIN_ID))),
        "net_version" => ok(&id, json!(CHAIN_ID.to_string())),
        "eth_gasPrice" | "eth_maxPriorityFeePerGas" => ok(&id, json!("0x0")),
        "eth_accounts" => ok(&id, json!([])),
        "web3_clientVersion" => ok(&id, json!("tensor-node/0.1.0")),
        "eth_syncing" => ok(&id, json!(false)),
        "eth_blockNumber" => ok(&id, json!(hex_u64(chain.head().number))),

        // Fee history with every field zero. There is no fee market here — gas is metered
        // so the cost of a token is measurable, not so anyone bids for space — but wallets
        // and `cast` ask before every send, and a missing method reads as a broken node.
        "eth_feeHistory" => {
            let n = params.first().and_then(parse_u64).unwrap_or(1).max(1) as usize;
            ok(
                &id,
                json!({
                    "oldestBlock": hex_u64(chain.head().number.saturating_sub(n as u64)),
                    "baseFeePerGas": vec!["0x0"; n + 1],
                    "gasUsedRatio": vec![0.0f64; n],
                    "reward": vec![vec!["0x0"]; n],
                }),
            )
        }

        "eth_getBalance" => match params.first().and_then(parse_addr) {
            Some(a) => ok(&id, json!(hex_u256(chain.balance(a)))),
            None => err(&id, -32602, "expected an address"),
        },
        "eth_getTransactionCount" => match params.first().and_then(parse_addr) {
            Some(a) => ok(&id, json!(hex_u64(chain.nonce(a)))),
            None => err(&id, -32602, "expected an address"),
        },
        "eth_getCode" => match params.first().and_then(parse_addr) {
            Some(a) => ok(&id, json!(format!("0x{}", hex::encode(chain.code(a))))),
            None => err(&id, -32602, "expected an address"),
        },

        "eth_getBlockByNumber" => {
            let n = match params.first().and_then(|v| v.as_str()) {
                Some("latest") | Some("pending") | None => chain.head().number,
                Some("earliest") => 0,
                Some(_) => match params.first().and_then(parse_u64) {
                    Some(n) => n,
                    None => return err(&id, -32602, "bad block number"),
                },
            };
            if n as usize >= chain.blocks.len() {
                return ok(&id, Value::Null);
            }
            let full = params.get(1).and_then(|v| v.as_bool()).unwrap_or(false);
            ok(&id, block_json(chain, n, full))
        }

        "eth_call" => {
            let Some(o) = params.first() else {
                return err(&id, -32602, "expected a call object");
            };
            match chain.simulate(call_env(o)) {
                // A revert carries its reason in the return data, and callers decode it —
                // so it travels as an error with the data attached rather than as a result.
                Ok((out, _, true)) => ok(&id, json!(format!("0x{}", hex::encode(out)))),
                Ok((out, _, false)) => json!({
                    "jsonrpc":"2.0","id":id,
                    "error":{"code":3,"message":"execution reverted",
                             "data":format!("0x{}", hex::encode(out))}
                }),
                Err(e) => err(&id, -32000, &e),
            }
        }

        "eth_estimateGas" => {
            let Some(o) = params.first() else {
                return err(&id, -32602, "expected a call object");
            };
            match chain.simulate(call_env(o)) {
                Ok((_, gas, _)) => ok(&id, json!(hex_u64(gas + gas / 4))),
                Err(e) => err(&id, -32000, &e),
            }
        }

        "eth_sendRawTransaction" => {
            let Some(raw) = params.first().and_then(|v| v.as_str()).and_then(hex_bytes) else {
                return err(&id, -32602, "expected raw transaction bytes");
            };
            let Ok(env) = TxEnvelope::decode(&mut raw.as_slice()) else {
                return err(&id, -32602, "could not decode the transaction");
            };
            let Ok(from) = env.recover_signer() else {
                return err(&id, -32000, "could not recover the sender");
            };
            let hash = *env.hash();
            // The envelope implements `Transaction` across all of its variants, so a
            // legacy, 2930, 1559 or 4844 transaction is read the same way.
            let tx = TxEnv {
                caller: from,
                kind: env.kind(),
                data: env.input().clone(),
                value: env.value(),
                gas_limit: env.gas_limit(),
                gas_price: 0,
                nonce: env.nonce(),
                chain_id: Some(CHAIN_ID),
                ..Default::default()
            };
            match chain.apply(tx, hash) {
                Ok(_) => ok(&id, json!(format!("{hash:#x}"))),
                Err(e) => err(&id, -32000, &e),
            }
        }

        "eth_getTransactionByHash" => {
            let Some(h) = params
                .first()
                .and_then(|v| v.as_str())
                .and_then(hex_bytes)
                .filter(|b| b.len() == 32)
                .map(|b| B256::from_slice(&b))
            else {
                return err(&id, -32602, "expected a transaction hash");
            };
            match chain.receipts.iter().find(|r| r.hash == h) {
                Some(r) => ok(
                    &id,
                    json!({
                        "hash": format!("{:#x}", r.hash),
                        "blockNumber": hex_u64(r.block),
                        "blockHash": format!("{:#x}", chain.blocks[r.block as usize].hash),
                        "from": format!("{:#x}", r.from),
                        "to": r.to.map(|a| format!("{a:#x}")),
                        "gas": hex_u64(r.gas_used),
                        "gasPrice": "0x0",
                        "value": "0x0",
                        "nonce": "0x0",
                        "input": "0x",
                        "transactionIndex": "0x0",
                        "type": "0x2",
                    }),
                ),
                None => ok(&id, Value::Null),
            }
        }

        "eth_getTransactionReceipt" => {
            let Some(h) = params
                .first()
                .and_then(|v| v.as_str())
                .and_then(hex_bytes)
                .filter(|b| b.len() == 32)
                .map(|b| B256::from_slice(&b))
            else {
                return err(&id, -32602, "expected a transaction hash");
            };
            let Some(r) = chain.receipts.iter().find(|r| r.hash == h) else {
                return ok(&id, Value::Null);
            };
            let logs: Vec<Value> = r
                .logs
                .iter()
                .enumerate()
                .map(|(i, (addr, topics, data))| {
                    json!({
                        "address": format!("{addr:#x}"),
                        "topics": topics.iter().map(|t| format!("{t:#x}")).collect::<Vec<_>>(),
                        "data": format!("0x{}", hex::encode(data)),
                        "blockNumber": hex_u64(r.block),
                        "blockHash": format!("{:#x}", chain.blocks[r.block as usize].hash),
                        "transactionHash": format!("{:#x}", r.hash),
                        "transactionIndex": "0x0",
                        "logIndex": hex_u64(i as u64),
                        "removed": false,
                    })
                })
                .collect();
            ok(
                &id,
                json!({
                    "transactionHash": format!("{:#x}", r.hash),
                    "blockNumber": hex_u64(r.block),
                    "blockHash": format!("{:#x}", chain.blocks[r.block as usize].hash),
                    "from": format!("{:#x}", r.from),
                    "to": r.to.map(|a| format!("{a:#x}")),
                    "contractAddress": r.created.map(|a| format!("{a:#x}")),
                    "transactionIndex": "0x0",
                    "gasUsed": hex_u64(r.gas_used),
                    "cumulativeGasUsed": hex_u64(r.gas_used),
                    "status": if r.success { "0x1" } else { "0x0" },
                    "logs": logs,
                    "logsBloom": format!("0x{}", "0".repeat(512)),
                    "type": "0x2",
                    "effectiveGasPrice": "0x0",
                }),
            )
        }

        // Not an Ethereum method. A receipt does not carry return data, because on
        // Ethereum a caller who wants it re-runs the call — which is exactly the thing a
        // chain built for auditing should not require, since re-running is what the record
        // exists to make unnecessary.
        "tensor_getOutput" => {
            let Some(h) = params
                .first()
                .and_then(|v| v.as_str())
                .and_then(hex_bytes)
                .filter(|b| b.len() == 32)
                .map(|b| B256::from_slice(&b))
            else {
                return err(&id, -32602, "expected a transaction hash");
            };
            match chain.receipts.iter().find(|r| r.hash == h) {
                Some(r) => ok(&id, json!(format!("0x{}", hex::encode(&r.output)))),
                None => ok(&id, Value::Null),
            }
        }

        _ => err(&id, -32601, &format!("{method} is not implemented")),
    }
}

/// Hex encoding, kept here rather than pulled in as a dependency for four lines of work.
mod hex {
    pub fn encode(b: impl AsRef<[u8]>) -> String {
        b.as_ref().iter().map(|x| format!("{x:02x}")).collect()
    }
}
