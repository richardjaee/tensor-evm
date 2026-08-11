//! An Ethereum client, speaking only JSON-RPC.
//!
//! This is what makes the generation it drives worth anything. Everything else in this
//! repository executes the model in the same process that holds the state, which proves the
//! arithmetic but not the chain — a bug shared between the two would be invisible. Here the
//! transactions are built and signed on this side, sent as bytes over HTTP, and the answers
//! come back from a node that shares no memory with the sender.

use alloy_consensus::{SignableTransaction, TxEip1559, TxEnvelope};
use alloy_eips::eip2718::Encodable2718;
use alloy_primitives::{Address, Bytes, TxKind, U256};
use alloy_signer::SignerSync;
use alloy_signer_local::PrivateKeySigner;
use serde_json::{json, Value};

pub struct Client {
    url: String,
    signer: PrivateKeySigner,
    chain_id: u64,
    nonce: u64,
    /// Gas offered per transaction. A token costs more than Ethereum's cap allows, which is
    /// one of the limits this chain raises on purpose.
    pub gas: u64,
    pub sent: u64,
    pub gas_used: u64,
}

impl Client {
    pub fn new(url: &str, key: &str, gas: u64) -> Result<Self, String> {
        let signer: PrivateKeySigner = key.parse().map_err(|e| format!("bad key: {e}"))?;
        let mut c = Self {
            url: url.to_string(),
            signer,
            chain_id: 0,
            nonce: 0,
            gas,
            sent: 0,
            gas_used: 0,
        };
        c.chain_id = u64::from_str_radix(
            c.call_str("eth_chainId", json!([]))?
                .trim_start_matches("0x"),
            16,
        )
        .map_err(|e| format!("chain id: {e}"))?;
        // Read once and tracked locally afterwards. Asking the node before every send would
        // make the client's behaviour depend on how promptly the node updates, which is a
        // race rather than a protocol.
        c.nonce = u64::from_str_radix(
            c.call_str(
                "eth_getTransactionCount",
                json!([format!("{:#x}", c.address()), "latest"]),
            )?
            .trim_start_matches("0x"),
            16,
        )
        .map_err(|e| format!("nonce: {e}"))?;
        Ok(c)
    }

    pub fn address(&self) -> Address {
        self.signer.address()
    }

    pub fn rpc(&self, method: &str, params: Value) -> Result<Value, String> {
        let body = json!({"jsonrpc":"2.0","id":1,"method":method,"params":params});
        let resp: Value = ureq::post(&self.url)
            .set("content-type", "application/json")
            .send_json(&body)
            .map_err(|e| format!("{method}: {e}"))?
            .into_json()
            .map_err(|e| format!("{method}: malformed reply: {e}"))?;
        if let Some(e) = resp.get("error") {
            return Err(format!("{method}: {e}"));
        }
        resp.get("result")
            .cloned()
            .ok_or_else(|| format!("{method}: no result"))
    }

    fn call_str(&self, method: &str, params: Value) -> Result<String, String> {
        self.rpc(method, params)?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{method}: expected a string"))
    }

    /// Signs and submits a transaction, then waits for it to be mined and requires success.
    ///
    /// A reverted transaction is an error rather than a value, because every call this
    /// client makes is one whose failure would invalidate everything after it.
    pub fn send(
        &mut self,
        to: Option<Address>,
        data: Vec<u8>,
        what: &str,
    ) -> Result<Value, String> {
        let tx = TxEip1559 {
            chain_id: self.chain_id,
            nonce: self.nonce,
            gas_limit: self.gas,
            // A real client's pool refuses a transaction that offers nothing, so these are
            // not zero. The development node ignores them and charges no fee either way;
            // what matters is that the same client works against both.
            max_fee_per_gas: 2_000_000_000,
            max_priority_fee_per_gas: 1_000_000_000,
            to: to.map(TxKind::Call).unwrap_or(TxKind::Create),
            value: U256::ZERO,
            input: Bytes::from(data),
            access_list: Default::default(),
        };
        let sig = self
            .signer
            .sign_hash_sync(&tx.signature_hash())
            .map_err(|e| format!("{what}: signing: {e}"))?;
        let env: TxEnvelope = tx.into_signed(sig).into();
        let raw = env.encoded_2718();

        let hash = self.call_str(
            "eth_sendRawTransaction",
            json!([format!("0x{}", hex(&raw))]),
        )?;
        self.nonce += 1;
        self.sent += 1;

        // The development node seals a block per transaction inside the send, so a receipt
        // was always there by the time it returned. A real client mines asynchronously, and
        // reading the receipt immediately gets `null` — which looked exactly like a
        // reverted transaction, because a missing status and a failed one are both "not
        // 0x1". So this waits, and treats absence as "not yet" rather than as failure.
        let mut receipt = Value::Null;
        for _ in 0..600 {
            let r = self.rpc("eth_getTransactionReceipt", json!([hash]))?;
            if r.get("status").is_some() {
                receipt = r;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        if receipt.is_null() {
            return Err(format!("{what}: not mined within a minute ({hash})"));
        }
        let status = receipt
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("0x0");
        if status != "0x1" {
            return Err(format!("{what}: reverted ({hash})"));
        }
        if let Some(g) = receipt.get("gasUsed").and_then(Value::as_str) {
            self.gas_used += u64::from_str_radix(g.trim_start_matches("0x"), 16).unwrap_or(0);
        }
        Ok(receipt)
    }

    /// A read, which costs nothing and changes nothing.
    pub fn call(&self, to: Address, data: Vec<u8>) -> Result<Vec<u8>, String> {
        let out = self.call_str(
            "eth_call",
            json!([{
                "to": format!("{to:#x}"),
                // Without this a call comes from the zero address, and a session that
                // checks its owner refuses it. The development node never noticed because
                // nothing it served was owned.
                "from": format!("{:#x}", self.address()),
                "data": format!("0x{}", hex(&data)),
                "gas": format!("{:#x}", self.gas)
            }, "latest"]),
        )?;
        unhex(&out).ok_or_else(|| "eth_call returned malformed hex".to_string())
    }

    /// Sends a transaction and reports what it returned.
    ///
    /// A receipt does not carry return data. `tensor-node` keeps it and serves it under
    /// `tensor_getOutput`, but a stock Ethereum client does not, and this has to work
    /// against reth. So the value is read with `eth_call` against the state the transaction
    /// is about to be applied to, and then the transaction is sent. The call and the send
    /// see the same state, so the value is the one the transaction produces.
    pub fn send_reading(
        &mut self,
        to: Address,
        data: Vec<u8>,
        what: &str,
    ) -> Result<Vec<u8>, String> {
        let returned = self.call(to, data.clone())?;
        self.send(Some(to), data, what)?;
        Ok(returned)
    }

    /// The return data of a transaction, which an Ethereum receipt does not carry. The node
    /// keeps it and serves it under `tensor_getOutput`, because on a chain built for
    /// auditing, re-running a call to learn what it returned defeats the purpose.
    pub fn output_of(&self, receipt: &Value) -> Result<Vec<u8>, String> {
        let hash = receipt
            .get("transactionHash")
            .and_then(Value::as_str)
            .ok_or("receipt has no hash")?;
        let out = self.call_str("tensor_getOutput", json!([hash]))?;
        unhex(&out).ok_or_else(|| "tensor_getOutput returned malformed hex".to_string())
    }

    /// The state root of block zero, which identifies the chain a model was uploaded to.
    ///
    /// Two chains started from different saved state have different genesis roots, and that
    /// is what stops a note written against one being believed by the other.
    pub fn genesis_root(&self) -> Result<String, String> {
        let b = self.rpc("eth_getBlockByNumber", json!(["0x0", false]))?;
        b.get("stateRoot")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| "block zero has no stateRoot".to_string())
    }

    /// How many bytes of code an account carries.
    ///
    /// Used to recognise a model uploaded by an earlier run. The address alone is not
    /// enough — addresses come from a sender and a nonce, so a different model uploaded in
    /// the same order lands in the same places — and the classifier is stored as a
    /// directory, so its code is a few kilobytes of chunk addresses rather than the whole
    /// table. Cheap to fetch and specific to the model.
    pub fn code_len(&self, at: Address) -> Result<usize, String> {
        let hex = self.call_str("eth_getCode", json!([format!("{at:#x}"), "latest"]))?;
        Ok(hex.trim_start_matches("0x").len() / 2)
    }

    pub fn block_number(&self) -> Result<u64, String> {
        Ok(u64::from_str_radix(
            self.call_str("eth_blockNumber", json!([]))?
                .trim_start_matches("0x"),
            16,
        )
        .unwrap_or(0))
    }

    pub fn state_root(&self) -> Result<String, String> {
        let b = self.rpc("eth_getBlockByNumber", json!(["latest", false]))?;
        b.get("stateRoot")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| "block has no stateRoot".to_string())
    }
}

/// The address a contract-creating transaction produced.
pub fn created(receipt: &Value) -> Result<Address, String> {
    receipt
        .get("contractAddress")
        .and_then(Value::as_str)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| "receipt carries no contract address".to_string())
}

/// The first indexed topic of the first log, which is how `Uploader` reports where it put a
/// chunk.
pub fn first_topic(receipt: &Value) -> Result<Address, String> {
    let t = receipt
        .get("logs")
        .and_then(Value::as_array)
        .and_then(|l| l.first())
        .and_then(|l| l.get("topics"))
        .and_then(Value::as_array)
        .and_then(|t| t.get(1))
        .and_then(Value::as_str)
        .ok_or("no indexed topic in the first log")?;
    let b = unhex(t).ok_or("topic is not hex")?;
    if b.len() != 32 {
        return Err("topic is not a word".into());
    }
    Ok(Address::from_slice(&b[12..]))
}

/// Every log's first indexed topic, as addresses, in the order they were emitted.
///
/// `storeMany` reports one `ChunkStored` per chunk, so a batched upload learns where all of
/// its chunks landed from a single receipt.
pub fn all_topics(receipt: &Value) -> Result<Vec<Address>, String> {
    receipt
        .get("logs")
        .and_then(Value::as_array)
        .ok_or("receipt has no logs")?
        .iter()
        .map(|l| {
            let t = l
                .get("topics")
                .and_then(Value::as_array)
                .and_then(|t| t.get(1))
                .and_then(Value::as_str)
                .ok_or("log has no indexed topic")?;
            let b = unhex(t).ok_or("topic is not hex")?;
            if b.len() != 32 {
                return Err("topic is not a word".to_string());
            }
            Ok(Address::from_slice(&b[12..]))
        })
        .collect()
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn unhex(s: &str) -> Option<Vec<u8>> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

/// Reads a 32-byte word out of return data as an unsigned integer.
pub fn word_at(data: &[u8], i: usize) -> Option<u64> {
    let w = data.get(i * 32..(i + 1) * 32)?;
    Some(u64::from_be_bytes(w[24..].try_into().ok()?))
}
