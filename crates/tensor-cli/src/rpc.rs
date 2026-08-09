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
            max_fee_per_gas: 0,
            max_priority_fee_per_gas: 0,
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

        let receipt = self.rpc("eth_getTransactionReceipt", json!([hash]))?;
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
