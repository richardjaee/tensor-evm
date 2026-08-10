#!/usr/bin/env python3
"""Relay one node's blocks to another through the engine API, and check it agrees.

This is the measurement the project was missing. Everything else shows that a model runs on
a chain and that the arithmetic is reproducible; none of it shows a second client accepting
blocks it did not produce.

Gossip does not do that. A post-merge Ethereum node propagates blocks over peer-to-peer but
only adopts one when a consensus client tells it to, so a follower with no consensus layer
sits at genesis holding valid blocks it has no authority to canonicalise. This script is the
consensus layer, reduced to its one relevant job: hand the follower a payload, let it
re-execute the block with its own EVM, and record whether it says VALID or INVALID.

That verdict is the point. The follower recomputes every transaction — including the matmuls,
which read weights out of its own copy of the state — and compares its state root against
the one in the payload. Agreement means two independent executions produced the same chain.

    python3 tools/engine_relay.py --from 8545 --to 8546 --jwt /tmp/tensor-jwt-8552.hex
"""

import argparse
import base64
import hashlib
import hmac
import json
import time
import urllib.request


def rpc(url, method, params, token=None):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, body.encode(), headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.load(r)
    if "error" in out:
        raise SystemExit(f"{method}: {out['error']}")
    return out["result"]


def jwt(secret_hex):
    """HS256 with an `iat` claim, which is all the engine API checks."""
    secret = bytes.fromhex(secret_hex.strip().removeprefix("0x"))
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=")
    header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    claims = b64(json.dumps({"iat": int(time.time())}).encode())
    signing = header + b"." + claims
    sig = b64(hmac.new(secret, signing, hashlib.sha256).digest())
    return (signing + b"." + sig).decode()


def payload_of(src, block, token_src):
    """An execution payload, rebuilt from a block the other node sealed.

    The transactions have to be the raw signed bytes rather than the objects `eth_getBlock`
    returns, because the follower verifies them itself — which is the entire point.
    """
    raw = [rpc(src, "debug_getRawTransaction", [h]) for h in block["transactions"]]
    return {
        "parentHash": block["parentHash"],
        "feeRecipient": block["miner"],
        "stateRoot": block["stateRoot"],
        "receiptsRoot": block["receiptsRoot"],
        "logsBloom": block["logsBloom"],
        "prevRandao": block["mixHash"],
        "blockNumber": block["number"],
        "gasLimit": block["gasLimit"],
        "gasUsed": block["gasUsed"],
        "timestamp": block["timestamp"],
        "extraData": block["extraData"],
        "baseFeePerGas": block.get("baseFeePerGas", "0x0"),
        "blockHash": block["hash"],
        "transactions": raw,
        "withdrawals": block.get("withdrawals", []),
        "blobGasUsed": block.get("blobGasUsed", "0x0"),
        "excessBlobGas": block.get("excessBlobGas", "0x0"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src_port", type=int, default=8545)
    ap.add_argument("--to", dest="dst_port", type=int, default=8546)
    ap.add_argument("--auth", type=int, default=8552, help="the follower's engine port")
    ap.add_argument("--jwt", required=True)
    a = ap.parse_args()

    src = f"http://127.0.0.1:{a.src_port}"
    dst = f"http://127.0.0.1:{a.dst_port}"
    engine = f"http://127.0.0.1:{a.auth}"
    token = jwt(open(a.jwt).read())

    head = int(rpc(src, "eth_blockNumber", []), 16)
    start = int(rpc(dst, "eth_blockNumber", []), 16)
    print(f"sealer at block {head}, follower at block {start}")
    if head == start:
        raise SystemExit("nothing to relay")

    print(f"\n{'block':>6}  {'txs':>4}  {'gas':>12}  status")
    print("-" * 46)

    for n in range(start + 1, head + 1):
        block = rpc(src, "eth_getBlockByNumber", [hex(n), False])
        payload = payload_of(src, block, token)

        # Prague is active from genesis on this chain, so V4 — which carries the execution
        # requests introduced with it. The empty list is what a block with none looks like.
        res = rpc(
            engine,
            "engine_newPayloadV4",
            [payload, [], block["parentBeaconBlockRoot"], []],
            token,
        )
        status = res["status"]
        print(
            f"{n:>6}  {len(block['transactions']):>4}  "
            f"{int(block['gasUsed'], 16):>12,}  {status}"
        )
        if status != "VALID":
            raise SystemExit(f"follower rejected block {n}: {res}")

        # Telling it which block is canonical is the other half of what a consensus client
        # does; without this the follower has verified the block and still not adopted it.
        rpc(
            engine,
            "engine_forkchoiceUpdatedV3",
            [
                {
                    "headBlockHash": block["hash"],
                    "safeBlockHash": block["hash"],
                    "finalizedBlockHash": block["hash"],
                },
                None,
            ],
            token,
        )

    a_root = rpc(src, "eth_getBlockByNumber", ["latest", False])["stateRoot"]
    b_root = rpc(dst, "eth_getBlockByNumber", ["latest", False])["stateRoot"]
    b_head = int(rpc(dst, "eth_blockNumber", []), 16)

    print(f"\nsealer   block {head}  root {a_root}")
    print(f"follower block {b_head}  root {b_root}")
    if a_root == b_root and b_head == head:
        print("\n  agreed: the follower re-executed every block and reached the same state")
    else:
        raise SystemExit("\n  DIVERGED")


if __name__ == "__main__":
    main()
