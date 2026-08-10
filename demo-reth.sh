#!/usr/bin/env bash
# A language model generating tokens on reth — a real Ethereum client.
#
# Starts a reth node carrying the tensor precompiles, uploads a transformer to it as
# contract code, and generates tokens by signed transactions over JSON-RPC. The tokens are
# checked against what a float64 implementation of the same model produces.
#
# Everything that makes this a chain is reth's: the network, the transaction pool, the
# payload builder, the consensus engine. The only substitution is the executor.
#
#   ./demo-reth.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-8545}
trap 'kill "${NODE:-}" 2>/dev/null || true; wait "${NODE:-}" 2>/dev/null || true' EXIT

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

say "building"
cargo build --release -q -p tensor-cli
(cd crates/tensor-reth && cargo build --release -q)

say "starting a reth node"
crates/tensor-reth/target/release/tensor-reth --rpc-port "$PORT" > /tmp/tensor-reth.log 2>&1 &
NODE=$!
for _ in $(seq 1 90); do
  curl -s -o /dev/null -X POST -H 'content-type: application/json' \
    --data '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}' \
    "http://127.0.0.1:$PORT" && break
  sleep 1
done
printf '  reth pid %s, chain %s, RPC on %s\n' "$NODE" \
  "$(curl -s -X POST -H 'content-type: application/json' \
     --data '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}' \
     "http://127.0.0.1:$PORT" | sed 's/.*"result":"\([^"]*\)".*/\1/')" "$PORT"

# One layer a transaction: a token exceeds EIP-7825's 2^24 gas cap, and reth's pool enforces
# that itself, before execution. Splitting a token across transactions is what the
# granularity knob is for, and it must not change the token.
RPC="http://127.0.0.1:$PORT" GAS=16000000 MAX_LAYERS=1 CHUNK_BYTES=8192 \
  ./target/release/tensor-cli "${1:-testdata/tiny}"
