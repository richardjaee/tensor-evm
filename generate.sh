#!/usr/bin/env bash
# A model uploaded to a live chain, generating tokens, then re-executed independently.
#
# This is the whole design in one script. A node is started; a client that shares no memory
# with it uploads a transformer as contract code and drives a generation over JSON-RPC; the
# tokens are checked against what float64 says the model emits; and then the transactions
# the node recorded are replayed from genesis to confirm a second execution reaches exactly
# the same state.
#
#   ./generate.sh
set -euo pipefail
cd "$(dirname "$0")"

PORT=${PORT:-8547}
JOURNAL=$(mktemp -t tensor-journal)
# `wait` after `kill` so bash reaps the job quietly rather than announcing it as
# Terminated, which reads like a failure at the end of a successful run.
trap 'kill "${NODE_PID:-}" 2>/dev/null || true; wait "${NODE_PID:-}" 2>/dev/null || true; rm -f "$JOURNAL"' EXIT

command -v solc >/dev/null || { echo "needs solc"; exit 1; }

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

cargo build --release -q -p tensor-node -p tensor-cli

say "starting a node"
./target/release/tensor-node "$PORT" --journal "$JOURNAL" > /dev/null 2>&1 &
NODE_PID=$!
for _ in $(seq 1 50); do
  curl -s -o /dev/null -X POST -H 'content-type: application/json' \
    --data '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}' \
    "http://127.0.0.1:$PORT" && break
  sleep 0.2
done
printf '  pid %s on port %s\n' "$NODE_PID" "$PORT"

RPC="http://127.0.0.1:$PORT" ./target/release/tensor-cli "${1:-testdata/tiny}"

# The claim the chain exists to support: somebody else can check the work. The journal holds
# the signed transactions, so replaying it verifies each signature and re-runs every token.
say "re-executing what the node recorded"
./target/release/tensor-node --replay "$JOURNAL" | tail -8
