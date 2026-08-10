#!/usr/bin/env bash
# Two reth nodes, and a second one verifying inference it did not produce.
#
# One node seals blocks; a model is uploaded to it and generates tokens. The other node is
# handed those blocks through the engine API — which is what a consensus client does — and
# re-executes every one of them with its own EVM, against its own copy of the state.
#
# Its verdict on each block is VALID or INVALID, decided by whether its own execution
# produces the state root the block claims. Agreement at the end is two independent
# executions of a language model reaching the same state.
#
#   ./demo-network.sh
set -euo pipefail
cd "$(dirname "$0")"

trap 'kill ${A:-} ${B:-} 2>/dev/null || true; wait ${A:-} ${B:-} 2>/dev/null || true' EXIT
say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

cargo build --release -q -p tensor-cli
(cd crates/tensor-reth && cargo build --release -q)
RETH=crates/tensor-reth/target/release/tensor-reth

say "starting the sealer"
$RETH --rpc-port 8545 --p2p-port 30303 --auth-port 8551 > /tmp/sealer.log 2>&1 &
A=$!
until cast chain-id --rpc-url http://127.0.0.1:8545 >/dev/null 2>&1; do sleep 1; done
ENODE=$(grep -o "enode://[^ ]*" /tmp/sealer.log | head -1)
printf '  pid %s, rpc 8545\n' "$A"

say "starting a follower, which seals nothing"
$RETH --follow --rpc-port 8546 --p2p-port 30304 --auth-port 8552 --peer "$ENODE" \
  > /tmp/follower.log 2>&1 &
B=$!
until cast chain-id --rpc-url http://127.0.0.1:8546 >/dev/null 2>&1; do sleep 1; done
printf '  pid %s, rpc 8546\n' "$B"

say "generating tokens on the sealer"
RPC=http://127.0.0.1:8545 GAS=16000000 MAX_LAYERS=1 CHUNK_BYTES=999424 \
  ./target/release/tensor-cli "${1:-testdata/tiny}" | tail -14

say "the follower re-executes what it was sent"
python3 tools/engine_relay.py --from 8545 --to 8546 --auth 8552 \
  --jwt /tmp/tensor-jwt-8552.hex | tail -8
