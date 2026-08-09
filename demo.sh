#!/usr/bin/env bash
# A model weight uploaded and multiplied by, through real transactions on a running chain.
#
# Everything else in this repository drives revm from inside a Rust process. This drives a
# node over JSON-RPC with the same tools anyone would use on any EVM chain — `cast` signing
# real transactions, blocks being sealed, state surviving between them, and the result read
# back afterwards.
#
#   cargo run --release -p tensor-node &
#   ./demo.sh
set -euo pipefail
cd "$(dirname "$0")"

RPC=${RPC:-http://127.0.0.1:8545}
# The account every development chain funds first; the key is public by design.
KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
SEND="cast send --rpc-url $RPC --private-key $KEY --json"

command -v cast >/dev/null || { echo "needs foundry's cast"; exit 1; }
cast chain-id --rpc-url "$RPC" >/dev/null 2>&1 || { echo "no node at $RPC"; exit 1; }

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

say "chain"
printf '  id          %s\n' "$(cast chain-id --rpc-url $RPC)"
printf '  block       %s\n' "$(cast block-number --rpc-url $RPC)"

# ---- a tiny weight matrix, in the format the precompile reads ----
# Eight rows of eight int16 weights, each row preceded by its power-of-two scale. Small
# enough to read by eye; the format is exactly what a 31.5MB layer uses.
say "building an 8x8 tensor"
BLOB=$(python3 - <<'PY'
rows, cols = 8, 8
scales = bytes([20] * rows)
vals = b"".join(
    int(((r * cols + c) % 17) - 8).to_bytes(2, "big", signed=True)
    for r in range(rows) for c in range(cols)
)
print((scales + vals).hex())
PY
)
printf '  %s bytes\n' $(( ${#BLOB} / 2 ))

say "deploying the uploader"
UP_BIN=$(solc --optimize --bin contracts/src/WeightStore.sol 2>/dev/null | awk '/:Uploader/{f=1;next} f&&/^[0-9a-f]+$/{print;exit}')
UPLOADER=$($SEND --create "0x$UP_BIN" | python3 -c 'import json,sys; print(json.load(sys.stdin)["contractAddress"])')
printf '  %s\n' "$UPLOADER"

say "storing the tensor as contract code"
CHUNK=$($SEND "$UPLOADER" "store(bytes)" "0x$BLOB" \
  | python3 -c 'import json,sys; r=json.load(sys.stdin); print(int(r["logs"][0]["topics"][1],16).to_bytes(20,"big").hex())')
printf '  0x%s\n' "$CHUNK"
CODE=$(cast code 0x$CHUNK --rpc-url $RPC); printf "  code on chain: %s bytes\n" $(( (${#CODE} - 2) / 2 ))

say "deploying the registry"
REG_BIN=$(solc --optimize --bin contracts/src/ModelRegistry.sol 2>/dev/null | awk '/:ModelRegistry/{f=1;next} f&&/^[0-9a-f]+$/{print;exit}')
REGISTRY=$($SEND --create "0x$REG_BIN" | python3 -c 'import json,sys; print(json.load(sys.stdin)["contractAddress"])')
printf '  %s\n' "$REGISTRY"

say "registering the model"
$SEND "$REGISTRY" \
  "registerModel(uint256,address,(uint16,uint16,uint16,uint16,uint16,uint16,uint32,uint32,uint32))" \
  1 "0x$CHUNK" "(8,8,1,1,1,8,8,1000000,0)" >/dev/null
printf '  weights hash committed: %s\n' \
  "$(cast call $REGISTRY 'verifyWeights(uint256)(bool)' 1 --rpc-url $RPC)"

say "multiplying by it, through the precompile"
# eight Q32 activations: 1.0, 2.0, ... 8.0
ACT="[$(python3 -c 'print(",".join(str((i+1)<<32) for i in range(8)))')]"
OUT=$(cast call "$REGISTRY" \
  "projectStateless(uint256,uint32,uint32,uint32,int64[])(int64[])" \
  1 8 8 0 "$ACT" --rpc-url "$RPC")
printf '  %s\n' "$OUT"

say "what the reference says"
python3 - "$OUT" <<'PY'
import sys, re
rows, cols = 8, 8
w = [[((r*cols+c) % 17) - 8 for c in range(cols)] for r in range(rows)]
x = [(i+1) << 32 for i in range(cols)]
want = [sum(w[r][c]*x[c] for c in range(cols)) >> 20 for r in range(rows)]
# cast annotates each value with a scientific-notation gloss — "688128 [6.881e5]" — so
# each entry is read up to its bracket rather than by scanning for digits, which would
# also collect the digits inside the gloss.
entries = sys.argv[1].strip().strip("[]").split(",")
got = [int(e.strip().split()[0]) for e in entries if e.strip()]
print("  " + ", ".join(str(v) for v in want))
print("\n  match:", "yes" if got == want else f"NO — chain gave {got}")
sys.exit(0 if got == want else 1)
PY

say "blocks sealed"
printf '  %s\n' "$(cast block-number --rpc-url $RPC)"
