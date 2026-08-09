#!/usr/bin/env bash
# Everything this project claims, checked in one run.
#
# Two tiers. The first needs nothing but the repository and proves the whole generation
# loop against float64. The second needs the 1.2 GB Qwen3-0.6B checkpoint and proves the
# same thing against a real published model.
#
# Usage:
#   ./verify.sh            # tier one only
#   ./verify.sh --full     # both, if models/ has been populated
set -uo pipefail
cd "$(dirname "$0")"

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

pass=0
fail=0

step() {
  local name="$1"; shift
  printf '%-46s' "$name"
  local out
  if out=$("$@" 2>&1); then
    if [ -n "${EXPECT:-}" ] && ! grep -qE "$EXPECT" <<<"$out"; then
      echo "FAIL (ran, but did not report success)"
      fail=$((fail + 1))
      return
    fi
    echo "ok"
    pass=$((pass + 1))
  else
    echo "FAIL"
    echo "$out" | tail -5 | sed 's/^/    /'
    fail=$((fail + 1))
  fi
  unset EXPECT
}

command -v solc >/dev/null || { echo "solc is required; see README"; exit 1; }

echo "tier one — no checkpoint needed"
echo
step "unit and integration tests" cargo test --release --quiet
EXPECT="0 warnings|^$" step "lint" cargo clippy --all-targets --release --quiet -- -D warnings
step "formatting" cargo fmt --all --check

if [ "$FULL" = 1 ]; then
  echo
  echo "tier two — needs models/, see README"
  echo
  if [ -d models/layer0 ]; then
    EXPECT="bit-identical" step "one block, against the Rust reference" \
      cargo run --release --quiet --example onchain_layer -- models/layer0
    EXPECT="bit-identical" step "resumable step, through chain state" \
      cargo run --release --quiet --example onchain_token -- models/layer0
  else
    echo "  models/layer0 missing — skipping block checks"
  fi
  if [ -d models/full ]; then
    EXPECT="matched float64" step "full token, real Qwen3-0.6B" \
      cargo run --release --quiet --example full_token -- models/full
  else
    echo "  models/full missing — skipping the full token"
  fi
  if [ -d models/long ]; then
    EXPECT="matched float64" step "64 positions of context" \
      cargo run --release --quiet --example long_context -- models/long
  else
    echo "  models/long missing — skipping long context"
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "$pass checks passed"
else
  echo "$pass passed, $fail failed"
  exit 1
fi
