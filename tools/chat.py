#!/usr/bin/env python3
"""Talk to a model running on the chain.

Text goes in, tokens come out of a transformer whose weights are contract code and whose
arithmetic happened inside the EVM. Every token is a transaction, and the gas is on a
receipt anyone can fetch.

Only two things here are off-chain, and both are lookups rather than computation: turning
text into token ids, and turning them back. The inference is not simulated, mocked, or
replayed — there is no path in this script that produces a token without a transaction
having produced it first.

    ./tools/chat.py --prompt "Hello," --generate 8

Wants a node already running and a model exported for enough positions:

    cargo run --release -p tensor-node -- 8545 &
    python3 tools/export_model.py --model models/qwen3-0.6b.safetensors \
        --config models/config.json --out models/chat --prompt-ids 9707 11 ...
"""

import argparse
import json
import os
import re
import subprocess
import sys

from tokenizers import Tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--generate", type=int, default=8)
    ap.add_argument("--raw", action="store_true", help="no chat template; pure autocomplete")
    ap.add_argument("--model", default="models/chat128")
    ap.add_argument("--rpc", default=os.environ.get("RPC", "http://127.0.0.1:8545"))
    ap.add_argument("--chunk-bytes", default="999424")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = os.path.join(root, a.model)
    tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
    meta = json.load(open(os.path.join(model, "meta.json")))

    # Qwen3 is instruction-tuned and expects its turn markers. Without them it behaves like
    # a base model and autocompletes the question instead of answering it — which looked
    # like a weak model and was mostly a missing wrapper. The empty <think> block is how
    # Qwen3 is told to answer directly rather than reason first.
    text = (
        a.prompt
        if a.raw
        else (
            "<|im_start|>user\n"
            f"{a.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    )
    ids = tok.encode(text, add_special_tokens=False).ids
    room = meta["positions"]
    if len(ids) + a.generate > room:
        raise SystemExit(
            f"{len(ids)} prompt tokens plus {a.generate} generated is {len(ids) + a.generate}, "
            f"and this export covers {room} positions. Re-export with more."
        )

    print(f'prompt   "{a.prompt}"')
    print(f"  tokens {ids}")
    print(f"  asking the chain for {a.generate} more\n")

    env = dict(
        os.environ,
        RPC=a.rpc,
        PROMPT_IDS=",".join(str(i) for i in ids),
        GENERATE=str(a.generate),
        CHUNK_BYTES=a.chunk_bytes,
    )
    # Streamed rather than captured. A run takes seconds when the model is already on the
    # chain and a minute when it is not, and buffering the output until the end makes both
    # look identical: frozen. Watching the positions land is also the only way to see that
    # transactions are going through at all.
    proc = subprocess.Popen(
        [os.path.join(root, "target/release/tensor-cli"), model],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    if proc.wait() != 0:
        raise SystemExit("the chain did not finish the generation")
    out = type("Run", (), {"stdout": "".join(lines)})()

    # The tokens the chain emitted, read back out of what it printed. Nothing here can
    # invent one: if the run failed there is no line to parse.
    m = re.search(r"all\s+\[([0-9,\s]+)\]", out.stdout)
    if not m:
        raise SystemExit("no tokens in the output")
    produced = [int(t) for t in m.group(1).split(",")]

    # The continuation starts at the token predicted after the last prompt token. Earlier
    # entries are the model's prediction at each prompt position, which it was not asked for
    # and which are not part of the reply.
    tail = produced[len(ids) - 1 :]
    # <|im_end|> ends the assistant's turn; anything after it is the model starting a new
    # one, which is not part of the answer.
    stop = tok.token_to_id("<|im_end|>")
    if stop in tail:
        tail = tail[: tail.index(stop)]
    reply = tok.decode(tail)
    print(f'\n\033[1mreply\033[0m  "{reply}"')
    print(f'\033[1mwhole\033[0m  "{a.prompt}{reply}"')


if __name__ == "__main__":
    main()
