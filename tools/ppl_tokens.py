#!/usr/bin/env python3
"""Turn evaluation text into the token ids both perplexity paths score.

Both sides must grade the same sequence or the comparison means nothing, so the ids are
written once, here, and read by `tools/ppl_f64.py` and the `perplexity` binary alike —
the latter through `export_model.py --prompt-ids`, which bakes them into the export.

WikiText-2 is the corpus the comparable projects report against, which is the only reason
to prefer it; any text file works.

    python3 tools/ppl_tokens.py --wikitext --limit 512 --out models/ppl/tokens.json
    python3 tools/ppl_tokens.py --text notes.txt --limit 512 --out models/ppl/tokens.json

`--limit` matters more than it looks. The export writes one embedding row and one rotary
row per position, and the integer path holds a KV cache for the whole sequence, so a long
evaluation is a large artifact and a quadratic attention cost on both sides. 512 tokens is
enough to separate a working model from a broken one; it is not a publication-grade
corpus, and the printed note says so.
"""

import argparse
import json
import os
import sys
import urllib.request

WIKITEXT = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/"
    "wikitext-2/valid.txt"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="a local text file")
    src.add_argument("--wikitext", action="store_true", help="fetch WikiText-2 validation")
    ap.add_argument(
        "--tokenizer",
        default="models/chat/tokenizer.json",
        help="HuggingFace tokenizer.json for the model being evaluated",
    )
    ap.add_argument("--limit", type=int, default=512, help="tokens to keep")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.wikitext:
        print(f"fetching {WIKITEXT}")
        with urllib.request.urlopen(WIKITEXT, timeout=60) as r:
            text = r.read().decode("utf-8")
        source = "wikitext-2 validation"
    else:
        text = open(a.text, encoding="utf-8").read()
        source = a.text

    # Blank lines and section headers are most of WikiText's raw bytes and none of its
    # language; keeping them would measure how well the model predicts " = = =".
    lines = [l.strip() for l in text.splitlines()]
    text = " ".join(l for l in lines if l and not l.startswith("="))
    if not text:
        sys.exit(f"{source}: no usable text after filtering")

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("pip install tokenizers")

    tok = Tokenizer.from_file(a.tokenizer)
    ids = tok.encode(text).ids[: a.limit]
    if len(ids) < 2:
        sys.exit(f"{source}: only {len(ids)} tokens")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"source": source, "tokenizer": a.tokenizer, "ids": ids}, f)

    print(f"  source     {source}")
    print(f"  tokenizer  {a.tokenizer}")
    print(f"  tokens     {len(ids)} (scores {len(ids) - 1} predictions)")
    print(f"  wrote      {a.out}")
    if len(ids) < 2048:
        print()
        print("  Note: this is a short sample. It separates a working model from a broken")
        print("  one, but report the token count alongside any perplexity taken from it.")
    print()
    print("  Next, bake the same ids into an export so both sides score one sequence:")
    print(f"    python3 tools/export_model.py --model <ckpt> --config <cfg> \\")
    print(f"      --out models/ppl --prompt-ids $(python3 -c \"import json;print(' '.join(map(str,json.load(open('{a.out}'))['ids'])))\")")


if __name__ == "__main__":
    main()
