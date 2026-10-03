#!/usr/bin/env python3
"""Perplexity of the original checkpoint, in float64, as the baseline the integer path is judged against.

Determinism and quality are separate claims. `crates/tensor-model/src/bin/perplexity.rs`
measures what the integer path predicts; this measures what the bf16 weights predict when
nothing is quantised, so the difference between the two numbers is the cost of the
arithmetic rather than of the model.

The forward pass here is batched over positions, where `tools/export_model.py`'s reference
walks them one at a time. Causal attention with a KV cache and masked attention over the
whole sequence are the same function, but "should be" is not evidence, so `--verify`
re-runs a short prompt and prints the per-position argmax for comparison against that
script's output. Run it once when changing either.

    python3 tools/ppl_f64.py --model models/qwen3-0.6b.safetensors \\
        --config models/config.json --tokens models/ppl/tokens.json

Prints mean NLL, perplexity and top-1 accuracy over `len(tokens) - 1` predictions.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantize import read_safetensors  # noqa: E402


def rmsnorm(x, w, eps):
    """Row-wise, matching tools/export_model.py for a single row."""
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps) * w


def rope(v, cos, sin):
    """Split-half rotary, the convention the exporter and the Rust kernel share.

    `v` is [P, hs]; `cos` and `sin` are [P, hs/2].
    """
    half = v.shape[-1] // 2
    a, b = v[:, :half], v[:, half:]
    return np.concatenate([a * cos - b * sin, a * sin + b * cos], axis=-1)


class Lazy:
    """Tensors read from disk one at a time, so a model larger than memory still runs.

    `read_safetensors` widens bf16 into float32, and a 4B checkpoint is 15 GB that way
    before a single float64 copy exists. Reading per tensor and dropping each after use
    holds peak memory to roughly the largest one instead of the whole checkpoint. Slower,
    because the header is re-parsed per access, and the only way this fits on a laptop.
    """

    def __init__(self, path):
        self.path = path
        self._names = set(read_safetensors(path, wanted=set()).keys()) or None

    def __contains__(self, name):
        try:
            return bool(read_safetensors(self.path, wanted={name}))
        except Exception:
            return False

    def __getitem__(self, name):
        got = read_safetensors(self.path, wanted={name})
        if name not in got:
            raise KeyError(name)
        return got[name]


def forward(t, cfg, ids, chunk=8192):
    """Every position's logits, float64 throughout. Returns [P, V].

    `t` is either a dict of tensors or a `Lazy`; both are indexed the same way, so the
    streaming case costs re-reads rather than a second code path.

    The classifier is applied in vocabulary chunks. Promoting a 151936-row embedding to
    float64 in one piece is 3.1 GB for a 4B model, and the result is the same either way
    because each output element is one dot product regardless of how the rows are grouped.
    """
    dim = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    hs = cfg["head_dim"]
    theta = float(cfg["rope_theta"])
    eps = cfg.get("rms_norm_eps", 1e-6)
    tied = cfg.get("tie_word_embeddings", False)
    group = H // KV
    half = hs // 2
    P = len(ids)

    def g(n):
        return t[n].astype(np.float64)

    inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hs)
    ang = np.arange(P, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)

    # The table is indexed in its stored precision and only the rows the sequence touches
    # are promoted. bf16 widens into float32 exactly and float32 into float64 exactly, so
    # this is the same number the whole-table promotion would give, at 1/2000th the memory.
    emb_src = t["model.embed_tokens.weight"]
    x = emb_src[list(ids)].astype(np.float64)  # [P, dim]
    del emb_src

    # Masked once and reused: -inf above the diagonal makes softmax over the whole
    # sequence identical to a cache that has only seen the past.
    mask = np.triu(np.full((P, P), -np.inf), k=1)

    for l in range(L):
        pre = f"model.layers.{l}."
        h = x
        n = rmsnorm(h, g(pre + "input_layernorm.weight"), eps)
        q = n @ g(pre + "self_attn.q_proj.weight").T
        k = n @ g(pre + "self_attn.k_proj.weight").T
        v = n @ g(pre + "self_attn.v_proj.weight").T

        qn = g(pre + "self_attn.q_norm.weight")
        kn = g(pre + "self_attn.k_norm.weight")
        for i in range(H):
            s = slice(i * hs, (i + 1) * hs)
            q[:, s] = rope(rmsnorm(q[:, s], qn, eps), cos, sin)
        for i in range(KV):
            s = slice(i * hs, (i + 1) * hs)
            k[:, s] = rope(rmsnorm(k[:, s], kn, eps), cos, sin)

        attn = np.zeros((P, H * hs), dtype=np.float64)
        for i in range(H):
            kvh = i // group
            qi = q[:, i * hs : (i + 1) * hs]
            ki = k[:, kvh * hs : (kvh + 1) * hs]
            vi = v[:, kvh * hs : (kvh + 1) * hs]
            sc = qi @ ki.T / np.sqrt(hs) + mask
            sc -= sc.max(axis=-1, keepdims=True)
            pr = np.exp(sc)
            pr /= pr.sum(axis=-1, keepdims=True)
            attn[:, i * hs : (i + 1) * hs] = pr @ vi

        h = h + attn @ g(pre + "self_attn.o_proj.weight").T
        n = rmsnorm(h, g(pre + "post_attention_layernorm.weight"), eps)
        gt = n @ g(pre + "mlp.gate_proj.weight").T
        up = n @ g(pre + "mlp.up_proj.weight").T
        x = h + ((gt / (1.0 + np.exp(-gt))) * up) @ g(pre + "mlp.down_proj.weight").T

    x = rmsnorm(x, g("model.norm.weight"), eps)

    head_name = "model.embed_tokens.weight" if tied else "lm_head.weight"
    head_src = t[head_name]
    V = head_src.shape[0]
    logits = np.empty((P, V), dtype=np.float64)
    for s in range(0, V, chunk):
        e = min(s + chunk, V)
        logits[:, s:e] = x @ head_src[s:e].astype(np.float64).T
    return logits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="safetensors file or sharded directory")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", help="JSON file: a list of token ids, or {\"ids\": [...]}")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="run export_model.py's default prompt and print per-position argmax instead",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="read tensors from disk per use; slower, and the only way a 4B model fits",
    )
    a = ap.parse_args()

    cfg = json.load(open(a.config))

    if a.verify:
        ids = [9707, 11, 847, 829, 374]
    else:
        if not a.tokens:
            ap.error("--tokens is required unless --verify")
        raw = json.load(open(a.tokens))
        ids = raw["ids"] if isinstance(raw, dict) else raw
    P = len(ids)
    if P < 2 and not a.verify:
        ap.error(f"perplexity needs at least two tokens, got {P}")

    print(f"float64 baseline")
    print(f"  model           {a.model}")
    print(f"  tokens          {P}")
    sys.stdout.flush()

    print("  reading checkpoint..." if not a.stream else "  streaming checkpoint...")
    sys.stdout.flush()
    t = Lazy(a.model) if a.stream else read_safetensors(a.model)

    print("  running float64 forward...")
    sys.stdout.flush()
    logits = forward(t, cfg, ids)

    if a.verify:
        print(f"\n  prompt {ids}")
        print("  per-position argmax:", [int(r.argmax()) for r in logits])
        last = logits[-1]
        top = np.argsort(last)[::-1][:10]
        print("\n  top tokens by float64 logit (compare against models/export.log):")
        for rank, i in enumerate(top, 1):
            print(f"  {rank:5}. id {int(i):6d}  logit {last[i]:+.4f}")
        return

    # log-softmax in the shifted form, then grade position p against token p+1.
    m = logits.max(axis=-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))
    targets = np.asarray(ids[1:], dtype=np.int64)
    chosen = logits[np.arange(P - 1), targets]
    nll = -(chosen - lse[: P - 1])

    mean_nll = float(nll.mean())
    top1 = int((logits[: P - 1].argmax(axis=-1) == targets).sum())
    print()
    print("float64 path")
    print(f"  mean NLL        {mean_nll:.6f} nats")
    print(f"  perplexity      {np.exp(mean_nll):.4f}")
    print(f"  top-1 accuracy  {100.0 * top1 / (P - 1):.2f}% ({top1}/{P - 1})")


if __name__ == "__main__":
    main()
