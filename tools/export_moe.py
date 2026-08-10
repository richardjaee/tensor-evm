#!/usr/bin/env python3
"""Export a whole Qwen3-MoE checkpoint, with a float64 reference, one layer at a time.

Qwen3-30B-A3B is 61GB. Holding it in float64 to produce a reference would be 244GB, which
is why this streams: a layer is loaded, quantised, written, run in float64, and freed before
the next is touched. Peak memory is one layer — about 6GB — rather than the model.

That works because a transformer is a chain. Running layer `L` for every position needs only
that layer's weights and the residual stream, and the key/value cache a layer builds is used
by nothing else. So the loop is layers on the outside and positions on the inside, which is
the opposite of how a model is usually run and the only way to do this in one pass.

    python3 tools/export_moe.py --model models/q30 --config models/q30/config.json \
        --out models/q30full --positions 5

Writes the layout the chain reads: per-layer attention tensors, every expert laid out expert
by expert so one stride addresses all of them, a router, and what float64 says the model
emits.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from quantize import blob, quantize_rows, read_safetensors

Q32 = 1 << 32


def to_q32(a):
    return np.rint(np.asarray(a, dtype=np.float64) * Q32).astype(np.int64)


def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x) + eps) * w


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="directory of safetensors shards")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--prompt", type=int, nargs="*", default=None)
    ap.add_argument("--layers", type=int, default=0, help="0 means all of them")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    dim = cfg["hidden_size"]
    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    hs = cfg["head_dim"]
    theta = float(cfg["rope_theta"])
    eps = cfg.get("rms_norm_eps", 1e-6)
    E = cfg["num_experts"]
    K = cfg["num_experts_per_tok"]
    hidden = cfg["moe_intermediate_size"]
    L = a.layers or cfg["num_hidden_layers"]
    tied = cfg.get("tie_word_embeddings", False)
    half, group = hs // 2, H // KV
    P = a.positions

    if not cfg.get("norm_topk_prob", True):
        raise SystemExit("norm_topk_prob is false; the chain's routing would not match")

    os.makedirs(a.out, exist_ok=True)

    def write(name, data):
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)

    print(f"Qwen3-MoE: dim {dim}, {L} layers, {E} experts of {hidden}, top {K}")

    # ---- embeddings, and the prompt ----
    t0 = time.time()
    emb = read_safetensors(a.model, {"model.embed_tokens.weight"})["model.embed_tokens.weight"]
    vocab = emb.shape[0]
    ids = a.prompt or [9707, 11, 847, 829, 374][:P]
    ids = (ids * P)[:P]
    xs = emb[ids].astype(np.float64)
    write("embed.bin", to_q32(xs.reshape(-1)).astype("<i8").tobytes())
    print(f"  vocab {vocab}, prompt {ids}  ({time.time()-t0:.0f}s)")
    del emb

    inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hs)
    ang = np.arange(P, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    write("cos.bin", to_q32(cos.reshape(-1)).astype("<i8").tobytes())
    write("sin.bin", to_q32(sin.reshape(-1)).astype("<i8").tobytes())

    # ---- layers, one at a time ----
    x = xs.copy()  # the residual stream, one row a position
    stride = None
    routed = []
    for l in range(L):
        t0 = time.time()
        pre = f"model.layers.{l}."
        names = {
            "input_norm": pre + "input_layernorm.weight",
            "q": pre + "self_attn.q_proj.weight",
            "k": pre + "self_attn.k_proj.weight",
            "v": pre + "self_attn.v_proj.weight",
            "q_norm": pre + "self_attn.q_norm.weight",
            "k_norm": pre + "self_attn.k_norm.weight",
            "o": pre + "self_attn.o_proj.weight",
            "post_norm": pre + "post_attention_layernorm.weight",
            "router": pre + "mlp.gate.weight",
        }
        for e in range(E):
            p = f"{pre}mlp.experts.{e}."
            names[f"e{e}_gate"] = p + "gate_proj.weight"
            names[f"e{e}_up"] = p + "up_proj.weight"
            names[f"e{e}_down"] = p + "down_proj.weight"

        t = read_safetensors(a.model, set(names.values()))
        missing = [n for n in names.values() if n not in t]
        if missing:
            raise SystemExit(f"layer {l}: missing {len(missing)}, first {missing[:2]}")
        W = {key: t[name].astype(np.float64) for key, name in names.items()}
        del t

        # ---- write it in the chain's layout ----
        for key in ("q", "k", "v", "o"):
            qm, sh = quantize_rows(W[key].astype(np.float32))
            write(f"l{l}_{key}.bin", blob(qm, sh))
        for key in ("input_norm", "q_norm", "k_norm", "post_norm"):
            write(f"l{l}_{key}.bin", to_q32(W[key]).astype("<i8").tobytes())
        parts = []
        for e in range(E):
            for key in ("gate", "up", "down"):
                qm, sh = quantize_rows(W[f"e{e}_{key}"].astype(np.float32))
                parts.append(blob(qm, sh))
        experts = b"".join(parts)
        write(f"l{l}_experts.bin", experts)
        del parts
        qm, sh = quantize_rows(W["router"].astype(np.float32))
        write(f"l{l}_router.bin", blob(qm, sh))
        if stride is None:
            stride = len(experts) // E
        del experts

        # ---- the float64 reference for this layer, every position ----
        kc, vc = [], []
        picked = []
        for p in range(P):
            h = x[p].copy()
            n = rmsnorm(h, W["input_norm"], eps)
            q, k, v = W["q"] @ n, W["k"] @ n, W["v"] @ n
            for i in range(H):
                s = slice(i * hs, (i + 1) * hs)
                q[s] = rmsnorm(q[s], W["q_norm"], eps)
                a_, b_ = q[s][:half].copy(), q[s][half:].copy()
                q[s] = np.concatenate([a_ * cos[p] - b_ * sin[p], a_ * sin[p] + b_ * cos[p]])
            for i in range(KV):
                s = slice(i * hs, (i + 1) * hs)
                k[s] = rmsnorm(k[s], W["k_norm"], eps)
                a_, b_ = k[s][:half].copy(), k[s][half:].copy()
                k[s] = np.concatenate([a_ * cos[p] - b_ * sin[p], a_ * sin[p] + b_ * cos[p]])
            kc.append(k)
            vc.append(v)

            attn = np.zeros(H * hs)
            for i in range(H):
                kvh = i // group
                qi = q[i * hs : (i + 1) * hs]
                sc = np.array(
                    [qi @ kc[tt][kvh * hs : (kvh + 1) * hs] / np.sqrt(hs) for tt in range(p + 1)]
                )
                pr = softmax(sc)
                for tt in range(p + 1):
                    attn[i * hs : (i + 1) * hs] += pr[tt] * vc[tt][kvh * hs : (kvh + 1) * hs]

            h = h + W["o"] @ attn
            n = rmsnorm(h, W["post_norm"], eps)

            sc = W["router"] @ n
            order = sorted(range(E), key=lambda i: (-sc[i], i))[:K]
            ex = np.exp(np.array([sc[i] for i in order]) - max(sc[i] for i in order))
            rw = ex / ex.sum()
            for slot, i in enumerate(order):
                g = W[f"e{i}_gate"] @ n
                u = W[f"e{i}_up"] @ n
                h = h + rw[slot] * (W[f"e{i}_down"] @ ((g / (1.0 + np.exp(-g))) * u))
            picked.append(order)
            x[p] = h

        routed.append(picked)
        del W
        peak = float(np.abs(x).max())
        print(
            f"  layer {l:>2}/{L}  {time.time()-t0:5.1f}s  peak residual {peak:8.1f}",
            flush=True,
        )

    # ---- the classifier ----
    t0 = time.time()
    want = {"model.norm.weight"} | ({"lm_head.weight"} if not tied else set())
    t = read_safetensors(a.model, want)
    final_norm = t["model.norm.weight"].astype(np.float64)
    head = (
        t["lm_head.weight"]
        if not tied
        else read_safetensors(a.model, {"model.embed_tokens.weight"})[
            "model.embed_tokens.weight"
        ]
    ).astype(np.float64)
    del t

    write("final_norm.bin", to_q32(final_norm).astype("<i8").tobytes())
    qm, sh = quantize_rows(head.astype(np.float32))
    write("lm_head.bin", blob(qm, sh))

    tokens = [int(np.argmax(head @ rmsnorm(x[p], final_norm, eps))) for p in range(P)]
    write("tokens.bin", np.asarray(tokens, dtype="<u4").tobytes())
    print(f"  classifier ({time.time()-t0:.0f}s)")

    meta = {
        "dim": dim, "hidden": hidden, "n_layers": L, "n_heads": H, "n_kv": KV,
        "head_dim": hs, "vocab": vocab, "positions": P, "rope_theta": theta,
        "prompt_ids": ids, "n_experts": E, "top_k": K, "expert_stride": stride,
        "tied": tied,
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(a.out, "routed.json"), "w") as f:
        json.dump(routed, f)

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"  tokens {tokens}")
    print(f"  expert stride {stride} bytes")
    print(f"  wrote {a.out} ({total/1e9:.1f} GB)")


if __name__ == "__main__":
    main()
