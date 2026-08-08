#!/usr/bin/env python3
"""Export a whole Qwen3 model, plus float64 logits, for the full forward pass.

One layer was checked in isolation. This is the rest: every layer, the embedding, the final
norm, and the classifier, together with what float64 says the logits should be for a given
prompt. A layer being right does not make twenty-eight of them right — errors that cancel
at one position can compound across depth, and only running the whole thing shows whether
they do.

The embedding is a lookup rather than a multiply, so it is exported dequantised into Q32
directly: there is no matmul for the quantiser's row scales to ride along with.
"""

import argparse
import json
import os
import struct
import sys

import numpy as np

from quantize import blob, quantize_rows, read_safetensors

Q32 = 1 << 32


def to_q32(a):
    return np.rint(np.asarray(a, dtype=np.float64) * Q32).astype(np.int64)


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean() + eps) * w


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--prompt-ids",
        type=int,
        nargs="+",
        default=[9707, 11, 847, 829, 374],
        help="token ids to run through the model",
    )
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    hs = cfg["head_dim"]
    V = cfg["vocab_size"]
    theta = float(cfg["rope_theta"])
    eps = cfg.get("rms_norm_eps", 1e-6)
    tied = cfg.get("tie_word_embeddings", False)

    ids = a.prompt_ids
    P = len(ids)
    print(f"Qwen3: dim {dim}, hidden {hidden}, layers {L}, heads {H}, kv {KV}, "
          f"head_dim {hs}, vocab {V}, tied {tied}")
    print(f"  prompt: {P} tokens {ids}")

    print("  reading checkpoint...")
    t = read_safetensors(a.model)
    g = lambda n: t[n].astype(np.float64)

    os.makedirs(a.out, exist_ok=True)

    def write(name, data):
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)

    # ---- rotary tables ----
    half = hs // 2
    inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hs)
    ang = np.arange(P, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    write("cos.bin", to_q32(cos.reshape(-1)).astype("<i8").tobytes())
    write("sin.bin", to_q32(sin.reshape(-1)).astype("<i8").tobytes())

    # ---- embedding rows for this prompt, in Q32 ----
    # Only the rows the prompt touches. The full table is 311 MB quantised and the forward
    # pass reads five rows of it; exporting all of it would make the artifact unusable for
    # no gain.
    emb = g("model.embed_tokens.weight")
    write("embed.bin", to_q32(emb[ids].reshape(-1)).astype("<i8").tobytes())

    # ---- per-layer weights ----
    print(f"  quantising {L} layers...")
    for l in range(L):
        p = f"model.layers.{l}."
        for key, name in [
            ("q", "self_attn.q_proj"),
            ("k", "self_attn.k_proj"),
            ("v", "self_attn.v_proj"),
            ("o", "self_attn.o_proj"),
            ("gate", "mlp.gate_proj"),
            ("up", "mlp.up_proj"),
            ("down", "mlp.down_proj"),
        ]:
            qm, sh = quantize_rows(g(p + name + ".weight").astype(np.float32))
            write(f"l{l}_{key}.bin", blob(qm, sh))
        for key, name in [
            ("input_norm", "input_layernorm"),
            ("post_norm", "post_attention_layernorm"),
            ("q_norm", "self_attn.q_norm"),
            ("k_norm", "self_attn.k_norm"),
        ]:
            write(f"l{l}_{key}.bin", to_q32(g(p + name + ".weight")).astype("<i8").tobytes())

    write("final_norm.bin", to_q32(g("model.norm.weight")).astype("<i8").tobytes())

    head = g("lm_head.weight") if "lm_head.weight" in t else emb
    qm, sh = quantize_rows(head.astype(np.float32))
    write("lm_head.bin", blob(qm, sh))

    # ---- float64 reference ----
    print("  running float64 reference...")
    group = H // KV
    per_position = []
    kc = [[] for _ in range(L)]
    vc = [[] for _ in range(L)]
    logits = None

    for p_ix in range(P):
        x = emb[ids[p_ix]].copy()
        for l in range(L):
            pre = f"model.layers.{l}."
            h = x.copy()
            n = rmsnorm(h, g(pre + "input_layernorm.weight"), eps)
            q = g(pre + "self_attn.q_proj.weight") @ n
            k = g(pre + "self_attn.k_proj.weight") @ n
            v = g(pre + "self_attn.v_proj.weight") @ n

            qn = g(pre + "self_attn.q_norm.weight")
            kn = g(pre + "self_attn.k_norm.weight")
            for i in range(H):
                s = slice(i * hs, (i + 1) * hs)
                q[s] = rmsnorm(q[s], qn, eps)
                a_, b_ = q[s][:half].copy(), q[s][half:].copy()
                q[s] = np.concatenate(
                    [a_ * cos[p_ix] - b_ * sin[p_ix], a_ * sin[p_ix] + b_ * cos[p_ix]]
                )
            for i in range(KV):
                s = slice(i * hs, (i + 1) * hs)
                k[s] = rmsnorm(k[s], kn, eps)
                a_, b_ = k[s][:half].copy(), k[s][half:].copy()
                k[s] = np.concatenate(
                    [a_ * cos[p_ix] - b_ * sin[p_ix], a_ * sin[p_ix] + b_ * cos[p_ix]]
                )
            kc[l].append(k)
            vc[l].append(v)

            attn = np.zeros(H * hs)
            for i in range(H):
                kvh = i // group
                qi = q[i * hs : (i + 1) * hs]
                sc = np.array(
                    [
                        qi @ kc[l][tt][kvh * hs : (kvh + 1) * hs] / np.sqrt(hs)
                        for tt in range(p_ix + 1)
                    ]
                )
                pr = softmax(sc)
                for tt in range(p_ix + 1):
                    attn[i * hs : (i + 1) * hs] += pr[tt] * vc[l][tt][kvh * hs : (kvh + 1) * hs]

            h = h + g(pre + "self_attn.o_proj.weight") @ attn
            n = rmsnorm(h, g(pre + "post_attention_layernorm.weight"), eps)
            gt = g(pre + "mlp.gate_proj.weight") @ n
            up = g(pre + "mlp.up_proj.weight") @ n
            h = h + g(pre + "mlp.down_proj.weight") @ ((gt / (1.0 + np.exp(-gt))) * up)
            x = h
        # Every position, not just the last. A chain that emits the right final token
        # having taken a wrong path to it would look identical from the outside, and only
        # per-position tokens distinguish the two.
        lg = head @ rmsnorm(x, g("model.norm.weight"), eps)
        per_position.append(int(np.argmax(lg)))
        if p_ix == P - 1:
            logits = lg

    write("logits.bin", logits.astype(np.float64).tobytes())
    write("tokens.bin", np.asarray(per_position, dtype="<u4").tobytes())
    print(f"  argmax per position: {per_position}")

    top = np.argsort(logits)[::-1][:10]
    print(f"\n  top tokens by float64 logit:")
    for r, ix in enumerate(top):
        print(f"    {r+1:2d}. id {ix:6d}  logit {logits[ix]:+.4f}")

    meta = {
        "dim": dim, "hidden": hidden, "n_layers": L, "n_heads": H, "n_kv": KV,
        "head_dim": hs, "vocab": V, "positions": P, "rope_theta": theta,
        "prompt_ids": ids, "tied": tied,
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"\n  wrote {a.out} ({total/1e9:.2f} GB)")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
