#!/usr/bin/env python3
"""Export one real Qwen3-MoE layer, plus a float64 forward pass of it.

The mixture is checked against float64 on synthetic weights already. That proves the
arithmetic and the routing rule. It does not prove either survives a real checkpoint, where
a router's scores are decided by trained weights rather than by a random matrix, and where
two experts out of a hundred and twenty-eight can be separated by very little.

Qwen3-30B-A3B is the target: 2048 wide, 128 experts of which 8 are visited, 48 layers. One
layer is 1.25GB, which is why a layer is the unit here — the whole model is 61GB, and a
layer is enough to answer whether the routing and the quantiser hold up on real weights.

A note on the routing rule, because it looks like a discrepancy and is not. Qwen3 softmaxes
over all 128 experts, takes the top 8, then renormalises those 8 to sum to one. This project
softmaxes over the chosen 8 directly. They are the same function: dividing exp(s_i)/Z by the
sum of exp(s_j)/Z over the chosen cancels Z exactly. `norm_topk_prob` being true in the
config is what makes that so; a model with it false would need the other form.

Writes the layer in the chain's layout — experts laid out expert by expert so one stride
addresses all of them — and what float64 says the layer produces.
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
    """Q32 as int64. int32 would span only [-0.5, 0.5), which normalised activations exceed."""
    return np.rint(np.asarray(a, dtype=np.float64) * Q32).astype(np.int64)


def rmsnorm(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean() + eps) * w


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()




def moe_forward(n, W, E, K):
    """The mixture, in float64, exactly as the chain does it."""
    sc = W["router"] @ n
    order = sorted(range(E), key=lambda i: (-sc[i], i))[:K]
    ex = np.exp(np.array([sc[i] for i in order]) - max(sc[i] for i in order))
    rw = ex / ex.sum()
    out = np.zeros_like(n)
    for slot, i in enumerate(order):
        g = W[f"e{i}_gate"] @ n
        u = W[f"e{i}_up"] @ n
        out = out + rw[slot] * (W[f"e{i}_down"] @ ((g / (1.0 + np.exp(-g))) * u))
    return out, order, rw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the safetensors shard holding the layer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
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
    L = a.layer

    if not cfg.get("norm_topk_prob", True):
        raise SystemExit(
            "norm_topk_prob is false: this model normalises over all experts, and the "
            "chain's softmax over the chosen would not match it"
        )

    names = {
        "input_norm": f"model.layers.{L}.input_layernorm.weight",
        "q": f"model.layers.{L}.self_attn.q_proj.weight",
        "k": f"model.layers.{L}.self_attn.k_proj.weight",
        "v": f"model.layers.{L}.self_attn.v_proj.weight",
        "q_norm": f"model.layers.{L}.self_attn.q_norm.weight",
        "k_norm": f"model.layers.{L}.self_attn.k_norm.weight",
        "o": f"model.layers.{L}.self_attn.o_proj.weight",
        "post_norm": f"model.layers.{L}.post_attention_layernorm.weight",
        "router": f"model.layers.{L}.mlp.gate.weight",
    }
    for e in range(E):
        p = f"model.layers.{L}.mlp.experts.{e}."
        names[f"e{e}_gate"] = p + "gate_proj.weight"
        names[f"e{e}_up"] = p + "up_proj.weight"
        names[f"e{e}_down"] = p + "down_proj.weight"

    t = read_safetensors(a.model, set(names.values()))
    missing = [n for n in names.values() if n not in t]
    if missing:
        raise SystemExit(f"missing {len(missing)} tensors, first: {missing[:3]}")
    W = {k: t[v].astype(np.float64) for k, v in names.items()}

    print(f"layer {L}: dim {dim}, heads {H}, kv {KV}, head_dim {hs}")
    print(f"  mixture: {E} experts of {hidden} wide, top {K}")
    print(f"  a token reads {K/E:.1%} of the feed-forward")

    half = hs // 2
    inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hs)
    ang = np.arange(a.positions, dtype=np.float64)[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)

    rng = np.random.default_rng(a.seed)
    xs = rng.normal(0.0, 1.0, size=(a.positions, dim))

    # ---- float64 reference ----
    kc, vc, outs, routed = [], [], [], []
    moe_in, moe_out = [], []
    for p in range(a.positions):
        h = xs[p].copy()
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

        group = H // KV
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
        mix, order, rw = moe_forward(n, W, E, K)
        h = h + mix
        outs.append(h)
        routed.append(order)
        # The exact vector the router scored, and what the mixture returned for it. Written
        # so the integer side can be asked the same question rather than a similar one:
        # comparing whole-layer outputs would only say that routing probably agreed.
        moe_in.append(n)
        moe_out.append(mix)
        print(f"  position {p}: experts {order}  weights {np.round(rw, 3).tolist()}")

    os.makedirs(a.out, exist_ok=True)

    def write(name, data):
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)

    for key in ("q", "k", "v", "o"):
        qm, sh = quantize_rows(W[key].astype(np.float32))
        write(f"{key}.bin", blob(qm, sh))
    for key in ("input_norm", "q_norm", "k_norm", "post_norm"):
        write(f"{key}.bin", to_q32(W[key]).astype("<i8").tobytes())

    # Expert by expert, so expert e's three tensors sit one stride past expert zero's.
    parts = []
    for e in range(E):
        for key in ("gate", "up", "down"):
            qm, sh = quantize_rows(W[f"e{e}_{key}"].astype(np.float32))
            parts.append(blob(qm, sh))
    experts = b"".join(parts)
    write("experts.bin", experts)
    qm, sh = quantize_rows(W["router"].astype(np.float32))
    write("router.bin", blob(qm, sh))

    write("cos.bin", to_q32(cos.reshape(-1)).astype("<i8").tobytes())
    write("sin.bin", to_q32(sin.reshape(-1)).astype("<i8").tobytes())
    write("x.bin", to_q32(xs.reshape(-1)).astype("<i8").tobytes())
    write("expected.bin", np.asarray(outs, dtype=np.float64).reshape(-1).tobytes())
    write("moe_in.bin", to_q32(np.asarray(moe_in).reshape(-1)).astype("<i8").tobytes())
    write("moe_out.bin", np.asarray(moe_out, dtype=np.float64).reshape(-1).tobytes())

    stride = len(experts) // E
    meta = {
        "dim": dim, "hidden": hidden, "n_heads": H, "n_kv": KV, "head_dim": hs,
        "positions": a.positions, "layer": L, "rope_theta": theta,
        "n_experts": E, "top_k": K, "expert_stride": stride,
        "routed": routed,
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"  expert stride {stride} bytes, {len(experts)/1e9:.2f} GB of experts")
    print(f"  wrote {a.out} ({total/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
