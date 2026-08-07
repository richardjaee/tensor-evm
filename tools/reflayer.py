#!/usr/bin/env python3
"""Export one real Qwen3 layer, plus a float64 forward pass of it, for the Rust example.

The integer layer is checked against a float64 implementation on synthetic weights in the
unit tests. That proves the arithmetic. It does not prove the arithmetic survives a real
checkpoint, where weight magnitudes vary by orders of magnitude across rows and the
quantiser has to cope with all of them at once.

This writes a directory the `real_layer` example reads: quantised tensors in the
precompile's format, norm weights and rotary tables in Q32, a starting residual stream, and
what float64 says the layer should produce.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    dim = cfg["hidden_size"]
    hidden = cfg["intermediate_size"]
    H = cfg["num_attention_heads"]
    KV = cfg["num_key_value_heads"]
    hs = cfg["head_dim"]
    theta = float(cfg["rope_theta"])
    eps = cfg.get("rms_norm_eps", 1e-6)
    L = a.layer

    names = {
        "input_norm": f"model.layers.{L}.input_layernorm.weight",
        "q": f"model.layers.{L}.self_attn.q_proj.weight",
        "k": f"model.layers.{L}.self_attn.k_proj.weight",
        "v": f"model.layers.{L}.self_attn.v_proj.weight",
        "q_norm": f"model.layers.{L}.self_attn.q_norm.weight",
        "k_norm": f"model.layers.{L}.self_attn.k_norm.weight",
        "o": f"model.layers.{L}.self_attn.o_proj.weight",
        "post_norm": f"model.layers.{L}.post_attention_layernorm.weight",
        "gate": f"model.layers.{L}.mlp.gate_proj.weight",
        "up": f"model.layers.{L}.mlp.up_proj.weight",
        "down": f"model.layers.{L}.mlp.down_proj.weight",
    }
    t = read_safetensors(a.model, set(names.values()))
    missing = [n for n in names.values() if n not in t]
    if missing:
        raise SystemExit(f"missing tensors: {missing}")
    W = {k: t[v].astype(np.float64) for k, v in names.items()}

    print(f"layer {L}: dim {dim}, hidden {hidden}, heads {H}, kv {KV}, head_dim {hs}")
    print(f"  q_proj {W['q'].shape}, o_proj {W['o'].shape}  (neither square)")

    # Rotary tables. Precomputed here so the chain never needs an integer sine.
    half = hs // 2
    inv = theta ** (-np.arange(0, half, dtype=np.float64) * 2.0 / hs)
    pos_ix = np.arange(a.positions, dtype=np.float64)[:, None]
    ang = pos_ix * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)

    rng = np.random.default_rng(a.seed)
    # Unit scale, as a real residual stream carries.
    xs = rng.normal(0.0, 1.0, size=(a.positions, dim))

    # ---- float64 reference ----
    x = xs.copy()
    kc, vc = [], []
    outs = []
    for p in range(a.positions):
        h = x[p].copy()
        n = rmsnorm(h, W["input_norm"], eps)
        q = W["q"] @ n
        k = W["k"] @ n
        v = W["v"] @ n

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
        g = W["gate"] @ n
        u = W["up"] @ n
        h = h + W["down"] @ ((g / (1.0 + np.exp(-g))) * u)
        outs.append(h)

    os.makedirs(a.out, exist_ok=True)

    def write(name, data):
        with open(os.path.join(a.out, name), "wb") as f:
            f.write(data)

    for key in ("q", "k", "v", "o", "gate", "up", "down"):
        qm, sh = quantize_rows(W[key].astype(np.float32))
        write(f"{key}.bin", blob(qm, sh))
    for key in ("input_norm", "q_norm", "k_norm", "post_norm"):
        write(f"{key}.bin", to_q32(W[key]).astype("<i8").tobytes())

    write("cos.bin", to_q32(cos.reshape(-1)).astype("<i8").tobytes())
    write("sin.bin", to_q32(sin.reshape(-1)).astype("<i8").tobytes())
    write("x.bin", to_q32(xs.reshape(-1)).astype("<i8").tobytes())
    write("expected.bin", np.asarray(outs, dtype=np.float64).reshape(-1).tobytes())

    meta = {
        "dim": dim,
        "hidden": hidden,
        "n_heads": H,
        "n_kv": KV,
        "head_dim": hs,
        "positions": a.positions,
        "rope_theta": theta,
        "layer": L,
        "shapes": {k: list(W[k].shape) for k in W},
    }
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total = sum(
        os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out)
    )
    print(f"  wrote {a.out} ({total/1e6:.1f} MB)")
    print(f"  residual magnitude after the layer: {np.abs(outs[-1]).mean():.4f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
