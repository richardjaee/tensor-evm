#!/usr/bin/env python3
"""Turn a HuggingFace safetensors model into blobs the matmul precompile can read.

The precompile's format is int16 weights with one power-of-two scale per row, laid out as
`rows` scale bytes followed by row-major big-endian int16 values. Power-of-two scales
rather than arbitrary floats for a reason that outlives the encoding: dequantising is then
a shift, so the whole forward pass stays in integers, and integers are what make every node
compute identical bits. A float scale would reintroduce exactly the nondeterminism the
design exists to avoid.

Big-endian throughout, because that is how the EVM lays out words and the Solidity side
should not have to swap.
"""

import argparse
import json
import struct
import sys

import numpy as np

# int16 holds [-32768, 32767]; 32767 is the largest magnitude representable in both
# directions, so rows are scaled to land at or under it.
Q_MAX = 32767


def read_safetensors(path, wanted=None):
    """Parse safetensors, widening bf16 by shifting into the high half of a float32.

    numpy has no bfloat16, but the formats agree bit for bit in their top sixteen bits, so
    a left shift by sixteen is an exact widening rather than an approximation.
    """
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if wanted is not None and name not in wanted:
                continue
            start, end = meta["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dtype = meta["dtype"]
            if dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                arr = (u16 << 16).view(np.float32)
            elif dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32)
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            else:
                raise SystemExit(f"{name}: unsupported dtype {dtype}")
            out[name] = arr.reshape(meta["shape"])
        return out


def quantize_rows(w):
    """int16 values plus one power-of-two shift per row.

    The shift is chosen so the row's largest magnitude lands just under int16's range:
    `q = round(v * 2**s)` with `s` the largest integer keeping every value in bounds. Rows
    are scaled independently because a single scale for a whole tensor would be set by its
    worst outlier and would flatten every other row into a handful of levels.
    """
    w = np.asarray(w, dtype=np.float32)
    rows, _ = w.shape
    peak = np.abs(w).max(axis=1)
    shifts = np.zeros(rows, dtype=np.int64)

    nonzero = peak > 0
    # floor(log2(Q_MAX / peak)) — the largest shift that still fits.
    shifts[nonzero] = np.floor(np.log2(Q_MAX / peak[nonzero])).astype(np.int64)
    # A dead row can take any shift; zero keeps the table boring.
    shifts[~nonzero] = 0
    # The blob stores shifts as a signed byte.
    shifts = np.clip(shifts, -128, 127)

    scaled = w * np.exp2(shifts.astype(np.float32))[:, None]
    q = np.clip(np.rint(scaled), -Q_MAX, Q_MAX).astype(np.int16)
    return q, shifts.astype(np.int8)


def dequantize(q, shifts):
    """Inverse of `quantize_rows`, for measuring what the encoding costs."""
    return q.astype(np.float32) * np.exp2(-shifts.astype(np.float32))[:, None]


def blob(q, shifts):
    """`rows` scale bytes, then row-major big-endian int16 — what the precompile reads."""
    return shifts.astype(np.int8).tobytes() + q.astype(">i2").tobytes()


def report(name, w, q, shifts):
    approx = dequantize(q, shifts)
    err = np.abs(approx - w)
    scale = np.abs(w).mean()
    rel = err.mean() / scale if scale > 0 else 0.0
    print(
        f"  {name:48s} {str(w.shape):16s} "
        f"shift {shifts.min():4d}..{shifts.max():<4d} "
        f"rel.err {rel:.5f}"
    )
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to model.safetensors")
    ap.add_argument("--config", help="path to config.json (default: alongside the model)")
    ap.add_argument("--out", required=True, help="output directory for blobs")
    ap.add_argument(
        "--tensors",
        nargs="*",
        help="tensor names to convert (default: every 2-D weight)",
    )
    ap.add_argument("--limit", type=int, help="stop after this many tensors")
    a = ap.parse_args()

    import os

    cfg_path = a.config or os.path.join(os.path.dirname(a.model), "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if cfg:
        arch = cfg.get("architectures", ["?"])[0]
        hs = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        print(f"{arch}: dim {cfg['hidden_size']}, layers {cfg['num_hidden_layers']}, "
              f"heads {cfg['num_attention_heads']}, kv {cfg['num_key_value_heads']}, "
              f"head_dim {hs}, vocab {cfg['vocab_size']}")
        # Loudly, because both are silent when unhandled: a wrong head_dim reshapes rotary
        # embeddings into nonsense, and a missing QK-norm shifts every attention score.
        if hs * cfg["num_attention_heads"] != cfg["hidden_size"]:
            print(f"  NOTE: head_dim*heads = {hs * cfg['num_attention_heads']} != hidden_size "
                  f"{cfg['hidden_size']}; q_proj and o_proj are not square")
        print(f"  rope_theta {cfg.get('rope_theta')}, tied {cfg.get('tie_word_embeddings')}")

    tensors = read_safetensors(a.model, set(a.tensors) if a.tensors else None)
    names = sorted(n for n, t in tensors.items() if t.ndim == 2)
    if a.limit:
        names = names[: a.limit]
    if not names:
        raise SystemExit("no 2-D tensors selected")

    os.makedirs(a.out, exist_ok=True)
    index = {}
    worst = 0.0
    total = 0

    print(f"\nquantising {len(names)} tensors to int16 with power-of-two row scales\n")
    for name in names:
        w = tensors[name]
        q, shifts = quantize_rows(w)
        worst = max(worst, report(name, w, q, shifts))
        data = blob(q, shifts)
        path = os.path.join(a.out, name.replace("/", "_") + ".bin")
        with open(path, "wb") as f:
            f.write(data)
        index[name] = {"rows": int(w.shape[0]), "cols": int(w.shape[1]), "bytes": len(data)}
        total += len(data)

    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump({"config": cfg, "tensors": index}, f, indent=2)

    print(f"\n{total/1e6:.1f} MB across {len(names)} tensors, worst relative error {worst:.5f}")
    print(f"index written to {os.path.join(a.out, 'index.json')}")


if __name__ == "__main__":
    sys.exit(main())
