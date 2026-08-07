#!/usr/bin/env python3
"""Reference vectors for the end-to-end test: real weights, real activations, float truth.

The Rust tests can check the precompile against the kernel, but both are integer code
sharing an implementation — agreeing proves consistency, not correctness. This computes the
same projection in float64 from the original bf16 weights, so the chain's answer is
measured against what the model actually means rather than against another copy of itself.

Writes JSON small enough to commit alongside the code, so the test survives without the
1.5 GB checkpoint.
"""

import argparse
import json
import os
import sys

import numpy as np

from quantize import Q_MAX, dequantize, quantize_rows, read_safetensors

# Activations are Q32: a fixed-point value v is stored as round(v * 2**32). The forward
# pass keeps them well under 1.0 in magnitude, so this range is representative without
# risking the int32 bound.
Q32 = 1 << 32


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--rows",
        type=int,
        help="keep only this many leading rows, so the fixture can be committed alongside "
        "the code instead of depending on a 1.5 GB checkpoint",
    )
    a = ap.parse_args()

    t = read_safetensors(a.model, {a.tensor})
    if a.tensor not in t:
        raise SystemExit(f"{a.tensor} not found in {a.model}")
    w = t[a.tensor].astype(np.float64)
    if a.rows:
        w = w[: a.rows]
    rows, cols = w.shape

    # Deterministic, so a failure is reproducible from the seed alone.
    rng = np.random.default_rng(a.seed)
    acts_f = rng.normal(0.0, 0.02, size=cols)
    acts_q = np.rint(acts_f * Q32).astype(np.int64)
    if np.abs(acts_q).max() > np.iinfo(np.int32).max:
        raise SystemExit("activations do not fit int32; reduce their spread")
    acts_q = acts_q.astype(np.int32)

    # What the model means: float64 against the original weights.
    truth = w @ (acts_q.astype(np.float64) / Q32)

    # What the chain will compute: integers throughout, with the same rounding the kernel
    # applies. Reproducing it here rather than trusting the kernel is the point — it is an
    # independent implementation of the same specification.
    q, shifts = quantize_rows(w.astype(np.float32))
    acc = q.astype(np.int64) @ acts_q.astype(np.int64)
    # The accumulator carries shift+32 fractional bits; a Q32 result wants 32, so shed
    # exactly `shift`. Writing `32 - shift` here instead agrees with a kernel making the
    # same mistake and disagrees with the model by a thousand percent.
    shift = shifts.astype(np.int64)
    exact = np.where(shift >= 0, acc >> np.clip(shift, 0, 63), acc << np.clip(-shift, 0, 62))
    exact = np.clip(exact, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)

    # The integer path's answer, read back as a real number, against float64 truth.
    got = exact.astype(np.float64) / Q32
    scale = np.abs(truth).mean()
    rel = np.abs(got - truth).mean() / scale if scale > 0 else 0.0

    # Separately: what the quantisation alone costs, ignoring fixed-point rounding.
    deq = dequantize(q, shifts).astype(np.float64)
    wq_err = np.abs(deq - w).mean() / (np.abs(w).mean() or 1.0)

    print(f"{a.tensor}")
    print(f"  shape                     {rows} x {cols}")
    print(f"  int16 vs bf16 weights     {wq_err:.3e} mean relative error")
    print(f"  full integer path vs f64  {rel:.3e} mean relative error")
    print(f"  peak |weight|             {np.abs(w).max():.5f}")
    print(f"  shifts                    {shifts.min()}..{shifts.max()}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    # The blob goes out in the precompile's own format, so the test feeds the chain exactly
    # what a deployed model would hold rather than a re-encoding of it.
    from quantize import blob as make_blob

    bin_path = os.path.splitext(a.out)[0] + ".bin"
    with open(bin_path, "wb") as f:
        f.write(make_blob(q, shifts))

    with open(a.out, "w") as f:
        json.dump(
            {
                "tensor": a.tensor,
                "rows": int(rows),
                "cols": int(cols),
                "seed": a.seed,
                "blob": os.path.basename(bin_path),
                "activations": [int(x) for x in acts_q],
                "expected": [int(x) for x in exact],
                "float_reference": [float(x) for x in truth],
                "relative_error": float(rel),
            },
            f,
        )
    print(f"  wrote {a.out} and {bin_path}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
