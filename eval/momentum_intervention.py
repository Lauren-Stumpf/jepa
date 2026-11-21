"""
Does an encoder's output respond to the momenta at all? (E25 part A, water.)

A read-out that finds no momentum in the features does not show the information is absent.
This script intervenes on the input instead. Each start frame goes through the encoder as
stored, and again with one thing changed:
  swap      positions kept; every atom's momentum replaced by the same atom's momentum
            --swap_fs later in the same run (unrelated in direction, typical in size)
  reverse   positions kept; every momentum reversed in sign
  move      momenta kept; positions replaced by the positions 16 fs later
  step      the real frame 16 fs later (positions and momenta), the E08 "change over one step"
  same      nothing changed (control: must be exactly 0)

Measure, per element: the mean over atoms of |h(changed) - h(stored)|^2, divided by the spread
of the features among atoms of the same element in one frame (the scale of
eval/feature_change_vs_jump.py). 0 means no response; about 2 means unrelated features.
The size of the swap itself is reported on the same scale for the momenta (about 2 when the two
momenta are unrelated), so a small response cannot come from a small intervention.

Encoders: untrained (seed 0), baseline (a trained model.ckpt, encoder weights only) and
pretrained (pretrained.ckpt), loaded as in eval/feature_change_vs_jump.py.

  python eval/momentum_intervention.py /storage3/lauren/data/water configs/flashmd_water16_scratch.yaml \
      --run_list results/eval/water_test_runs.txt \
      --baseline results/runs/flashmd_water16_scratch/model.ckpt \
      --pretrained plain=results/runs/jepa_pretrain_plain_water16/pretrained.ckpt \
      --out results/eval/momentum_intervention_water.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from checks.check_water_run_io import P_AU_TO_ASE, read_ipi_xyz  # noqa: E402
from eval.feature_change_vs_jump import DT_FS, features, load_encoder  # noqa: E402
from train.pretrain_jepa import load_config  # noqa: E402

STEP_FS = 16.0
KINDS = ["same", "swap", "reverse", "move", "step"]


def relative_change(h0, h1, types, elements):
    """Per element: mean squared feature change over the spread among atoms of that element."""
    out = {}
    for z in elements:
        change = np.mean([((b[t == z] - a[t == z]) ** 2).sum(1).mean() for a, b, t in zip(h0, h1, types)])
        spread = np.mean([((a[t == z] - a[t == z].mean(0)) ** 2).sum(1).mean() for a, t in zip(h0, types)])
        out[str(z)] = float(change / spread)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root")
    ap.add_argument("config")
    ap.add_argument("--run_list", default="", help="file with one run directory name per line (for example the test runs); default: the first --runs runs")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=200.0, help="fs between start times within a run")
    ap.add_argument("--swap_fs", type=float, default=200.0, help="how much later the swapped-in momenta are taken")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--pretrained", nargs="*", default=[], help="name=path/to/pretrained.ckpt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, hypers, masses, timestep, info = load_config(args.config)

    if args.run_list:
        dirs = [os.path.join(args.runs_root, line.strip()) for line in open(args.run_list) if line.strip()]
    else:
        dirs = sorted(glob.glob(os.path.join(args.runs_root, "run_*")))[:args.runs]
    assert dirs, "no runs found"

    from ase.data import atomic_numbers
    frames = {k: [] for k in KINDS}
    swap_size = []
    n_step, n_swap = int(round(STEP_FS / DT_FS)), int(round(args.swap_fs / DT_FS))
    for d in dirs:
        q, symbols, cells = read_ipi_xyz(os.path.join(d, "nve.pos_0.xyz"))
        p, _, _ = read_ipi_xyz(os.path.join(d, "nve.mom_0.xyz"))
        p = p * P_AU_TO_ASE
        types = torch.tensor([atomic_numbers[s] for s in symbols], dtype=torch.int32)
        cell = torch.tensor(np.diag(cells[0]))
        tn = types.numpy()
        for t0 in range(0, len(q) - max(n_step, n_swap), int(round(args.spacing / DT_FS))):
            q0, p0 = torch.tensor(q[t0]), torch.tensor(p[t0])
            q1, p1 = torch.tensor(q[t0 + n_step]), torch.tensor(p[t0 + n_step])
            ps = torch.tensor(p[t0 + n_swap])
            frames["same"].append((types, q0, p0, cell))
            frames["swap"].append((types, q0, ps, cell))
            frames["reverse"].append((types, q0, -p0, cell))
            frames["move"].append((types, q1, p0, cell))
            frames["step"].append((types, q1, p1, cell))
            # size of the swap on the momenta themselves, per element, on the same scale
            swap_size.append({int(z): float(((p[t0 + n_swap][tn == z] - p[t0][tn == z]) ** 2).sum(1).mean()
                                             / ((p[t0][tn == z] - p[t0][tn == z].mean(0)) ** 2).sum(1).mean())
                              for z in sorted(set(tn.tolist()))})
    types = [t.numpy() for t, _, _, _ in frames["same"]]
    elements = sorted(set(np.concatenate(types).tolist()))
    size = {str(z): float(np.mean([s[z] for s in swap_size])) for z in elements}
    print(f"{len(types)} start frames from {len(dirs)} runs; swap {args.swap_fs} fs later; "
          f"size of the swap on the momenta (about 2 = unrelated): {size}")

    encoders = [("untrained", "untrained", "")]
    if args.baseline:
        encoders.append(("baseline", "baseline", args.baseline))
    for spec in args.pretrained:
        name, path = spec.split("=", 1)
        encoders.append((name, "pretrained", path))

    res = {"start_frames": len(types), "runs": [os.path.basename(d) for d in dirs], "swap_fs": args.swap_fs,
           "step_fs": STEP_FS, "swap_size_on_momenta": size, "encoders": {}}
    for name, kind, path in encoders:
        model = load_encoder(kind, path, hypers, info, masses, timestep, device)
        h0 = features(model, frames["same"], device)
        row = {}
        for k in KINDS:
            h1 = features(model, frames[k], device)
            row[k] = relative_change(h0, h1, types, elements)
        assert all(v == 0.0 for v in row["same"].values()), f"{name}: the same input gave different features: {row['same']}"
        res["encoders"][name] = {"path": path, "relative_change_by_element": row}
        for z in elements:
            print(f"{name:14s} element {z:2d}  " + "  ".join(f"{k}: {row[k][str(z)]:.5f}" for k in KINDS), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
