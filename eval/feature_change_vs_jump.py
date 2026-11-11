"""
How much an encoder's per-atom features change between a frame and a later one, as a
function of the gap, on our water runs. This is the quantity behind the organic
finding that "features change by about 0.4% in 8 fs, so there is almost nothing to
predict": if water features also barely change at 16 fs, pretraining at that stride has
the same weakness, and a longer gap should be used.

Measure, for a gap tau: the mean over atoms of |h(t+tau) - h(t)|^2, divided by the mean
over atoms of |h(t) - mean h|^2 (the spread of the features across atoms at one
moment). 0 means the features do not move at all; 1 means they move as much as they
differ between atoms. This is the error of the copying predictor relative to the
feature spread.

Encoders measured (same function, same frames):
  untrained   built with torch.manual_seed(0), as the pretraining starts
  baseline    the encoder of the trained water 16 fs baseline (results/runs/<name>/model.ckpt)
  each pretrained encoder given with --pretrained (pretrained.ckpt)

Frames: from --runs constant-energy runs (raw i-PI output, 0.25 fs), start times every
--spacing fs, for gaps 1, 4, 8, 16, 32 and 64 fs. Water runs are used whole (all
splits), as this is a property of the encoder and the dynamics, not a test score.

  python eval/feature_change_vs_jump.py /storage3/lauren/data/water configs/flashmd_water16_scratch.yaml \
      --baseline results/runs/flashmd_water16_scratch/model.ckpt --out results/eval/feature_change_water.json
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
from train.pretrain_jepa import build_model, encode, load_config, make_system  # noqa: E402

DT_FS = 0.25
GAPS_FS = [1, 4, 8, 16, 32, 64]


def load_encoder(kind, path, hypers, info, masses, timestep, device):
    torch.manual_seed(0)
    model = build_model(hypers, info, masses, timestep)
    if kind == "pretrained":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "finetune_config"}
        model.load_state_dict(sd)
    elif kind == "baseline":
        # a trained baseline carries heads and a scaler for its targets; only the shared
        # encoder weights are copied into the target-free model (strict on those names)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("best_model_state_dict") or ckpt["model_state_dict"]
        own = model.state_dict()
        shared = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        missing = [k for k in own if k not in shared]
        assert not [k for k in missing if not k.startswith(("scaler", "additive_models"))], \
            f"encoder weights missing from {path}: {missing[:5]}"
        model.load_state_dict(shared, strict=False)
    model = model.to(device=device, dtype=torch.float32)
    for additive in model.additive_models:
        additive.to(dtype=torch.float64)
    model.scaler.to(dtype=torch.float64)
    return model


def features(model, frames, device):
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
    nl = model.requested_neighbor_lists()
    out = []
    with torch.no_grad():
        for types, q, p, cell in frames:
            s = make_system(types, q, p, cell, torch.tensor([True, True, True]), torch.float32, device)
            out.append(encode(model, [get_system_with_neighbor_lists(s, nl)]).cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root")
    ap.add_argument("config")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=200.0, help="fs between start times within a run")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--pretrained", nargs="*", default=[], help="name=path/to/pretrained.ckpt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, hypers, masses, timestep, info = load_config(args.config)

    starts, later = [], {g: [] for g in GAPS_FS}
    for d in sorted(glob.glob(os.path.join(args.runs_root, "run_*")))[:args.runs]:
        q, symbols, cells = read_ipi_xyz(os.path.join(d, "nve.pos_0.xyz"))
        p, _, _ = read_ipi_xyz(os.path.join(d, "nve.mom_0.xyz"))
        p = p * P_AU_TO_ASE
        from ase.data import atomic_numbers
        types = torch.tensor([atomic_numbers[s] for s in symbols], dtype=torch.int32)
        cell = torch.tensor(np.diag(cells[0]))
        step = int(round(args.spacing / DT_FS))
        for t0 in range(0, len(q) - int(round(max(GAPS_FS) / DT_FS)), step):
            starts.append((types, torch.tensor(q[t0]), torch.tensor(p[t0]), cell))
            for g in GAPS_FS:
                t1 = t0 + int(round(g / DT_FS))
                later[g].append((types, torch.tensor(q[t1]), torch.tensor(p[t1]), cell))
    print(f"{len(starts)} start frames from {args.runs} runs; gaps {GAPS_FS} fs")

    encoders = [("untrained", "untrained", "")]
    if args.baseline:
        encoders.append(("baseline", "baseline", args.baseline))
    for spec in args.pretrained:
        name, path = spec.split("=", 1)
        encoders.append((name, "pretrained", path))
    res = {"gaps_fs": GAPS_FS, "start_frames": len(starts), "encoders": {}}
    for name, kind, path in encoders:
        model = load_encoder(kind, path, hypers, info, masses, timestep, device)
        h0 = features(model, starts, device)
        types = [t.numpy() for t, _, _, _ in starts]
        spread = float(np.mean([((h - h.mean(0)) ** 2).sum(1).mean() for h in h0]))
        # per element: change of that element's atoms over the spread among atoms of the SAME
        # element (the all-atom spread is mostly the oxygen/hydrogen difference, which makes
        # the all-atom ratio look small)
        elements = sorted(set(np.concatenate(types).tolist()))
        spread_el = {z: float(np.mean([((h[t == z] - h[t == z].mean(0)) ** 2).sum(1).mean() for h, t in zip(h0, types)]))
                     for z in elements}
        row, row_el = {}, {str(z): {} for z in elements}
        for g in GAPS_FS:
            h1 = features(model, later[g], device)
            d2 = [((b - a) ** 2).sum(1) for a, b in zip(h0, h1)]
            row[str(g)] = float(np.mean([d.mean() for d in d2])) / spread
            for z in elements:
                row_el[str(z)][str(g)] = float(np.mean([d[t == z].mean() for d, t in zip(d2, types)])) / spread_el[z]
        res["encoders"][name] = {"path": path, "relative_change": row, "relative_change_by_element": row_el}
        print(f"{name:12s} all atoms  " + "  ".join(f"{g} fs: {row[str(g)]:.4f}" for g in GAPS_FS), flush=True)
        for z in elements:
            print(f"{'':12s} element {z:2d} " + "  ".join(f"{g} fs: {row_el[str(z)][str(g)]:.4f}" for g in GAPS_FS), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
