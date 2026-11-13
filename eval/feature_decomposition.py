"""
Where an encoder's feature spread comes from, and how much features change over the gap the
pretraining used (E08, hypothesis H2). Same measurements for organic and water encoders.

1. Spread split. Run the encoder on many frames of the same atoms. Each feature vector is
   split into three parts that add up exactly (sums of squares over the 640 numbers):
     element   the average over all atoms of the same element
     atom      this atom's own average over all frames, minus its element's average
               ("which atom am I": chemical surroundings, which do not change)
     moment    this frame's value minus the atom's own average ("what is happening now")
   Reported as shares of the total spread (they add to 1). H2 expects organic features to be
   almost all element + atom, and water features to have a large moment share.
2. Change over the gap. Mean over atoms of |h(later) - h(now)|^2, divided by the spread of
   features across the atoms of one frame (the measure in eval/feature_change_vs_jump.py),
   and the same within each element. 0 means the features do not move.

Data, never used for training by any encoder measured:
  organic  data/organic8/val.xyz (21 molecules, 500 frames each); every --every-th frame, forward
           frames only (the file stores each start frame followed by its time-reversed copy);
           "later" is the frame's stored 8 fs future. Atom identity = (molecule, index), with
           molecule ids from data/organic8/frames.csv.
  water    the raw constant-energy runs of the validation split (data/water/pairs/16fs/
           manifest.json), start frames every --spacing fs; "later" is 8 and 16 fs on.
           Atom identity = (run, index).

Encoders: name=kind:path, kind untrained (path ignored), baseline (a trained model.ckpt, encoder
weights only) or pretrained (pretrained.ckpt). Loading as in eval/feature_change_vs_jump.py.

  python eval/feature_decomposition.py organic configs/flashmd_organic8_scratch.yaml \
      --encoders untrained=untrained: baseline=baseline:results/runs/flashmd_organic8_scratch_10pct_s0/model.ckpt \
      plain=pretrained:results/runs/jepa_pretrain_plain_organic8/pretrained.ckpt --out results/eval/decomp_organic.json
"""

import argparse
import json
import os
import sys

import ase.io
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from checks.check_water_run_io import P_AU_TO_ASE, read_ipi_xyz  # noqa: E402
from eval.feature_change_vs_jump import features, load_encoder  # noqa: E402
from train.pretrain_jepa import load_config  # noqa: E402

DT_FS = 0.25


def organic_frames(path, masses, every, frames_csv="data/organic8/frames.csv", split="val"):
    """(groups, now, later): per sampled frame its molecule id, (types, q, p, cell) now and 8 fs later.
    Only forward frames are used (even positions within a molecule; each is followed in the file
    by its time-reversed copy, which is skipped): water frames come from raw forward runs, so
    both systems are measured on forward dynamics only. Molecule ids come from frames.csv."""
    rows = pd.read_csv(frames_csv)
    rows = rows[(rows.split == split) & (rows.written == 1)].reset_index(drop=True)
    rel = (rows.frame - rows.groupby("group").frame.transform("min")).to_numpy()
    assert every % 2 == 0, "--every must be even so that only forward frames are taken"
    groups, now, later = [], [], []
    for i, a in enumerate(ase.io.iread(path, index=":", format="extxyz")):
        if i % every:
            continue
        assert rel[i] % 2 == 0, f"frame {i} is not a forward frame"
        m = np.array([masses[int(z)] for z in a.numbers])
        sm = np.sqrt(m)[:, None]
        q = a.get_positions()
        q_fut = q + (a.arrays["future_positions"] - q) / sm
        p_fut = a.arrays["future_momenta"] * sm
        types = torch.tensor(a.numbers, dtype=torch.int32)
        cell = torch.tensor(a.cell.array)
        groups.append(int(rows.group[i]))
        now.append((types, torch.tensor(q), torch.tensor(a.get_momenta()), cell))
        later.append({8: (types, torch.tensor(q_fut), torch.tensor(p_fut), cell)})
    return groups, now, later


def water_frames(runs_root, run_names, spacing_fs, gaps=(8, 16)):
    from ase.data import atomic_numbers
    groups, now, later = [], [], []
    for g, name in enumerate(run_names):
        d = os.path.join(runs_root, name)
        q, symbols, cells = read_ipi_xyz(os.path.join(d, "nve.pos_0.xyz"))
        p, _, _ = read_ipi_xyz(os.path.join(d, "nve.mom_0.xyz"))
        p = p * P_AU_TO_ASE
        types = torch.tensor([atomic_numbers[s] for s in symbols], dtype=torch.int32)
        cell = torch.tensor(np.diag(cells[0]))
        step = int(round(spacing_fs / DT_FS))
        for t0 in range(0, len(q) - int(round(max(gaps) / DT_FS)), step):
            groups.append(g)
            now.append((types, torch.tensor(q[t0]), torch.tensor(p[t0]), cell))
            later.append({gap: (types, torch.tensor(q[t0 + int(round(gap / DT_FS))]),
                                torch.tensor(p[t0 + int(round(gap / DT_FS))]), cell) for gap in gaps})
    return groups, now, later


def spread_split(h_list, groups, types_list):
    """Shares of the total sum of squares from element, atom and moment parts."""
    H = np.concatenate(h_list).astype(np.float64)                     # all atom-frames
    Z = np.concatenate([t.numpy() for t in types_list])
    total = ((H - H.mean(0)) ** 2).sum()
    elem_mean = {z: H[Z == z].mean(0) for z in np.unique(Z)}
    E = np.stack([elem_mean[z] for z in Z])
    element = ((E - H.mean(0)) ** 2).sum()
    atom = moment = 0.0
    for g in sorted(set(groups)):
        idx = [k for k, gg in enumerate(groups) if gg == g]
        stack = np.stack([h_list[k] for k in idx]).astype(np.float64)  # frames x atoms x d
        t = types_list[idx[0]].numpy()
        assert all(np.array_equal(types_list[k].numpy(), t) for k in idx), f"group {g}: atoms differ between frames"
        a_mean = stack.mean(0)                                        # atoms x d
        atom += len(idx) * ((a_mean - np.stack([elem_mean[z] for z in t])) ** 2).sum()
        moment += ((stack - a_mean[None]) ** 2).sum()
    parts = {"element": element / total, "atom": atom / total, "moment": moment / total}
    parts["sum_check"] = sum(parts.values())
    return parts


def change(h0_list, h1_list, types_list):
    spread = float(np.mean([((h - h.mean(0)) ** 2).sum(1).mean() for h in h0_list]))
    d2 = [((b - a) ** 2).sum(1) for a, b in zip(h0_list, h1_list)]
    out = {"all": float(np.mean([d.mean() for d in d2])) / spread}
    for z in sorted(set(np.concatenate([t.numpy() for t in types_list]).tolist())):
        sp = [((h[t.numpy() == z] - h[t.numpy() == z].mean(0)) ** 2).sum(1).mean() for h, t in zip(h0_list, types_list)
              if (t.numpy() == z).sum() > 1]
        ch = [d[t.numpy() == z].mean() for d, t in zip(d2, types_list) if (t.numpy() == z).sum() > 1]
        out[f"element_{z}"] = float(np.mean(ch)) / float(np.mean(sp))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system", choices=["organic", "water"])
    ap.add_argument("config")
    ap.add_argument("--encoders", nargs="+", required=True, help="name=kind:path")
    ap.add_argument("--organic_file", default="data/organic8/val.xyz")
    ap.add_argument("--every", type=int, default=10, help="organic: every this many frames (even: forward frames only)")
    ap.add_argument("--water_runs", default="/storage3/lauren/data/water")
    ap.add_argument("--water_manifest", default="data/water/pairs/16fs/manifest.json")
    ap.add_argument("--spacing", type=float, default=100.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, hypers, masses, timestep, info = load_config(args.config)

    if args.system == "organic":
        groups, now, later = organic_frames(args.organic_file, masses, args.every)
    else:
        runs = json.load(open(args.water_manifest))["split"]["runs"]["val"]
        groups, now, later = water_frames(args.water_runs, runs, args.spacing)
    types_list = [t for t, _, _, _ in now]
    gaps = sorted(later[0].keys())
    print(f"{args.system}: {len(now)} frames in {len(set(groups))} groups; gaps {gaps} fs", flush=True)

    res = {"system": args.system, "frames": len(now), "groups": len(set(groups)), "encoders": {}}
    for spec in args.encoders:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        model = load_encoder(kind, path, hypers, info, masses, timestep, device)
        h0 = features(model, now, device)
        r = {"kind": kind, "path": path, "spread_split": spread_split(h0, groups, types_list), "change": {}}
        for g in gaps:
            h1 = features(model, [l[g] for l in later], device)
            r["change"][str(g)] = change(h0, h1, types_list)
        res["encoders"][name] = r
        s = r["spread_split"]
        print(f"{name:14s} element {s['element']:.3f} atom {s['atom']:.3f} moment {s['moment']:.3f} (sum {s['sum_check']:.3f}) | "
              + "  ".join(f"change {g} fs: {r['change'][str(g)]['all']:.4f}" for g in gaps), flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
