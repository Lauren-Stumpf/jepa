"""
Organic pairs with a longer gap, for pretraining only (E12, hypothesis H2).
The 100 fs spacing of start times within a run is from notes/decisions.md ("25 pairs per run
every 100 fs"); the 50-frame runs are confirmed by the total energy, constant within a block.

The organic pairs files (data/organic8/{train,val}.xyz) hold frames in the order of the
FlashMD file: for each molecule 500 frames = 10 constant-energy runs of 50 frames; within a
run, 25 start times 100 fs apart, each start frame followed by its own 8 fs future. So the
frame two places later in the same run is the same molecule 100 fs later.

This script writes new pairs files in the same format, where each frame's "future" is that
frame two places later (100 fs) instead of its stored 8 fs future. A frame is paired only
if, according to the catalogue of the source file:
  - the later frame is exactly two frames later in the source file,
  - both lie in the same 50-frame block (the same run) and the same structure group,
  - their total energy per atom differs by less than --max_de eV (same constant-energy run),
  - no atom moves more than --max_disp A between them (catches a wrap across the box).
Every exclusion is counted and printed. Targets are stored mass-scaled exactly as in
data/prepare_organic.py (future_positions := q + dq * sqrt(m), future_momenta := p' / sqrt(m)),
so train/pretrain_jepa.py reads the files unchanged.

The fine-tuning task is NOT changed: fine-tunes still use data/organic8/train_10pct.xyz (8 fs).

  python data/make_gap_pairs.py data/organic8 train data/organic8_gap100 configs/flashmd_organic8_scratch.yaml
  python data/make_gap_pairs.py data/organic8 val   data/organic8_gap100 configs/flashmd_organic8_scratch.yaml
"""

import argparse
import json
import os

import ase.io
import ase.units
import numpy as np
import pandas as pd
import yaml

RUN_FRAMES = 50
K_B = ase.units.kB   # eV per K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("out_dir")
    ap.add_argument("config")
    ap.add_argument("--catalogue", default="results/data/catalogue_8fs.csv")
    ap.add_argument("--gap_frames", type=int, default=2)
    ap.add_argument("--max_de", type=float, default=1e-3, help="eV per atom")
    ap.add_argument("--max_disp", type=float, default=3.0, help="A")
    args = ap.parse_args()

    masses = {int(k): float(v) for k, v in yaml.safe_load(open(args.config))["architecture"]["training"]["masses"].items()}
    frames_csv = pd.read_csv(os.path.join(args.src_dir, "frames.csv"))
    rows = frames_csv[(frames_csv.split == args.split) & (frames_csv.written == 1)].reset_index(drop=True)
    cat = pd.read_csv(args.catalogue, usecols=["frame", "group", "formula", "kinetic_T_K", "pe_eV_atom"]).set_index("frame")
    src = list(ase.io.iread(os.path.join(args.src_dir, f"{args.split}.xyz"), index=":", format="extxyz"))
    assert len(src) == len(rows), f"{len(src)} frames in the file but {len(rows)} written rows in frames.csv"

    orig = rows.frame.to_numpy()
    pos_of = {f: i for i, f in enumerate(orig)}     # source frame id -> position in this split's file
    e_tot = (1.5 * K_B * cat.kinetic_T_K + cat.pe_eV_atom)
    counts = {"frames": len(src), "paired": 0, "later_frame_not_in_split": 0, "different_run_block": 0,
              "different_group": 0, "energy_differs": 0, "moved_too_far": 0, "atoms_differ": 0}
    out = []
    for i, a in enumerate(src):
        f0 = int(orig[i])
        f1 = f0 + args.gap_frames
        if f1 not in pos_of:
            counts["later_frame_not_in_split"] += 1
            continue
        if f0 // RUN_FRAMES != f1 // RUN_FRAMES:
            counts["different_run_block"] += 1
            continue
        if cat.group[f0] != cat.group[f1]:
            counts["different_group"] += 1
            continue
        if abs(e_tot[f0] - e_tot[f1]) > args.max_de:
            counts["energy_differs"] += 1
            continue
        b = src[pos_of[f1]]
        if len(b) != len(a) or not np.array_equal(a.numbers, b.numbers):
            counts["atoms_differ"] += 1
            continue
        dq = b.get_positions() - a.get_positions()
        if np.abs(np.linalg.norm(dq, axis=1)).max() > args.max_disp:
            counts["moved_too_far"] += 1
            continue
        sm = np.sqrt(np.array([masses[int(z)] for z in a.numbers]))[:, None]
        c = a.copy()
        c.set_momenta(a.get_momenta())
        c.arrays["future_positions"] = a.get_positions() + dq * sm
        c.arrays["future_momenta"] = b.get_momenta() / sm
        out.append(c)
        counts["paired"] += 1

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.split}.xyz")
    ase.io.write(path, out, format="extxyz")
    counts["gap_fs"] = 100 * args.gap_frames // 2
    with open(os.path.join(args.out_dir, f"manifest_{args.split}.json"), "w") as fh:
        json.dump({"source": os.path.join(args.src_dir, f"{args.split}.xyz"), "rule": vars(args), "counts": counts}, fh, indent=1)
    print(json.dumps(counts, indent=1))
    print(f"written {path}")


if __name__ == "__main__":
    main()
