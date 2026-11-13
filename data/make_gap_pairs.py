"""
Organic pairs with a longer gap, for pretraining only (E12, hypothesis H2).

Layout of the organic pairs files (data/organic8/{train,val}.xyz), checked on 13 Nov: frames
are in the order of the FlashMD source file; for each molecule 500 frames = 10 constant-energy
runs of 50 frames; within a run, 25 start times 100 fs apart (notes/decisions.md), each start
frame (forward, even position within the molecule) followed by its time-reversed copy (odd
position: positions 8 fs later, momenta of 8 fs later with the sign flipped). Every molecule
has all 500 frames and starts at a multiple of 500 in the source file.

This script writes new pairs files in the same format where each frame's "future" is the
state 100 fs later along that frame's own direction of time:
  forward frame f    ->  frame f + 2  (the next start frame, 100 fs later)
  reversed copy f    ->  frame f - 2  (the previous reversed copy: for reversed dynamics,
                                       later in its own time means earlier in the run)
so the time-reversal augmentation of the 8 fs pairs is kept, as in E01's pretraining.
Before pairing, every forward frame is checked to be followed by its exact time-reversed copy
(positions equal to its stored future positions, momenta equal to minus its stored future
momenta); the script stops if any is not.

A frame is paired only if its partner lies in the same 50-frame run of the same molecule,
their total energy per atom (catalogue: 1.5 k_B T + potential energy) differs by less than
--max_de eV, and no atom moves more than half the smallest cell length (a jump that large
would mean a wrap across the box; positions are otherwise unwrapped). Every exclusion is
counted, by reason and by temperature band, and printed.
Targets are stored mass-scaled exactly as in data/prepare_organic.py
(future_positions := q + dq * sqrt(m), future_momenta := p' / sqrt(m)), so
train/pretrain_jepa.py reads the files unchanged.

The fine-tuning task is NOT changed: fine-tunes still use data/organic8/train_10pct.xyz (8 fs).

  python data/make_gap_pairs.py data/organic8 train data/organic8_gap100 configs/flashmd_organic8_scratch.yaml
  python data/make_gap_pairs.py data/organic8 val   data/organic8_gap100 configs/flashmd_organic8_scratch.yaml
"""

import argparse
import json
import os
from collections import Counter

import ase.io
import ase.units
import numpy as np
import pandas as pd
import yaml

MOLECULE_FRAMES = 500
RUN_FRAMES = 50


def tband(t):
    return "below 500 K" if t < 500 else ("500 to 1000 K" if t <= 1000 else "above 1000 K")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("split", choices=["train", "val", "test"])
    ap.add_argument("out_dir")
    ap.add_argument("config")
    ap.add_argument("--catalogue", default="results/data/catalogue_8fs.csv")
    ap.add_argument("--max_de", type=float, default=1e-3, help="eV per atom")
    args = ap.parse_args()

    masses = {int(k): float(v) for k, v in yaml.safe_load(open(args.config))["architecture"]["training"]["masses"].items()}
    frames_csv = pd.read_csv(os.path.join(args.src_dir, "frames.csv"))
    rows = frames_csv[(frames_csv.split == args.split) & (frames_csv.written == 1)].reset_index(drop=True)
    cat = pd.read_csv(args.catalogue, usecols=["frame", "group", "kinetic_T_K", "pe_eV_atom"]).set_index("frame")
    src = list(ase.io.iread(os.path.join(args.src_dir, f"{args.split}.xyz"), index=":", format="extxyz"))
    assert len(src) == len(rows), f"{len(src)} frames in the file but {len(rows)} written rows in frames.csv"

    # layout checks: every molecule complete, starting at a multiple of 500 in the source file
    orig = rows.frame.to_numpy()
    first = rows.groupby("group").frame.transform("min").to_numpy()
    sizes = rows.groupby("group").size()
    assert (sizes == MOLECULE_FRAMES).all(), f"molecules without {MOLECULE_FRAMES} frames: {sizes[sizes != MOLECULE_FRAMES].index.tolist()[:5]}"
    rel = orig - first                                   # position within the molecule, 0..499
    pos_of = {int(f): i for i, f in enumerate(orig)}     # source frame id -> position in this split's file

    def sm_of(a):
        return np.sqrt(np.array([masses[int(z)] for z in a.numbers]))[:, None]

    # check: each forward frame is followed by its exact time-reversed copy
    bad = 0
    for i in range(0, len(src), 2):
        a, b = src[i], src[i + 1]
        assert rel[i] % 2 == 0 and rel[i + 1] == rel[i] + 1
        sm = sm_of(a)
        q_fut = a.get_positions() + (a.arrays["future_positions"] - a.get_positions()) / sm
        p_fut = a.arrays["future_momenta"] * sm
        if not (np.allclose(b.get_positions(), q_fut, atol=1e-5) and np.allclose(b.get_momenta(), -p_fut, atol=1e-5)):
            bad += 1
    assert bad == 0, f"{bad} forward frames are not followed by their time-reversed copy; layout differs from the one assumed"

    e_tot = 1.5 * ase.units.kB * cat.kinetic_T_K + cat.pe_eV_atom
    counts = Counter()
    excluded_by_t = Counter()
    out = []
    for i, a in enumerate(src):
        f0 = int(orig[i])
        forward = rel[i] % 2 == 0
        f1 = f0 + 2 if forward else f0 - 2
        t = tband(float(cat.kinetic_T_K[f0]))
        reason = None
        if f1 not in pos_of:
            reason = "partner outside the split"
        elif (rel[i] // RUN_FRAMES) != ((f1 - first[i]) // RUN_FRAMES) or not (0 <= f1 - first[i] < MOLECULE_FRAMES):
            reason = "partner in another run"
        elif abs(e_tot[f0] - e_tot[f1]) > args.max_de:
            reason = "energy differs"
        else:
            b = src[pos_of[f1]]
            if len(b) != len(a) or not np.array_equal(a.numbers, b.numbers):
                reason = "atoms differ"
            else:
                dq = b.get_positions() - a.get_positions()
                limit = 0.5 * min(np.linalg.norm(a.cell.array, axis=1))
                if np.linalg.norm(dq, axis=1).max() > limit:
                    reason = "moved more than half the box"
        if reason:
            counts[reason] += 1
            excluded_by_t[f"{reason} | {t}"] += 1
            continue
        sm = sm_of(a)
        c = a.copy()
        c.arrays["future_positions"] = a.get_positions() + dq * sm
        c.arrays["future_momenta"] = b.get_momenta() / sm
        out.append(c)
        counts["paired forward" if forward else "paired reversed"] += 1

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.split}.xyz")
    ase.io.write(path, out, format="extxyz")
    summary = {"frames": len(src), "counts": dict(counts), "excluded_by_temperature": dict(excluded_by_t), "gap_fs": 100}
    with open(os.path.join(args.out_dir, f"manifest_{args.split}.json"), "w") as fh:
        json.dump({"source": os.path.join(args.src_dir, f"{args.split}.xyz"), "rule": vars(args), **summary}, fh, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"written {path}")


if __name__ == "__main__":
    main()
