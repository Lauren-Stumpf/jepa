"""
Build the organic-molecule training set from the Zenodo 8 fs file.

Selection (from the per-structure table written by data/organic_breakdown.py):
  - elements within H, C, N, O
  - contains carbon
  - at least --min_atoms atoms (default 10)
  - "vacuum" kind: an isolated molecule in a large periodic box (> 60 A^3/atom)
  - broken frames (exploded reference runs) dropped

Split by structure, so all 500 frames of one molecule land in one split.
Writes train/val/test xyz files in exactly the format metatrain's FlashMD
expects (positions, momenta, future_positions, future_momenta per atom), plus
a JSON manifest listing which structure groups went where.

Usage:
  python data/make_organic_split.py data/universal_8fs.xyz \
      results/data/catalogue_8fs.csv results/data/catalogue_8fs_organic_structures.csv \
      --out_dir data/organic8 --seed 0
"""

import argparse
import json
import os
import random

import ase.io
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("catalogue")
    ap.add_argument("structures")
    ap.add_argument("--out_dir", default="data/organic8")
    ap.add_argument("--min_atoms", type=int, default=10)
    ap.add_argument("--kind", default="vacuum", choices=["vacuum", "dense", "loose", "all"])
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per = pd.read_csv(args.structures)
    sel = per[per["has_C"] & (per["natoms"] >= args.min_atoms)]
    if args.kind != "all":
        sel = sel[sel["kind"] == args.kind]
    groups = sorted(sel["group"].tolist())
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    n_val = int(round(len(groups) * args.val_frac))
    n_test = int(round(len(groups) * args.test_frac))
    split = {}
    for g in groups[:n_val]:
        split[g] = "val"
    for g in groups[n_val:n_val + n_test]:
        split[g] = "test"
    for g in groups[n_val + n_test:]:
        split[g] = "train"
    print(f"{len(groups)} structures selected: "
          f"{sum(v == 'train' for v in split.values())} train, {n_val} val, {n_test} test")

    cat = pd.read_csv(args.catalogue, usecols=["frame", "group", "broken"])
    cat = cat[cat["group"].isin(split)]
    keep = cat[cat["broken"] == 0]
    frame_split = dict(zip(keep["frame"].astype(int), keep["group"].map(split)))
    last_frame = max(frame_split)
    print(f"{len(frame_split)} frames to write ({int(cat['broken'].sum())} broken dropped)")

    writers = {s: open(os.path.join(args.out_dir, f"{s}.xyz"), "w") for s in ("train", "val", "test")}
    counts = {s: 0 for s in writers}
    for i, atoms in enumerate(ase.io.iread(args.xyz, index=":", format="extxyz")):
        if i > last_frame:
            break
        s = frame_split.get(i)
        if s is None:
            continue
        # keep only what training needs; drop stored forces/energies/stress
        for k in list(atoms.info):
            if k not in ("Lattice", "pbc"):
                del atoms.info[k]
        for k in list(atoms.arrays):
            if k not in ("numbers", "positions", "masses", "momenta",
                         "future_positions", "future_momenta"):
                del atoms.arrays[k]
        atoms.calc = None
        ase.io.write(writers[s], atoms, format="extxyz")
        counts[s] += 1
        if (i + 1) % 500000 == 0:
            print(f"  scanned {i + 1} frames, written {counts}", flush=True)
    for w in writers.values():
        w.close()

    manifest = {
        "source": args.xyz, "seed": args.seed, "kind": args.kind, "min_atoms": args.min_atoms,
        "frames": counts,
        "structures": {s: sorted(int(g) for g, v in split.items() if v == s) for s in counts},
        "formulas": {s: sorted(set(per[per["group"].isin([g for g, v in split.items() if v == s])]["formula"]))
                     for s in counts},
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("written", counts, "->", args.out_dir)


if __name__ == "__main__":
    main()
