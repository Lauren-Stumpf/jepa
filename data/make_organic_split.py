"""
Build the organic-molecule training set from the Zenodo 8 fs file.

Selection (from the per-structure table written by data/organic_breakdown.py):
  - elements within H, C, N, O
  - contains carbon
  - at least --min_atoms atoms (default 10)
  - "vacuum" kind: an isolated molecule in a large periodic box (> 60 A^3/atom)
  - broken frames (exploded reference runs) dropped

Split by structure, so all 500 frames of one molecule land in one split.
Writes train/val/test xyz files in the format metatrain's FlashMD expects,
with the TARGETS STORED MASS-SCALED as in the FlashMD paper (Eq. 6):
    future_positions := positions + (q_future - q_now) * sqrt(m)
    future_momenta   := p_future / sqrt(m)
while the input "momenta" column stays raw. With these targets metatrain's
pipeline (PositionAdditive subtracts positions, the scaler divides by the
dataset standard deviation, MSE loss) computes exactly the paper's loss.
Anything that reads a trained model must undo the scaling (eval/eval_onestep.py
does). Raw targets can be kept with --raw_targets.

Molecules stored split across the periodic boundary are UNWRAPPED: atoms are
placed by walking the bond graph with minimum-image offsets, so the whole
molecule sits on one side of the box (an exact operation for a periodic
system). The same shift is applied to the atom's future position. A frame is
dropped only if the walk cannot reach every atom (fragmented molecule) or if
an atom's stored displacement exceeds --max_atom_disp (a jump that is not a
wrap).

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
import numpy as np
import pandas as pd


def unwrap_molecule(q, cell_lengths, bond_cut=1.9):
    """Shift atoms by box vectors so the molecule is connected under no wrapping.

    Returns (shifted positions, shift per atom in A, all_reached flag)."""
    n = len(q)
    out = q.copy()
    shift = np.zeros_like(q)
    placed = np.zeros(n, dtype=bool)
    placed[0] = True
    frontier = [0]
    while frontier:
        i = frontier.pop()
        d = q - q[i]                       # raw offsets to every atom
        img = np.round(d / cell_lengths)   # which periodic image is nearest
        dmin = d - img * cell_lengths
        close = (np.linalg.norm(dmin, axis=1) < bond_cut) & ~placed
        for j in np.where(close)[0]:
            # place j next to the already-placed i, carrying i's accumulated shift
            shift[j] = shift[i] - img[j] * cell_lengths
            out[j] = q[j] + shift[j]
            placed[j] = True
            frontier.append(j)
    return out, shift, bool(placed.all())


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
    ap.add_argument("--raw_targets", action="store_true", help="store raw future positions/momenta")
    ap.add_argument("--max_atom_disp", type=float, default=2.0,
                    help="drop a frame if any atom moves more than this (A) over the stride")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    per = pd.read_csv(args.structures)
    sel = per[per["has_C"] & (per["natoms"] >= args.min_atoms)]
    if args.kind != "all":
        sel = sel[sel["kind"] == args.kind]
    # The unit of the split is the MOLECULE, identified by (formula, atom count,
    # cell volume): the file holds some molecules twice as separate groups, and
    # both copies must land in the same split or the held-out sets leak.
    sel = sel.copy()
    sel["molecule"] = list(zip(sel["formula"], sel["natoms"].astype(int),
                               (sel["vol_per_atom"] * sel["natoms"]).round(1)))
    molecules = sorted(set(sel["molecule"]))
    rng = random.Random(args.seed)
    rng.shuffle(molecules)
    n_val = int(round(len(molecules) * args.val_frac))
    n_test = int(round(len(molecules) * args.test_frac))
    mol_split = {}
    for m in molecules[:n_val]:
        mol_split[m] = "val"
    for m in molecules[n_val:n_val + n_test]:
        mol_split[m] = "test"
    for m in molecules[n_val + n_test:]:
        mol_split[m] = "train"
    split = {int(g): mol_split[m] for g, m in zip(sel["group"], sel["molecule"])}
    groups = sorted(split)
    dup = len(sel) - len(molecules)
    print(f"{len(sel)} structures = {len(molecules)} distinct molecules ({dup} duplicate copies): "
          f"{sum(v == 'train' for v in mol_split.values())} train, {n_val} val, {n_test} test molecules")

    cat = pd.read_csv(args.catalogue, usecols=["frame", "group", "broken"])
    cat = cat[cat["group"].isin(list(split))]
    keep = cat[cat["broken"] == 0]
    frame_split = dict(zip(keep["frame"].astype(int), keep["group"].map(split)))
    last_frame = max(frame_split)
    print(f"{len(frame_split)} frames to write ({int(cat['broken'].sum())} broken dropped)")

    writers = {s: open(os.path.join(args.out_dir, f"{s}.xyz"), "w") for s in ("train", "val", "test")}
    counts = {s: 0 for s in writers}
    dropped = {"jump": 0, "fragmented": 0}
    frag_by_formula = {}
    unwrapped = 0
    for i, atoms in enumerate(ase.io.iread(args.xyz, index=":", format="extxyz")):
        if i > last_frame:
            break
        s = frame_split.get(i)
        if s is None:
            continue
        q = atoms.get_positions()
        fut = atoms.arrays["future_positions"]
        if atoms.pbc.any():
            Lc = atoms.cell.lengths()
            if ((q.max(0) - q.min(0)) > 0.5 * Lc).any():
                q_new, shift, complete = unwrap_molecule(q, Lc)
                if not complete:
                    dropped["fragmented"] += 1
                    frag_by_formula[atoms.get_chemical_formula()] = frag_by_formula.get(atoms.get_chemical_formula(), 0) + 1
                    continue
                unwrapped += 1
                q = q_new
                fut = fut + shift
                atoms.set_positions(q)
                atoms.arrays["future_positions"] = fut
        dq = fut - q
        if np.abs(dq).max() > args.max_atom_disp:
            dropped["jump"] += 1
            continue
        if not args.raw_targets:
            sm = np.sqrt(atoms.get_masses())[:, None]
            atoms.arrays["future_positions"] = q + dq * sm
            atoms.arrays["future_momenta"] = atoms.arrays["future_momenta"] / sm
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

    print(f"unwrapped {unwrapped} frames whose molecule straddled the box; dropped: {dropped}")
    if frag_by_formula:
        top = sorted(frag_by_formula.items(), key=lambda kv: -kv[1])[:10]
        print("fragmented (unreachable atoms) by molecule, top 10:", top)
    manifest = {
        "source": args.xyz, "seed": args.seed, "kind": args.kind, "min_atoms": args.min_atoms,
        "targets": "raw" if args.raw_targets else "mass_scaled",
        "max_atom_disp": args.max_atom_disp, "dropped": dropped,
        "unwrapped_frames": unwrapped, "fragmented_by_formula": frag_by_formula,
        "frames": counts,
        "structures": {s: sorted(int(g) for g, v in split.items() if v == s) for s in counts},
        "formulas": {s: sorted(set(per[per["group"].isin([g for g, v in split.items() if v == s])]["formula"]))
                     for s in counts},
    }
    # leakage check: the same molecule (formula, atoms, cell volume) must not sit in two splits
    sel_idx = sel.set_index("group")
    sig_split = {}
    for g, sname in split.items():
        r = sel_idx.loc[g]
        sig = (r["formula"], int(r["natoms"]), round(float(r["vol_per_atom"]) * int(r["natoms"]), 1))
        sig_split.setdefault(sig, set()).add(sname)
    leaks = {k: v for k, v in sig_split.items() if len(v) > 1}
    if leaks:
        print("WARNING: identical molecules in more than one split:", leaks)
    manifest["leak_warnings"] = [list(k) + [sorted(v)] for k, v in leaks.items()]
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("written", counts, "->", args.out_dir)


if __name__ == "__main__":
    main()
