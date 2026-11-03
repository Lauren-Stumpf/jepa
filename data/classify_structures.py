"""
Classify each candidate organic structure ONCE, so the training-set selection
rule is explicit and recorded, and the split script does no geometry beyond
unwrapping.

For every structure in the per-structure table (from data/organic_breakdown.py)
this reads its coldest and hottest frames from the Zenodo file and records:
  box_edge_A          cell lengths
  n_fragments_cold    connected pieces (nearest-image bond walk, --bond_cut)
                      in the coldest frame
  n_fragments_hot     the same in the hottest frame
  extent_cold_A       largest extent of the (unwrapped) structure, coldest frame
  gap_to_image_A      box edge minus extent (per axis min); < model cutoff
                      means the model sees the structure's own image
  class               one of
                        molecule          one fragment, gap >= model cutoff
                        molecule_hot_frag one fragment cold, several when hot
                        multi_fragment    several fragments even when cold
                        self_image        gap to own image below model cutoff

Usage:
  python data/classify_structures.py data/universal_8fs.xyz \
      results/data/catalogue_8fs.csv results/data/catalogue_8fs_organic_structures.csv \
      --out results/data/structure_classes.csv
"""

import argparse
import sys
import os

import ase.io
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_organic_split import nearest_image  # noqa: E402


def fragments(q, cell, bond_cut):
    """Connected components under nearest-image bonding; returns list of index arrays
    and the unwrapped positions (each fragment placed rigidly)."""
    n = len(q)
    placed = np.zeros(n, dtype=bool)
    out = q.copy()
    comps = []
    for start in range(n):
        if placed[start]:
            continue
        placed[start] = True
        comp, frontier = [start], [start]
        while frontier:
            i = frontier.pop()
            dmin = nearest_image(q - q[i], cell)
            close = (np.linalg.norm(dmin, axis=1) < bond_cut) & ~placed
            for j in np.where(close)[0]:
                out[j] = out[i] + dmin[j]
                placed[j] = True
                comp.append(j)
                frontier.append(j)
        comps.append(np.array(comp))
    return comps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("catalogue")
    ap.add_argument("structures")
    ap.add_argument("--out", default="results/data/structure_classes.csv")
    ap.add_argument("--bond_cut", type=float, default=3.0)
    ap.add_argument("--model_cutoff", type=float, default=4.5)
    ap.add_argument("--kind", default="vacuum")
    ap.add_argument("--min_atoms", type=int, default=10)
    args = ap.parse_args()

    per = pd.read_csv(args.structures)
    sel = per[per["has_C"] & (per["natoms"] >= args.min_atoms)]
    if args.kind != "all":
        sel = sel[sel["kind"] == args.kind]
    cat = pd.read_csv(args.catalogue, usecols=["frame", "group", "kinetic_T_K", "broken"])
    cat = cat[cat["group"].isin(list(sel["group"])) & (cat["broken"] == 0)]
    # coldest and hottest frame per group
    cold = cat.loc[cat.groupby("group")["kinetic_T_K"].idxmin()].set_index("group")["frame"]
    hot = cat.loc[cat.groupby("group")["kinetic_T_K"].idxmax()].set_index("group")["frame"]
    wanted = {}
    for g in sel["group"]:
        wanted[int(cold[g])] = (int(g), "cold")
        wanted[int(hot[g])] = (int(g), "hot")
    last = max(wanted)

    rows = {}
    for i, atoms in enumerate(ase.io.iread(args.xyz, index=":", format="extxyz")):
        if i > last:
            break
        if i not in wanted:
            continue
        g, which = wanted[i]
        comps, out = fragments(atoms.get_positions(), atoms.cell.array, args.bond_cut)
        L = atoms.cell.lengths()
        r = rows.setdefault(g, {"group": g, "formula": atoms.get_chemical_formula(),
                                "natoms": len(atoms), "box_edge_A": float(L.min())})
        r[f"n_fragments_{which}"] = len(comps)
        r[f"T_{which}_K"] = float(2 * atoms.get_kinetic_energy() / (3 * len(atoms) * 8.617333e-5))
        if which == "cold":
            big = max(comps, key=len)
            ext = out[big].max(0) - out[big].min(0)
            r["extent_cold_A"] = float(ext.max())
            r["gap_to_image_A"] = float((L - ext).min())
            r["largest_fragment_atoms"] = int(len(big))

    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["group"]))

    def classify(r):
        if r["n_fragments_cold"] > 1:
            return "multi_fragment"
        if r["gap_to_image_A"] < args.model_cutoff:
            return "self_image"
        if r["n_fragments_hot"] > 1:
            return "molecule_hot_frag"
        return "molecule"

    df["class"] = df.apply(classify, axis=1)
    df.to_csv(args.out, index=False)
    print(df["class"].value_counts().to_string())
    print("\nexcluded structures (not a single connected molecule, or within model cutoff of own image):")
    print(df[df["class"].isin(["multi_fragment", "self_image"])][
        ["formula", "natoms", "box_edge_A", "extent_cold_A", "gap_to_image_A", "n_fragments_cold", "n_fragments_hot"]].to_string(index=False))
    print("\nmolecules that fragment only when hot (kept, all frames):")
    print(df[df["class"] == "molecule_hot_frag"][["formula", "natoms", "T_hot_K", "n_fragments_hot"]].to_string(index=False))
    print("\nwritten", args.out)


if __name__ == "__main__":
    main()
