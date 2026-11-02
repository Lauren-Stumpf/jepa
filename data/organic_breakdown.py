"""
Break the organic (H C N O only) part of the catalogue down by what kind of
system each structure is, using cell volume per atom as the discriminator:
  dense     < 15 A^3 per atom   (molecular crystal, ice, dense liquid)
  loose     15 to 60             (open framework or small box)
  vacuum    > 60                 (an isolated molecule padded with vacuum)
and by composition (has carbon or not) and size.

Writes a per-structure table and prints counts per class and temperature band.

Usage:
  python data/organic_breakdown.py results/data/catalogue_8fs.csv
"""

import argparse

import numpy as np
import pandas as pd


def classify(vol_per_atom):
    if vol_per_atom != vol_per_atom:  # NaN: no cell
        return "nonperiodic"
    if vol_per_atom < 15:
        return "dense"
    if vol_per_atom < 60:
        return "loose"
    return "vacuum"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    d = d[d["broken"] == 0]
    org = d[d["elements"].apply(lambda s: set(str(s).split("-")) <= {"H", "C", "N", "O"})].copy()
    org["vol_per_atom"] = org["volume_A3"] / org["natoms"]
    org["kind"] = org["vol_per_atom"].apply(classify)
    org["has_C"] = org["elements"].str.contains("C")
    org["is_water"] = org["elements"] == "H-O"

    per = org.groupby("group").agg(
        formula=("formula", "first"), natoms=("natoms", "first"),
        vol_per_atom=("vol_per_atom", "first"), kind=("kind", "first"),
        has_C=("has_C", "first"), is_water=("is_water", "first"),
        n_frames=("frame", "size"),
        n_frames_250_350=("kinetic_T_K", lambda t: int(((t >= 250) & (t < 350)).sum())),
        T_min=("kinetic_T_K", "min"), T_max=("kinetic_T_K", "max"),
    ).reset_index()
    out = args.csv.replace(".csv", "_organic_structures.csv")
    per.to_csv(out, index=False)

    print(f"organic structures: {len(per)}, frames: {len(org)}")
    print("\nby kind (volume per atom):")
    print(per.groupby("kind").agg(structures=("group", "size"), frames=("n_frames", "sum"),
                                  frames_250_350=("n_frames_250_350", "sum"),
                                  median_atoms=("natoms", "median")).to_string())
    print("\nby composition:")
    comp = per.assign(comp=np.where(per["is_water"], "water", np.where(per["has_C"], "carbon-containing", "other H/N/O")))
    print(comp.groupby(["comp", "kind"]).agg(structures=("group", "size"), frames=("n_frames", "sum"),
                                              frames_250_350=("n_frames_250_350", "sum")).to_string())
    print("\ncarbon-containing, >= 10 atoms, by kind:")
    cc = per[per["has_C"] & (per["natoms"] >= 10)]
    print(cc.groupby("kind").agg(structures=("group", "size"), frames=("n_frames", "sum"),
                                 frames_250_350=("n_frames_250_350", "sum"),
                                 median_atoms=("natoms", "median")).to_string())
    print("\nexample vacuum-type structures (isolated molecules):")
    print(per[per["kind"] == "vacuum"].sort_values("natoms").head(20)[["formula", "natoms", "vol_per_atom", "n_frames_250_350"]].to_string(index=False))
    print("\nwater structures:")
    print(per[per["is_water"]][["formula", "natoms", "vol_per_atom", "kind", "n_frames_250_350", "T_min", "T_max"]].to_string(index=False))
    print("\nwritten", out)


if __name__ == "__main__":
    main()
