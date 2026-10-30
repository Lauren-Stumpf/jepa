"""
Summarise a catalogue CSV written by data/catalogue.py.

Prints: overall counts, broken frames, periodic vs non-periodic, element
coverage, and a focused section on organic molecular fragments (non-periodic,
elements within H C N O), with size and temperature distributions and the
most common formulas.

Usage:
  python data/summarise_catalogue.py results/data/catalogue_8fs.csv
"""

import argparse

import numpy as np
import pandas as pd


def describe(name, d):
    print(f"\n== {name}: {len(d)} frames, {d['group'].nunique()} structures")
    if len(d) == 0:
        return
    print("  atoms per structure: ", d.groupby("group")["natoms"].first().describe()[["min", "25%", "50%", "75%", "max"]].to_dict())
    t = d["kinetic_T_K"]
    bands = [(0, 250), (250, 350), (350, 600), (600, 1000), (1000, 1e9)]
    print("  frames by temperature band:",
          {f"{lo}-{hi if hi < 1e9 else 'inf'}": int(((t >= lo) & (t < hi)).sum()) for lo, hi in bands})
    print("  structures with at least one frame in 250-350 K:",
          d[(t >= 250) & (t < 350)]["group"].nunique())
    print("  most common formulas:", d.groupby("group")["formula"].first().value_counts().head(15).to_dict())
    print("  stored displacement rms (mass-scaled): median %.3f, 90th pct %.3f" %
          (d["disp_ms_rms"].median(), d["disp_ms_rms"].quantile(0.9)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    print(f"total frames {len(d)}, structure groups {d['group'].nunique()}, "
          f"frames per group median {d.groupby('group').size().median():.0f}")
    print(f"broken frames {int(d['broken'].sum())} in {d[d['broken'] == 1]['group'].nunique()} groups")
    print("periodic frames", int(d["periodic"].sum()), "non-periodic", int((d["periodic"] == 0).sum()))
    extra = d["extra_info"].dropna()
    extra = extra[extra != ""]
    print("frames with extra header fields:", len(extra),
          ("example: " + str(extra.iloc[0])) if len(extra) else "(none: no subset label stored)")

    d = d[d["broken"] == 0]
    organic = d[d["elements"].apply(lambda s: set(s.split("-")) <= {"H", "C", "N", "O"})]
    describe("organic (H C N O only), all", organic)
    describe("organic molecular fragments (non-periodic)", organic[organic["periodic"] == 0])
    describe("organic periodic (molecular crystals etc.)", organic[organic["periodic"] == 1])
    frag = organic[organic["periodic"] == 0]
    frag.to_csv(args.csv.replace(".csv", "_organic_fragments.csv"), index=False)
    print("\norganic fragment rows written to", args.csv.replace(".csv", "_organic_fragments.csv"))


if __name__ == "__main__":
    main()
