"""
Look inside a few structure groups of the catalogue to understand how the
frames of one structure are organised: how many runs, how many frames per run,
where the time-reversed copies sit.

Within one constant-energy run the kinetic temperature stays close to one
value, and each run was started at a different random temperature, so
temperature acts as a fingerprint for the run a frame came from. A
time-reversed copy of a frame has identical kinetic temperature and identical
stored displacement size (same positions, momenta negated, start and end
swapped).

Usage:
  python data/inspect_groups.py results/data/catalogue_8fs.csv --groups 0 1 2 500 5000
"""

import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--groups", type=int, nargs="*", default=[0, 1, 2, 500, 5000])
    ap.add_argument("--show", type=int, default=60, help="frames to print per group")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)

    sizes = d.groupby("group").size()
    print("frames per group:", sizes.describe()[["min", "50%", "max"]].to_dict())
    print("groups with exactly 500 frames:", int((sizes == 500).sum()), "of", len(sizes))

    for g in args.groups:
        s = d[d["group"] == g].reset_index(drop=True)
        print(f"\n=== group {g}: {s['formula'].iloc[0]}, {s['natoms'].iloc[0]} atoms, "
              f"{len(s)} frames, periodic={s['periodic'].iloc[0]}")
        t = s["kinetic_T_K"].values
        # a new run starts where the temperature jumps by more than fluctuations allow
        jumps = np.where(np.abs(np.diff(t)) > 0.25 * np.maximum(t[:-1], 50.0))[0] + 1
        starts = np.concatenate([[0], jumps])
        print(f"  apparent runs (by temperature jumps): {len(starts)}, "
              f"frames per run: {np.diff(np.append(starts, len(s))).tolist()[:20]}")
        print(f"  run temperatures (K): {[round(float(t[i])) for i in starts][:20]}")
        # time-reversed pairs: identical T and displacement, usually adjacent
        dup = s.duplicated(subset=["kinetic_T_K", "disp_ms_rms"], keep=False)
        print(f"  frames with an identical (T, displacement) twin: {int(dup.sum())} of {len(s)}")
        print("  first frames (frame, T_K, disp_ms_rms, pe_eV_atom):")
        for _, r in s.head(args.show).iterrows():
            print(f"    {int(r['frame']):8d} {r['kinetic_T_K']:8.1f} {r['disp_ms_rms']:8.4f} {r['pe_eV_atom']:10.5f}")


if __name__ == "__main__":
    main()
