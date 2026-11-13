"""
Organic momentum error split by the kind of frame (E09, hypothesis H3): does pretraining lose
to the baseline everywhere, or mainly on unusual frames?

Frame kinds, from data/organic8/frames.csv (written by data/prepare_organic.py, one row per
source frame):
  normal        one piece, does not see its own periodic image
  apart         the molecule is in more than one piece (n_pieces > 1)
  own image     an atom is within the cutoff of its own periodic image (n_image_pairs > 0)
and, separately, by kinetic temperature: below 500 K, 500 to 1,000 K, above 1,000 K.

Inputs are the per-frame result files of eval/eval_onestep.py, whose "frame" column is the
frame's position in the evaluated file; the position is mapped to its frames.csv row through
the rows of that split with written == 1, in order. The mapping is checked on every frame:
the atom count and the formula must agree.

Errors are pooled over atoms as the evaluation does: sqrt(sum natoms * rmse^2 / sum natoms).

  python eval/errors_by_frame_type.py --set test     # results/eval/test/<run>.csv on test.xyz
  python eval/errors_by_frame_type.py --set unused   # results/eval/seen/<run>_unused.csv on train.xyz
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd

ARMS = ["baseline", "plain", "normtarget", "resid", "proj", "residrel"]


def run_name(arm, seed):
    return f"flashmd_organic8_scratch_10pct_s{seed}" if arm == "baseline" else f"jepa_finetune_{arm}_organic8_10pct_s{seed}"


def formula_key(f):
    """Element counts from a formula string, so 'C6H12N2' and 'H12C6N2' compare equal."""
    import re
    return Counter({el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f)})


def pooled(df, col="rmse_dp_massscaled"):
    return float(np.sqrt((df.natoms * df[col] ** 2).sum() / df.natoms.sum())) if len(df) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["test", "unused", "unused_fwd", "labelled"], required=True)
    ap.add_argument("--frames_csv", default="data/organic8/frames.csv")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    split = "test" if args.set == "test" else "train"
    rows = pd.read_csv(args.frames_csv)
    rows = rows[(rows.split == split) & (rows.written == 1)].reset_index(drop=True)

    def kinds(r):
        k = "normal"
        if r.n_pieces > 1:
            k = "apart"
        elif r.n_image_pairs > 0:
            k = "own image"
        return k

    def tband(t):
        return "below 500 K" if t < 500 else ("500 to 1000 K" if t <= 1000 else "above 1000 K")

    res = {"set": args.set, "arms": {}}
    counts_done = False
    for arm in ARMS:
        per_seed = []
        for s in (0, 1, 2):
            r = run_name(arm, s)
            path = f"results/eval/test/{r}.csv" if args.set == "test" else f"results/eval/seen/{r}_{args.set}.csv"
            d = pd.read_csv(path)
            info = rows.iloc[d.frame.to_numpy()].reset_index(drop=True)
            assert (info.formula.map(formula_key) == d.formula.map(formula_key)).all(), f"{path}: formula mismatch"
            d["kind"] = [kinds(x) for x in info.itertuples()]
            d["tband"] = [tband(t) for t in info.kinetic_T_K]
            rel = info.frame.to_numpy() - rows.groupby("group").frame.transform("min").to_numpy()[d.frame.to_numpy()]
            d["direction"] = np.where(rel % 2 == 0, "forward", "reversed")
            if not counts_done:
                res["frames_by_kind"] = d.kind.value_counts().to_dict()
                res["frames_by_tband"] = d.tband.value_counts().to_dict()
                res["frames_by_direction"] = d.direction.value_counts().to_dict()
                counts_done = True
            out = {"all": pooled(d)}
            for k, g in d.groupby("kind"):
                out[k] = pooled(g)
            for k, g in d.groupby("tband"):
                out[k] = pooled(g)
            for k, g in d.groupby("direction"):
                out[k] = pooled(g)
            per_seed.append(out)
        keys = per_seed[0].keys()
        res["arms"][arm] = {k: {"seeds": [p.get(k) for p in per_seed], "mean": float(np.mean([p.get(k) for p in per_seed]))}
                            for k in keys}
    base = res["arms"]["baseline"]
    print(f"set: {args.set}; frames by kind {res['frames_by_kind']}; by temperature {res['frames_by_tband']}")
    cats = list(base.keys())
    print("momentum error (mean of 3 seeds), and % against the baseline")
    print(f"{'arm':12s}" + "".join(f"{c:>22s}" for c in cats))
    for arm in ARMS:
        a = res["arms"][arm]
        cells = []
        for c in cats:
            m = a[c]["mean"]
            pct = "" if arm == "baseline" else f" ({100 * (m / base[c]['mean'] - 1):+.1f}%)"
            cells.append(f"{m:.4f}{pct}")
        print(f"{arm:12s}" + "".join(f"{x:>22s}" for x in cells))
    out = args.out or f"results/eval/errors_by_frame_type_{args.set}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
