"""
How often do an atom's neighbours change over a few 16 fs steps in liquid water? (Read-only
measurement on the exact trajectories, to choose horizons for a first latent-rollout test at
which the neighbour list can be treated as fixed.)

For start frames every --spacing fs in the runs of one split of the manifest, the set of atom
pairs closer than the cutoff (nearest periodic copy) is compared with the set k steps later,
k in --steps. Reported per k:
  lost     share of the start frame's pairs that are no longer within the cutoff
  gained   pairs within the cutoff later that were not at the start, as a share of the start pairs
  atoms    share of atoms with at least one neighbour lost or gained
Also the mean number of neighbours per atom at the start.

  python eval/neighbour_change.py /storage3/lauren/data/water --out results/eval/neighbour_change_water.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from checks.check_water_run_io import read_ipi_xyz  # noqa: E402

DT_FS = 0.25
STEP_FS = 16.0


def pairs_within(q, box, cutoff):
    """Boolean (N, N) matrix: pair closer than the cutoff, nearest periodic copy, no self pairs."""
    d = q[:, None, :] - q[None, :, :]
    d -= box * np.round(d / box)
    close = (d ** 2).sum(-1) < cutoff ** 2
    np.fill_diagonal(close, False)
    return close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root")
    ap.add_argument("--water_manifest", default="data/water/pairs/16fs/manifest.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_runs", type=int, default=6)
    ap.add_argument("--cutoff", type=float, default=4.5, help="the model's cutoff in A")
    ap.add_argument("--spacing", type=float, default=200.0, help="fs between start frames")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = json.load(open(args.water_manifest))["split"]["runs"][args.split][:args.max_runs]
    per_step = int(round(STEP_FS / DT_FS))
    acc = {k: {"lost": [], "gained": [], "atoms": []} for k in args.steps}
    n_neigh = []
    for name in names:
        q, _, cells = read_ipi_xyz(os.path.join(args.runs_root, name, "nve.pos_0.xyz"))
        box = np.diag(np.asarray(cells[0])) if np.ndim(cells[0]) == 2 else np.asarray(cells[0])
        for t0 in range(0, len(q) - max(args.steps) * per_step, int(round(args.spacing / DT_FS))):
            a = pairs_within(q[t0], box, args.cutoff)
            n_neigh.append(a.sum(1).mean())
            for k in args.steps:
                b = pairs_within(q[t0 + k * per_step], box, args.cutoff)
                acc[k]["lost"].append((a & ~b).sum() / a.sum())
                acc[k]["gained"].append((b & ~a).sum() / a.sum())
                acc[k]["atoms"].append((a != b).any(1).mean())
        print(f"{name} done", flush=True)
    res = {"runs": names, "cutoff_A": args.cutoff, "start_frames": len(n_neigh),
           "neighbours_per_atom": float(np.mean(n_neigh)),
           "by_steps": {str(k): {m: float(np.mean(v)) for m, v in acc[k].items()} for k in args.steps}}
    for k in args.steps:
        r = res["by_steps"][str(k)]
        print(f"{k:3d} steps ({k * STEP_FS:5.0f} fs): pairs lost {100 * r['lost']:.2f}%, gained {100 * r['gained']:.2f}%, "
              f"atoms with a changed neighbour {100 * r['atoms']:.1f}%")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
