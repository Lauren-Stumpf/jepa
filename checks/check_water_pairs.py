"""
Check the parsed water pairs (data/water/parse_runs.py output) against the raw
i-PI trajectories they came from. Prints PASS or FAIL per check.

  1. every run listed in the manifest is complete and used exactly once, in one split
  2. the pair count per run and per split matches the manifest and the hashes match the files
  3. for a random sample of forward pairs, undoing the mass scaling and the
     unit conversion recovers the trajectory frame exactly stride fs later
  4. every time-reversed copy starts at its forward pair's future with
     negated momenta and lands on the forward pair's start with negated momenta
  5. kinetic temperature of the inputs is near 300 K with the right spread
  6. no frame in train/val/test holds a position displacement that crosses the box

Usage:
  python checks/check_water_pairs.py data/water/pairs/16fs /storage3/lauren/data/water
"""

import argparse
import hashlib
import json
import os
import sys

import ase.io
import ase.units as U
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from checks.check_water_run_io import P_AU_TO_ASE, read_ipi_xyz  # noqa: E402

DT_FS = 0.25


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def report(ok, name, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs_dir")
    ap.add_argument("runs_root")
    ap.add_argument("--sample_runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    m = json.load(open(os.path.join(args.pairs_dir, "manifest.json")))
    masses = {int(k): v for k, v in m["masses"].items()}
    stride_steps = int(round(m["stride_fs"] / DT_FS))
    spacing_steps = int(round(m["spacing_fs"] / DT_FS))
    per_run = m["pairs_per_run"]
    difference = bool(m.get("momenta_as_difference", False))

    def future_p(a):   # undo the stored momentum target
        sm = np.sqrt([masses[int(z)] for z in a.numbers])[:, None]
        return a.get_momenta() + (a.arrays["future_momenta"] - a.get_momenta()) * sm if difference else a.arrays["future_momenta"] * sm
    all_ok = True

    runs = m["split"]["runs"]
    listed = sum(runs.values(), [])
    all_ok &= report(len(listed) == len(set(listed)) and not (set(listed) & set(m["skipped_runs"])),
                     "runs", f"{len(listed)} runs, each in exactly one split; {len(m['skipped_runs'])} skipped")

    frames = {s: ase.io.read(os.path.join(args.pairs_dir, f"{s}.xyz"), index=":") if os.path.getsize(os.path.join(args.pairs_dir, f"{s}.xyz")) else []
              for s in ("train", "val", "test")}
    counts_ok = all(len(frames[s]) == m["pairs"][s] == per_run * len(runs[s]) for s in frames)
    hashes_ok = all(sha256(os.path.join(args.pairs_dir, f"{s}.xyz")) == m["sha256"][f"{s}.xyz"] for s in frames)
    all_ok &= report(counts_ok and hashes_ok, "counts and hashes",
                     f"{ {s: len(f) for s, f in frames.items()} } pairs, {per_run} per run; hashes {'match' if hashes_ok else 'differ'}")

    rng = np.random.default_rng(args.seed)
    worst_q = worst_p = 0.0
    worst_rev = 0.0
    n_checked = 0
    for s in frames:
        if not runs[s]:
            continue
        for name in rng.choice(runs[s], size=min(args.sample_runs, len(runs[s])), replace=False):
            q, _, cells = read_ipi_xyz(os.path.join(args.runs_root, name, "nve.pos_0.xyz"))
            p, _, _ = read_ipi_xyz(os.path.join(args.runs_root, name, "nve.mom_0.xyz"))
            p = p * P_AU_TO_ASE
            k0 = runs[s].index(name) * per_run   # pairs of one run are contiguous, forward then reversed
            for j, t0 in enumerate(range(0, len(q) - stride_steps, spacing_steps)):
                fwd, rev = frames[s][k0 + 2 * j], frames[s][k0 + 2 * j + 1]
                sm = np.sqrt([masses[int(z)] for z in fwd.numbers])[:, None]
                t1 = t0 + stride_steps
                q1 = fwd.get_positions() + (fwd.arrays["future_positions"] - fwd.get_positions()) / sm
                p1 = future_p(fwd)
                worst_q = max(worst_q, np.abs(fwd.get_positions() - q[t0]).max(), np.abs(q1 - q[t1]).max())
                worst_p = max(worst_p, np.abs(fwd.get_momenta() - p[t0]).max(), np.abs(p1 - p[t1]).max())
                q0r = rev.get_positions() + (rev.arrays["future_positions"] - rev.get_positions()) / sm
                worst_rev = max(worst_rev, np.abs(rev.get_positions() - q1).max(), np.abs(rev.get_momenta() + p1).max(),
                                np.abs(q0r - fwd.get_positions()).max(), np.abs(future_p(rev) + fwd.get_momenta()).max())
                n_checked += 1
    all_ok &= report(worst_q < 1e-6 and worst_p < 1e-6, "pairs match the trajectory",
                     f"{n_checked} forward pairs; largest position error {worst_q:.1e} A, momentum {worst_p:.1e}")
    all_ok &= report(worst_rev < 1e-6, "time-reversed copies", f"largest inconsistency {worst_rev:.1e}")

    allf = sum(frames.values(), [])
    T = np.array([2 * f.get_kinetic_energy() / (3 * len(f) * U.kB) for f in allf])
    expected_sd = 300.0 * np.sqrt(2.0 / (3 * len(allf[0])))
    all_ok &= report(abs(T.mean() - 300) < 15 and 0.5 * expected_sd < T.std() < 2.5 * expected_sd, "input temperature",
                     f"mean {T.mean():.1f} K, sd {T.std():.1f} K (expected sd {expected_sd:.1f} K for {len(allf[0])} atoms)")

    half = min(f.cell.lengths().min() for f in allf) * 0.5
    worst_disp = max(np.linalg.norm((f.arrays["future_positions"] - f.get_positions())
                                    / np.sqrt([masses[int(z)] for z in f.numbers])[:, None], axis=1).max() for f in allf)
    all_ok &= report(worst_disp < half, "no displacement crosses the box", f"largest {worst_disp:.3f} A, half box {half:.2f} A")
    print(f"momentum target: {'change (predict_momenta_as_difference)' if difference else 'absolute'}")
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
