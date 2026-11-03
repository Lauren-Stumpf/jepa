"""
Turn finished q-TIP4P/f water runs (data/water/run_one.sh output) into FlashMD
training pairs, one dataset per stride, in the same format as data/organic8.

Following the FlashMD water protocol: from each 2 ps constant-energy run a pair
(now, future) is taken every --spacing fs (default 100 fs), the future being
--stride fs later; each pair also contributes its time-reversed copy
(future with reversed momenta -> now with reversed momenta). Runs are split
randomly 80/10/10 so that no run contributes to two splits (frames within one
run are correlated).

Units: i-PI writes positions in angstrom and momenta in atomic units; momenta
are converted to ASE's sqrt(eV u) with P_AU_TO_ASE (verified to 2e-7 in
checks/check_water_run.py). Positions are left unfolded, as i-PI writes them.

Targets are stored mass-scaled (paper Eq. 6) with the masses from the config:
  future_positions := positions + (future - now) * sqrt(m)
  future_momenta   := future_momenta / sqrt(m)
The input momenta column stays raw. eval/eval_onestep.py --scaled_targets undoes this.

A run is used only if its position and momentum files both hold exactly
--nve_steps + 1 frames (a complete run). Incomplete runs are listed and skipped.

Usage:
  python data/water/parse_runs.py /storage3/lauren/data/water configs/flashmd_water16_scratch.yaml \
      --strides 1 4 16 --out_root data/water/pairs
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys

import ase.io
import numpy as np
import yaml
from ase import Atoms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from checks.check_water_run_io import P_AU_TO_ASE, read_ipi_xyz  # noqa: E402

DT_FS = 0.25   # production time step of run_one.sh


def load_masses(config):
    with open(config) as f:
        cfg = yaml.safe_load(f)
    return {int(k): float(v) for k, v in cfg["architecture"]["training"]["masses"].items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def count_frames(path):
    """Number of frames in an i-PI xyz file, from the atom count on its first line."""
    with open(path) as f:
        n = int(f.readline())
    with open(path, "rb") as f:
        lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 24), b""))
    return lines // (n + 2)


def load_run(run_dir, n_expected):
    """Return (q[nframes, natoms, 3] in A, p[nframes, natoms, 3] in ASE units, symbols, cell lengths) or None."""
    pos_path = os.path.join(run_dir, "nve.pos_0.xyz")
    mom_path = os.path.join(run_dir, "nve.mom_0.xyz")
    if not (os.path.exists(pos_path) and os.path.exists(mom_path)):
        return None
    q, symbols, cells = read_ipi_xyz(pos_path)
    p, _, _ = read_ipi_xyz(mom_path)
    if len(q) != n_expected or len(p) != n_expected:
        return None
    assert np.allclose(cells, cells[0]), "cell changed during a constant-volume run"
    return q, p * P_AU_TO_ASE, symbols, cells[0]


def make_pair(q0, p0, q1, p1, symbols, cell, masses):
    """One training frame: inputs (q0, p0), mass-scaled targets built from (q1, p1)."""
    atoms = Atoms(symbols=symbols, positions=q0, cell=cell, pbc=True)
    atoms.set_momenta(p0)
    sm = np.sqrt([masses[int(z)] for z in atoms.numbers])[:, None]
    atoms.arrays["future_positions"] = q0 + (q1 - q0) * sm
    atoms.arrays["future_momenta"] = p1 / sm
    return atoms


def pairs_from_run(q, p, symbols, cell, masses, stride_steps, spacing_steps):
    """Forward pairs every spacing_steps, each followed by its time-reversed copy."""
    out = []
    for t0 in range(0, len(q) - stride_steps, spacing_steps):
        t1 = t0 + stride_steps
        out.append(make_pair(q[t0], p[t0], q[t1], p[t1], symbols, cell, masses))
        out.append(make_pair(q[t1], -p[t1], q[t0], -p[t0], symbols, cell, masses))
    return out


def split_runs(run_ids, seed, val_frac=0.1, test_frac=0.1):
    ids = sorted(run_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    n_test = int(round(len(ids) * test_frac))
    split = {r: "val" for r in ids[:n_val]}
    split.update({r: "test" for r in ids[n_val:n_val + n_test]})
    split.update({r: "train" for r in ids[n_val + n_test:]})
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root", help="directory holding run_001, run_002, ...")
    ap.add_argument("config", help="metatrain yaml; masses are read from it")
    ap.add_argument("--strides", type=float, nargs="+", default=[1.0, 4.0, 16.0], help="fs")
    ap.add_argument("--spacing", type=float, default=100.0, help="fs between successive pairs from one run")
    ap.add_argument("--nve_steps", type=int, default=8000)
    ap.add_argument("--out_root", default="data/water/pairs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    masses = load_masses(args.config)
    run_dirs = sorted(glob.glob(os.path.join(args.runs_root, "run_*")))
    n_expected = args.nve_steps + 1
    complete, skipped = [], []
    for d in run_dirs:   # cheap completeness test: frame count from file size of the position file
        pos, mom = os.path.join(d, "nve.pos_0.xyz"), os.path.join(d, "nve.mom_0.xyz")
        if os.path.exists(pos) and os.path.exists(mom) and count_frames(pos) == n_expected and count_frames(mom) == n_expected:
            complete.append(os.path.basename(d))
        else:
            skipped.append(os.path.basename(d))
    print(f"{len(complete)} complete runs; {len(skipped)} incomplete or missing, skipped: {skipped[:10]}"
          + (" ..." if len(skipped) > 10 else ""))
    if not complete:
        raise SystemExit("no complete runs")
    split = split_runs(complete, args.seed)
    print("split by run:", {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")})

    spacing_steps = int(round(args.spacing / DT_FS))
    strides = []
    for stride in args.strides:
        stride_steps = int(round(stride / DT_FS))
        assert abs(stride_steps * DT_FS - stride) < 1e-9, f"stride {stride} fs is not a multiple of {DT_FS} fs"
        out_dir = os.path.join(args.out_root, f"{stride:g}fs")
        os.makedirs(out_dir, exist_ok=True)
        writers = {s: open(os.path.join(out_dir, f"{s}.xyz"), "w") for s in ("train", "val", "test")}
        strides.append({"stride": stride, "steps": stride_steps, "dir": out_dir, "writers": writers,
                        "counts": {s: 0 for s in writers}, "max_disp": 0.0})
    half_box = np.inf
    for k, name in enumerate(complete):   # runs are read one at a time (each is ~75 MB per file)
        q, p, symbols, cell = load_run(os.path.join(args.runs_root, name), n_expected)
        half_box = min(half_box, 0.5 * float(min(cell)))
        for st in strides:
            pairs = pairs_from_run(q, p, symbols, cell, masses, st["steps"], spacing_steps)
            for a in pairs:
                sm = np.sqrt([masses[int(z)] for z in a.numbers])[:, None]
                raw = (a.arrays["future_positions"] - a.get_positions()) / sm
                st["max_disp"] = max(st["max_disp"], float(np.linalg.norm(raw, axis=1).max()))
            ase.io.write(st["writers"][split[name]], pairs, format="extxyz")
            st["counts"][split[name]] += len(pairs)
        if (k + 1) % 20 == 0:
            print(f"  {k + 1} runs parsed", flush=True)
    for st in strides:
        for w in st["writers"].values():
            w.close()
        assert st["max_disp"] < half_box, "a displacement exceeds half the box: positions are not unfolded as assumed"
        manifest = {
            "stride_fs": st["stride"], "spacing_fs": args.spacing, "dt_fs": DT_FS, "nve_steps": args.nve_steps,
            "time_reversed_copies": True, "targets": "mass-scaled (Eq. 6) with config masses; input momenta raw",
            "momentum_unit_conversion_au_to_ase": P_AU_TO_ASE, "masses": masses,
            "config": {"path": args.config, "sha256": sha256(args.config)},
            "split": {"by": "run", "seed": args.seed, "runs": {s: sorted(r for r, v in split.items() if v == s) for s in st["counts"]}},
            "skipped_runs": skipped, "pairs": st["counts"],
            "pairs_per_run": 2 * len(range(0, n_expected - st["steps"], spacing_steps)),
            "max_raw_displacement_A": st["max_disp"],
            "sha256": {f"{s}.xyz": sha256(os.path.join(st["dir"], f"{s}.xyz")) for s in st["counts"]},
        }
        with open(os.path.join(st["dir"], "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"stride {st['stride']:g} fs: {st['counts']} pairs ({manifest['pairs_per_run']} per run), "
              f"largest raw displacement {st['max_disp']:.3f} A -> {st['dir']}")


if __name__ == "__main__":
    main()
