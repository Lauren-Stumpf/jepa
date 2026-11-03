"""
Prepare the organic-molecule training set from the Zenodo 8 fs pairs file.

One script, one pass. Every rule is printed together with what it excludes,
and everything that is measured but not used to drop frames is recorded in
frames.csv. The result is three xyz files that both the from-scratch FlashMD
baseline and our model read unchanged, plus a manifest with their hashes.

Structure rules (ours, not the paper's):
  1. elements within {H, C, N, O} and containing carbon (definition of organic)
  2. periodic cell with more than --min_vol_per_atom A^3 per atom (an isolated
     molecule in a large box). The sorted volume-per-atom values of all
     organic structures are printed so the threshold can be seen to sit in a
     gap of the distribution. Non-periodic entries are excluded and counted.
  3. at least --min_atoms atoms (default 0, i.e. off; if used, print what it removes)
  4. one copy per molecule: groups with the same (formula, atom count, cell
     volume to 0.1 A^3) are the same molecule stored twice; the lowest group
     id is kept, the rest excluded and recorded.

Frame rules: none, except exploded reference runs (kinetic temperature above
--max_T or raw per-atom displacement over the stride above --max_disp). The
distributions of both quantities over the selected frames are printed before
the thresholds are applied, so the thresholds can be checked to sit in an
empty region. Nothing else is dropped: hot frames where the molecule has come
apart, and structures that see their own periodic image at the model cutoff,
are kept because the authors trained on them and the comparison is baseline
versus our model on identical frames.

Not done: no unwrapping, recentring or nearest-image placement. metatrain's
input is a neighbour list with cell shifts, which gives identical edge vectors
whether or not atoms sit inside the box; its position target is the stored
future minus now with no wrapping; i-PI writes unfolded positions. The script
stops if any kept frame has a raw displacement larger than half the shortest
cell height, which would be evidence that the file was wrapped afterwards.

Recorded per frame (diagnostic only): n_pieces (connected components of the
bond graph using 1.25 x covalent radii, with periodic images), n_image_pairs
(directed neighbour pairs within the model cutoff whose cell shift is nonzero,
i.e. the model sees the molecule's own image), max_disp_raw_A, kinetic_T_K.

Targets are stored mass-scaled (paper Eq. 6) with the masses from the config:
  future_positions := positions + (future - now) * sqrt(m)
  future_momenta   := future_momenta / sqrt(m)
The input momenta column stays raw. eval/eval_onestep.py --scaled_targets undoes this.

Usage:
  python data/prepare_organic.py data/universal_8fs.xyz results/data/catalogue_8fs.csv \
      configs/flashmd_organic8_scratch.yaml --out_dir data/organic8 --seed 0
"""

import argparse
import csv
import hashlib
import json
import os
import random

import ase.io
import ase.units
import numpy as np
import pandas as pd
import yaml
from ase.neighborlist import natural_cutoffs, neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ORGANIC_ELEMENTS = {"H", "C", "N", "O"}
TARGET_ARRAYS = ("numbers", "positions", "momenta", "future_positions", "future_momenta")
# two atoms are bonded if closer than BOND_MULT * (sum of their covalent radii); 1.25 keeps
# a bond stretched by thermal motion at the file's temperatures (only used for the n_pieces diagnostic)
BOND_MULT = 1.25


def load_config(path):
    """Return (model cutoff in A, {atomic number: mass in u}) from a metatrain config."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    arch = cfg["architecture"]
    cutoff = float(arch["model"]["cutoff"])
    masses = {int(k): float(v) for k, v in arch["training"]["masses"].items()}
    return cutoff, masses


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def print_distribution(name, values, unit, edges):
    """Text histogram so a threshold can be checked against the data it cuts."""
    values = np.asarray(values, dtype=float)
    counts, _ = np.histogram(values, bins=edges)
    print(f"  {name}: n={len(values)}, min={values.min():.3g}, max={values.max():.3g} {unit}")
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        if c:
            print(f"    [{lo:>8.3g}, {hi:>8.3g}) {unit}: {c}")
    above = int((values >= edges[-1]).sum())
    if above:
        print(f"    >= {edges[-1]:.3g} {unit}: {above}")


def select_structures(cat, min_vol_per_atom, min_atoms):
    """Per-structure table with a keep flag and an exclusion reason for every group."""
    per = cat.groupby("group").agg(
        formula=("formula", "first"), natoms=("natoms", "first"), elements=("elements", "first"),
        periodic=("periodic", "first"), volume_A3=("volume_A3", "first"),
        n_frames=("frame", "size"), frame_first=("frame", "min"), frame_last=("frame", "max"),
    ).reset_index()
    bad = per[per["n_frames"] != 500]
    if len(bad):
        print(f"WARNING: {len(bad)} groups do not have exactly 500 frames (listed in manifest)")
    per["vol_per_atom"] = per["volume_A3"] / per["natoms"]
    per["reason"] = ""

    el_ok = per["elements"].apply(lambda s: set(s.split("-")) <= ORGANIC_ELEMENTS and "C" in s.split("-"))
    per.loc[~el_ok, "reason"] = "not organic (elements outside H,C,N,O or no carbon)"
    organic = per[el_ok]
    print(f"rule 1 (elements within H,C,N,O and contains C): {len(organic)} of {len(per)} structures kept")

    nonper = organic[organic["periodic"] == 0]
    per.loc[nonper.index, "reason"] = "non-periodic entry"
    print(f"rule 2a (periodic cell): {len(nonper)} non-periodic organic structures excluded"
          + (": " + ", ".join(sorted(set(nonper["formula"]))[:20]) if len(nonper) else ""))
    periodic = organic[organic["periodic"] == 1]
    vpa = np.sort(periodic["vol_per_atom"].values)
    print(f"rule 2b (volume per atom > {min_vol_per_atom} A^3): distribution over the "
          f"{len(periodic)} periodic organic structures, so the threshold can be seen against a gap:")
    print_distribution("volume per atom", vpa, "A^3", np.array([0, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200, 400, 1e9]))
    below = vpa[vpa <= min_vol_per_atom]
    above = vpa[vpa > min_vol_per_atom]
    if len(below) and len(above):
        print(f"  nearest values either side of the threshold: {below.max():.1f} and {above.min():.1f} A^3/atom")
    dense = periodic[periodic["vol_per_atom"] <= min_vol_per_atom]
    per.loc[dense.index, "reason"] = f"volume per atom <= {min_vol_per_atom} A^3 (condensed phase)"
    vac = periodic[periodic["vol_per_atom"] > min_vol_per_atom]
    print(f"  kept {len(vac)}, excluded {len(dense)}")

    small = vac[vac["natoms"] < min_atoms]
    per.loc[small.index, "reason"] = f"fewer than {min_atoms} atoms"
    vac = vac[vac["natoms"] >= min_atoms]
    print(f"rule 3 (at least {min_atoms} atoms): excluded {len(small)}"
          + (": " + ", ".join(sorted(set(small["formula"]))) if len(small) else ""))

    key = list(zip(vac["formula"], vac["natoms"].astype(int), vac["volume_A3"].round(1)))
    first_by_key = {}
    dup_idx = []
    for idx, k in zip(vac.index, key):
        if k in first_by_key:
            dup_idx.append(idx)
            per.loc[idx, "reason"] = f"duplicate copy of group {int(per.loc[first_by_key[k], 'group'])}"
        else:
            first_by_key[k] = idx
    kept = vac.drop(index=dup_idx)
    print(f"rule 4 (one copy per molecule): {len(dup_idx)} duplicate copies excluded"
          + (": " + ", ".join(f"{per.loc[i, 'formula']} (group {int(per.loc[i, 'group'])})" for i in dup_idx) if dup_idx else ""))
    per["keep"] = per.index.isin(kept.index)
    print(f"selected {int(per['keep'].sum())} structures, {int(per.loc[per['keep'], 'n_frames'].sum())} frames")
    return per


def split_structures(groups, seed, val_frac, test_frac):
    groups = sorted(int(g) for g in groups)
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = int(round(len(groups) * val_frac))
    n_test = int(round(len(groups) * test_frac))
    split = {}
    for g in groups[:n_val]:
        split[g] = "val"
    for g in groups[n_val:n_val + n_test]:
        split[g] = "test"
    for g in groups[n_val + n_test:]:
        split[g] = "train"
    return split


def measure_frame(atoms, cutoff):
    """(n_pieces, n_image_pairs, max_disp_raw_A, kinetic_T_K) for one frame."""
    n = len(atoms)
    i, j = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=BOND_MULT))
    adj = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    n_pieces = int(connected_components(adj, directed=False)[0])
    if atoms.pbc.any():
        _, _, S = neighbor_list("ijS", atoms, cutoff)
        n_image_pairs = int((np.abs(S).sum(1) > 0).sum())
    else:
        n_image_pairs = 0
    disp = atoms.arrays["future_positions"] - atoms.get_positions()
    max_disp = float(np.linalg.norm(disp, axis=1).max())
    kin_T = 2.0 * atoms.get_kinetic_energy() / (3.0 * n * ase.units.kB)
    return n_pieces, n_image_pairs, max_disp, float(kin_T)


def scale_targets(atoms, masses):
    """Replace the raw targets by the mass-scaled ones (paper Eq. 6). Returns the
    largest relative difference between the config masses and the file's masses."""
    m_cfg = np.array([masses[int(z)] for z in atoms.numbers])
    m_file = atoms.get_masses()
    rel = float(np.abs(m_cfg - m_file).max() / m_file.max())
    sm = np.sqrt(m_cfg)[:, None]
    q = atoms.get_positions()
    atoms.arrays["future_positions"] = q + (atoms.arrays["future_positions"] - q) * sm
    atoms.arrays["future_momenta"] = atoms.arrays["future_momenta"] / sm
    return rel


def strip(atoms, keep_masses=False):
    """Drop everything training does not read (stored forces, energies, stress, ...)."""
    keep = TARGET_ARRAYS + (("masses",) if keep_masses else ())
    for k in list(atoms.info):
        if k not in ("Lattice", "pbc"):
            del atoms.info[k]
    for k in list(atoms.arrays):
        if k not in keep:
            del atoms.arrays[k]
    atoms.calc = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("catalogue")
    ap.add_argument("config", help="metatrain yaml; the model cutoff and masses are read from it")
    ap.add_argument("--out_dir", default="data/organic8")
    ap.add_argument("--min_vol_per_atom", type=float, default=60.0)
    ap.add_argument("--min_atoms", type=int, default=0)
    ap.add_argument("--max_T", type=float, default=5000.0, help="exploded-run rule: kinetic temperature (K)")
    ap.add_argument("--max_disp", type=float, default=3.0, help="exploded-run rule: raw per-atom displacement over the stride (A)")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cutoff, masses = load_config(args.config)
    print(f"config {args.config}: model cutoff {cutoff} A, masses {masses}")
    cat = pd.read_csv(args.catalogue)
    per = select_structures(cat, args.min_vol_per_atom, args.min_atoms)
    kept = per[per["keep"]]
    split = split_structures(kept["group"], args.seed, args.val_frac, args.test_frac)
    print("split by structure:", {s: sum(v == s for v in split.values()) for s in ("train", "val", "test")})
    wanted = cat[cat["group"].isin(list(split))]
    frame_group = dict(zip(wanted["frame"].astype(int), wanted["group"].astype(int)))
    last_frame = max(frame_group)

    # pass 1: read the selected frames, measure them, keep them in memory
    print(f"reading {len(frame_group)} selected frames (scanning the file up to frame {last_frame})", flush=True)
    frames = []
    rows = []
    for i, atoms in enumerate(ase.io.iread(args.xyz, index=":", format="extxyz")):
        if i > last_frame:
            break
        g = frame_group.get(i)
        if g is None:
            continue
        n_pieces, n_img, max_disp, kin_T = measure_frame(atoms, cutoff)
        rows.append({"frame": i, "group": g, "formula": atoms.get_chemical_formula(), "split": split[g],
                     "kinetic_T_K": kin_T, "n_pieces": n_pieces, "n_image_pairs": n_img,
                     "max_disp_raw_A": max_disp})
        strip(atoms, keep_masses=True)   # keep memory small; masses still needed for the check below
        frames.append(atoms)
        if len(frames) % 20000 == 0:
            print(f"  {len(frames)} frames read", flush=True)
    fr = pd.DataFrame(rows)

    print("frame diagnostics over the selected frames (recorded, not used to drop):")
    print_distribution("kinetic temperature", fr["kinetic_T_K"], "K",
                       np.array([0, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 1e4, 1e5, 1e9]))
    print_distribution("raw max displacement over 8 fs", fr["max_disp_raw_A"], "A",
                       np.array([0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9]))
    print(f"  frames with more than one piece: {int((fr['n_pieces'] > 1).sum())}; "
          f"by molecule (top 10): "
          f"{fr[fr['n_pieces'] > 1].groupby('formula').size().sort_values(ascending=False).head(10).to_dict()}")
    print(f"  frames where the model sees the molecule's own image (n_image_pairs > 0): "
          f"{int((fr['n_image_pairs'] > 0).sum())}; by molecule: "
          f"{fr[fr['n_image_pairs'] > 0].groupby('formula').size().sort_values(ascending=False).head(10).to_dict()}")

    exploded = (fr["kinetic_T_K"] > args.max_T) | (fr["max_disp_raw_A"] > args.max_disp)
    fr["exploded"] = exploded.astype(int)
    print(f"exploded-run rule (T > {args.max_T} K or max displacement > {args.max_disp} A): "
          f"{int(exploded.sum())} frames excluded"
          + (f"; by molecule: {fr[exploded].groupby('formula').size().to_dict()}" if exploded.any() else ""))

    # wrapping check: on an unwrapped trajectory no kept displacement can approach half the box
    half_box = np.array([0.5 * a.cell.lengths().min() if a.pbc.any() else np.inf for a in frames])
    suspicious = (~exploded.values) & (fr["max_disp_raw_A"].values > half_box)
    if suspicious.any():
        raise SystemExit(f"STOP: {int(suspicious.sum())} kept frames have a raw displacement larger than half "
                         f"the shortest cell height; the file may have been wrapped after the fact. Frames: "
                         f"{fr.loc[suspicious, 'frame'].tolist()[:20]}")

    # pass 2: scale targets, strip, write
    writers = {s: open(os.path.join(args.out_dir, f"{s}.xyz"), "w") for s in ("train", "val", "test")}
    counts = {s: 0 for s in writers}
    mass_rel = 0.0
    fr["written"] = 0
    for k, (atoms, ex) in enumerate(zip(frames, exploded.values)):
        if ex:
            continue
        mass_rel = max(mass_rel, scale_targets(atoms, masses))
        strip(atoms)   # drops the masses column; the config masses are the ones used
        s = fr.at[k, "split"]
        ase.io.write(writers[s], atoms, format="extxyz")
        counts[s] += 1
        fr.at[k, "written"] = 1
    for w in writers.values():
        w.close()
    print(f"largest relative difference between config masses and the file's masses: {mass_rel:.2e}"
          + (" (WARNING: above 1%)" if mass_rel > 0.01 else ""))
    print("written", counts, "->", args.out_dir)

    frames_csv = os.path.join(args.out_dir, "frames.csv")
    fr.to_csv(frames_csv, index=False)
    excluded = per[~per["keep"]]
    manifest = {
        "source": {"path": args.xyz, "size_bytes": os.path.getsize(args.xyz)},
        "catalogue": {"path": args.catalogue, "sha256": sha256(args.catalogue)},
        "config": {"path": args.config, "sha256": sha256(args.config), "cutoff_A": cutoff, "masses": masses},
        "rules": {
            "elements": sorted(ORGANIC_ELEMENTS), "requires_carbon": True,
            "min_vol_per_atom_A3": args.min_vol_per_atom, "min_atoms": args.min_atoms,
            "one_copy_per_molecule": "same (formula, natoms, cell volume to 0.1 A^3): lowest group id kept",
            "exploded_frame": {"max_T_K": args.max_T, "max_disp_A": args.max_disp},
            "split": {"by": "structure (group id)", "seed": args.seed, "val_frac": args.val_frac, "test_frac": args.test_frac},
            "targets": "mass-scaled (Eq. 6) with config masses; input momenta raw",
            "unwrapping": "none (metatrain uses cell-shifted neighbour lists and raw displacements)",
        },
        "groups_not_500_frames": per[per["n_frames"] != 500][["group", "formula", "n_frames"]].to_dict("records"),
        "excluded_structures": excluded[["group", "formula", "natoms", "frame_first", "frame_last", "reason"]]
            .assign(group=lambda d: d["group"].astype(int)).to_dict("records"),
        "selected_structures": {s: sorted(g for g, v in split.items() if v == s) for s in counts},
        "formulas": {s: sorted(set(kept[kept["group"].isin([g for g, v in split.items() if v == s])]["formula"])) for s in counts},
        "frames": {"selected": int(len(fr)), "exploded_excluded": int(exploded.sum()), "written": counts,
                   "multi_piece": int((fr["n_pieces"] > 1).sum()), "own_image_visible": int((fr["n_image_pairs"] > 0).sum())},
        "mass_relative_difference_config_vs_file": mass_rel,
        "sha256": {f"{s}.xyz": sha256(os.path.join(args.out_dir, f"{s}.xyz")) for s in counts},
    }
    manifest["sha256"]["frames.csv"] = sha256(frames_csv)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest and frames.csv written")


if __name__ == "__main__":
    main()
