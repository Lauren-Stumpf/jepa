"""
Unit tests for data/prepare_organic.py.

  python checks/test_prepare.py

Covers: piece counting (two molecules; one molecule straddling the box edge;
a non-periodic molecule), own-image detection at the model cutoff (zero in a
large box, nonzero when the box shrinks), config reading, mass-scaling round
trip, structure selection on a synthetic catalogue (every rule, duplicate
kept once), split determinism, and an end-to-end run on a small synthetic
pairs file whose manifest hashes are reproducible.
"""

import json
import os
import subprocess
import sys
import tempfile

import ase.io
import numpy as np
import pandas as pd
from ase import Atoms
from ase.build import molecule

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.prepare_organic import (load_config, measure_frame, scale_targets,  # noqa: E402
                                  select_structures, split_structures)

CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "flashmd_organic8_scratch.yaml")


def with_targets(atoms, disp=0.05, seed=0):
    rng = np.random.default_rng(seed)
    atoms.arrays["future_positions"] = atoms.get_positions() + rng.normal(0, disp, (len(atoms), 3))
    atoms.set_momenta(rng.normal(0, 0.1, (len(atoms), 3)))
    atoms.arrays["future_momenta"] = rng.normal(0, 0.5, (len(atoms), 3))
    return atoms


def test_pieces_two_molecules():
    a = molecule("CH4")
    b = molecule("CH4")
    b.translate([6.0, 0, 0])
    both = a + b
    both.set_cell([20, 20, 20]); both.pbc = True; both.center()
    with_targets(both)
    n_pieces, _, _, _ = measure_frame(both, 4.5)
    assert n_pieces == 2, n_pieces


def test_pieces_straddling_box():
    a = molecule("C2H6")
    a.set_cell([12, 12, 12]); a.pbc = True; a.center()
    a.translate([5.5, 0, 0])            # carbon-carbon bond now crosses the x face
    a.wrap()
    with_targets(a)
    n_pieces, _, _, _ = measure_frame(a, 4.5)
    assert n_pieces == 1, n_pieces      # periodic images join the two halves


def test_pieces_nonperiodic():
    a = molecule("C2H6"); a.pbc = False
    with_targets(a)
    n_pieces, n_img, _, _ = measure_frame(a, 4.5)
    assert n_pieces == 1 and n_img == 0


def test_image_pairs():
    a = molecule("C6H6")
    a.set_cell([20, 20, 20]); a.pbc = True; a.center()
    with_targets(a)
    _, n_img, _, _ = measure_frame(a, 4.5)
    assert n_img == 0, n_img
    extent = np.ptp(a.get_positions(), axis=0).max()
    a.set_cell([extent + 3.0] * 3); a.center()   # box smaller than extent + cutoff: image within reach
    _, n_img, _, _ = measure_frame(a, 4.5)
    assert n_img > 0, n_img


def test_disp_and_T():
    a = molecule("H2O"); a.set_cell([10, 10, 10]); a.pbc = True; a.center()
    a.arrays["future_positions"] = a.get_positions() + np.array([[0.3, 0, 0], [0, 0, 0], [0, 0.4, 0]])
    a.set_momenta(np.zeros((3, 3)))
    _, _, max_disp, kin_T = measure_frame(a, 4.5)
    assert abs(max_disp - 0.4) < 1e-12 and kin_T == 0.0


def test_config():
    cutoff, masses = load_config(CONFIG)
    assert cutoff == 4.5 and masses[1] == 1.00794 and masses[8] == 15.9994


def test_scale_round_trip():
    _, masses = load_config(CONFIG)
    a = with_targets(molecule("CH3OH"))
    q = a.get_positions().copy()
    fut = a.arrays["future_positions"].copy()
    pf = a.arrays["future_momenta"].copy()
    rel = scale_targets(a, masses)
    assert rel < 0.01, rel
    sm = np.sqrt([masses[z] for z in a.numbers])[:, None]
    assert np.allclose(q + (a.arrays["future_positions"] - q) / sm, fut)
    assert np.allclose(a.arrays["future_momenta"] * sm, pf)


def synthetic_catalogue():
    rows = []
    specs = [  # group, formula, natoms, elements, periodic, volume
        (0, "C2H6", 8, "C-H", 1, 8000.0),       # keep
        (1, "H2O", 3, "H-O", 1, 8000.0),        # no carbon
        (2, "C2H6", 8, "C-H", 1, 8000.0),       # duplicate of group 0
        (3, "C6H6", 12, "C-H", 1, 100.0),       # dense (8.3 A^3/atom)
        (4, "CH4", 5, "C-H", 0, np.nan),        # non-periodic
        (5, "C2H5Cl", 8, "C-Cl-H", 1, 8000.0),  # chlorine
        (6, "CH3OH", 6, "C-H-O", 1, 7000.0),    # keep
        (7, "C2H6", 8, "C-H", 1, 9000.0),       # same formula, different box: a different entry, keep
    ]
    for g, f, n, el, per, vol in specs:
        for k in range(500):
            rows.append({"frame": g * 500 + k, "group": g, "formula": f, "natoms": n, "periodic": per,
                         "volume_A3": vol, "elements": el})
    return pd.DataFrame(rows)


def test_select():
    per = select_structures(synthetic_catalogue(), 60.0, 0)
    kept = sorted(per.loc[per["keep"], "group"].astype(int))
    assert kept == [0, 6, 7], kept
    reasons = dict(zip(per["group"].astype(int), per["reason"]))
    assert reasons[2] == "duplicate copy of group 0"
    assert "non-periodic" in reasons[4] and "not organic" in reasons[5] and "no carbon" in reasons[1]
    per = select_structures(synthetic_catalogue(), 60.0, 7)
    assert sorted(per.loc[per["keep"], "group"].astype(int)) == [0, 7]   # CH3OH has 6 atoms


def test_split_deterministic():
    s1 = split_structures(range(100), 0, 0.1, 0.1)
    s2 = split_structures(range(100), 0, 0.1, 0.1)
    assert s1 == s2
    counts = {k: sum(v == k for v in s1.values()) for k in ("train", "val", "test")}
    assert counts == {"train": 80, "val": 10, "test": 10}, counts


def test_end_to_end():
    """Fake pairs file: 4 structures x 500 frames; one structure is a duplicate, one hot frame."""
    _, masses = load_config(CONFIG)
    with tempfile.TemporaryDirectory() as d:
        xyz = os.path.join(d, "pairs.xyz")
        rows = []
        mols = [("C2H6", 0), ("CH3OH", 1), ("C2H6", 2), ("C2H4", 3)]
        frames = []
        for name, g in mols:
            base = molecule(name); base.set_cell([20, 20, 20]); base.pbc = True; base.center()
            for k in range(500):
                a = base.copy()
                with_targets(a, seed=g * 1000 + k)
                a.arrays["forces"] = np.zeros((len(a), 3))
                a.info["potential_energy"] = -1.0
                if g == 3 and k == 7:
                    a.set_momenta(a.get_momenta() * 200.0)   # exploded frame
                frames.append(a)
                kin_T = 2.0 * a.get_kinetic_energy() / (3.0 * len(a) * 8.617333262e-5)
                rows.append({"frame": len(frames) - 1, "group": g, "formula": a.get_chemical_formula(),
                             "natoms": len(a), "periodic": 1, "volume_A3": 8000.0,
                             "kinetic_T_K": kin_T, "elements": "-".join(sorted(set(a.get_chemical_symbols())))})
        ase.io.write(xyz, frames, format="extxyz")
        cat = os.path.join(d, "cat.csv")
        pd.DataFrame(rows).to_csv(cat, index=False)
        out1, out2 = os.path.join(d, "o1"), os.path.join(d, "o2")
        script = os.path.join(os.path.dirname(__file__), "..", "data", "prepare_organic.py")
        for out in (out1, out2):
            r = subprocess.run([sys.executable, script, xyz, cat, CONFIG, "--out_dir", out, "--val_frac", "0.34",
                                "--test_frac", "0.33"], capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
        m1 = json.load(open(os.path.join(out1, "manifest.json")))
        m2 = json.load(open(os.path.join(out2, "manifest.json")))
        assert m1["sha256"] == m2["sha256"], "not reproducible"
        assert [e["group"] for e in m1["excluded_structures"]] == [2]
        assert m1["frames"]["selected"] == 1500 and m1["frames"]["exploded_excluded"] == 1
        assert sum(m1["frames"]["written"].values()) == 1499
        fr = pd.read_csv(os.path.join(out1, "frames.csv"))
        assert fr["n_pieces"].max() == 1 and fr["n_image_pairs"].max() == 0
        # written file holds mass-scaled targets and nothing else
        a = next(ase.io.iread(os.path.join(out1, "train.xyz"), index="0", format="extxyz"))
        assert set(a.arrays) == {"numbers", "positions", "momenta", "future_positions", "future_momenta"}, set(a.arrays)
        assert "potential_energy" not in a.info
        # the first written frame is the first frame of whichever structure was first in the file, in the same split
        first_written = fr[(fr["written"] == 1) & (fr["split"] == fr[fr["written"] == 1].iloc[0]["split"])].iloc[0]
        src = frames[int(first_written["frame"])]
        q = src.get_positions()
        sm = np.sqrt([masses[z] for z in src.numbers])[:, None]
        a_split = next(ase.io.iread(os.path.join(out1, f"{first_written['split']}.xyz"), index="0", format="extxyz"))
        assert np.allclose(a_split.arrays["future_positions"], q + (src.arrays["future_positions"] - q) * sm, atol=1e-6)
        assert np.allclose(a_split.arrays["future_momenta"], src.arrays["future_momenta"] / sm, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} tests passed")
