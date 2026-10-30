"""
Catalogue every frame of a FlashMD Zenodo pairs file in one pass.

Writes one CSV row per frame with:
  frame        index in the file
  group        structure group id (increments when formula or atom count changes
               between consecutive frames; frames of one structure are contiguous)
  formula      chemical formula
  natoms
  periodic     1 if any periodic direction, else 0
  volume_A3    cell volume if periodic, else empty
  kinetic_T_K  temperature from stored momenta
  pe_eV_atom   stored potential energy per atom
  disp_ms_rms  rms of stored mass-scaled displacement over the stride
  p_ms_rms     rms of stored mass-scaled momentum
  broken       1 if kinetic_T_K > 5000 or disp_ms_rms > 5, else 0
  elements     sorted element symbols joined by '-'
  extra_info   any header fields other than the known ones (possible subset label)

Everything downstream (counts, distributions, choosing a subset) is then quick
pandas work on this table instead of another pass over the 39 GB file.

Usage:
  python data/catalogue.py data/universal_8fs.xyz --out results/data/catalogue_8fs.csv
"""

import argparse
import csv
import os

import ase.io
import ase.units
import numpy as np

KNOWN_INFO = {"Lattice", "Properties", "pbc", "potential_energy", "kinetic_energy",
              "total_energy", "energy", "stress", "virial", "cell"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--out", default="results/data/catalogue_8fs.csv")
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fields = ["frame", "group", "formula", "natoms", "periodic", "volume_A3", "kinetic_T_K",
              "pe_eV_atom", "disp_ms_rms", "p_ms_rms", "broken", "elements", "extra_info"]
    group = -1
    prev_key = None
    n_broken = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, atoms in enumerate(ase.io.iread(args.xyz, index=":", format="extxyz")):
            if args.max_frames and i >= args.max_frames:
                break
            n = len(atoms)
            formula = atoms.get_chemical_formula()
            key = (formula, n)
            if key != prev_key:
                group += 1
                prev_key = key
            periodic = int(bool(atoms.pbc.any()))
            volume = f"{atoms.get_volume():.2f}" if periodic else ""
            m = atoms.get_masses()[:, None]
            kin_T = 2.0 * atoms.get_kinetic_energy() / (3.0 * n * ase.units.kB)
            disp = (atoms.arrays["future_positions"] - atoms.get_positions()) * np.sqrt(m)
            disp_rms = float(np.sqrt((disp ** 2).mean()))
            p_rms = float(np.sqrt(((atoms.get_momenta() / np.sqrt(m)) ** 2).mean()))
            pe = atoms.info.get("potential_energy")
            if pe is None and atoms.calc is not None:
                pe = atoms.calc.results.get("energy")
            pe_atom = f"{pe / n:.5f}" if pe is not None else ""
            broken = int(kin_T > 5000.0 or disp_rms > 5.0)
            n_broken += broken
            extra = ";".join(f"{k}={atoms.info[k]}" for k in sorted(atoms.info) if k not in KNOWN_INFO)
            elements = "-".join(sorted(set(atoms.get_chemical_symbols())))
            w.writerow([i, group, formula, n, periodic, volume, f"{kin_T:.1f}", pe_atom,
                        f"{disp_rms:.4f}", f"{p_rms:.4f}", broken, elements, extra])
            if (i + 1) % 100000 == 0:
                print(f"{i + 1} frames, {group + 1} groups, {n_broken} broken so far", flush=True)
    print(f"done: {i + 1} frames, {group + 1} groups, {n_broken} broken; written {args.out}")


if __name__ == "__main__":
    main()
