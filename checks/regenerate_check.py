"""
Regeneration check for FlashMD Zenodo pairs.

For each structure in the input file, take the stored positions and momenta
("before"), run velocity Verlet with PET-MAD at 0.25 fs for stride/0.25 steps,
and compare the result with the stored future_positions / future_momenta
("after"). If our potential, integrator, time step and units match the
authors', the differences should be tiny.

Also records total energy before and after the regenerated steps, as a first
energy-conservation check.

Usage:
  python regenerate_check.py flashmd8_test.xyz --stride_fs 8 --n 100
"""

import argparse
import csv

import ase.io
import ase.units
import numpy as np
import torch
from ase.md.verlet import VelocityVerlet
from pet_mad.calculator import PETMADCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz")
    parser.add_argument("--stride_fs", type=float, default=8.0)
    parser.add_argument("--dt_fs", type=float, default=0.25)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--pet_mad_version", default="1.0.0",
                        help="PET-MAD tag; valid ones are 1.0.0 and 1.0.2 (the paper says 1.0)")
    parser.add_argument("--out", default="regenerate_check.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = PETMADCalculator(version=args.pet_mad_version, device=device)
    n_steps = int(round(args.stride_fs / args.dt_fs))
    print(f"device {device}, PET-MAD {args.pet_mad_version}, {n_steps} steps of {args.dt_fs} fs")

    rows = []
    for i, atoms in enumerate(ase.io.iread(args.xyz, index=":", format="extxyz")):
        if i >= args.n:
            break
        ref_q = atoms.arrays["future_positions"].copy()
        ref_p = atoms.arrays["future_momenta"].copy()

        a = atoms.copy()
        a.calc = calc
        pe0 = a.get_potential_energy()
        e0 = pe0 + a.get_kinetic_energy()
        # single-point check: our PET-MAD against the forces/energy stored in the file
        # ASE puts a frame's stored forces/energy on a results object, not in arrays/info
        stored = atoms.calc.results if atoms.calc is not None else {}
        stored_forces = atoms.arrays.get("forces", stored.get("forces"))
        stored_pe = atoms.info.get("potential_energy", stored.get("energy"))
        if stored_forces is not None:
            force_mae_meV_A = float(np.abs(a.get_forces() - stored_forces).mean() * 1000.0)
        else:
            force_mae_meV_A = float("nan")
        if stored_pe is not None:
            pe_diff_meV_atom = float((pe0 - stored_pe) / len(a) * 1000.0)
        else:
            pe_diff_meV_atom = float("nan")
        dyn = VelocityVerlet(a, timestep=args.dt_fs * ase.units.fs)
        dyn.run(n_steps)
        e1 = a.get_potential_energy() + a.get_kinetic_energy()

        dq = a.get_positions() - ref_q
        dp = a.get_momenta() - ref_p
        # mass-scaled, as in FlashMD, so hydrogens and heavy atoms are comparable
        m = a.get_masses()[:, None]
        dq_ms = dq * np.sqrt(m)
        dp_ms = dp / np.sqrt(m)
        rows.append({
            "index": i,
            "natoms": len(a),
            "formula": a.get_chemical_formula(),
            "max_abs_dq_A": float(np.abs(dq).max()),
            "rms_dq_A": float(np.sqrt((dq ** 2).mean())),
            "rms_dq_massscaled": float(np.sqrt((dq_ms ** 2).mean())),
            "rms_dp_massscaled": float(np.sqrt((dp_ms ** 2).mean())),
            "ref_p_rms_massscaled": float(np.sqrt(((ref_p / np.sqrt(m)) ** 2).mean())),
            "energy_drift_meV_per_atom": float((e1 - e0) / len(a) * 1000.0),
            "force_mae_vs_stored_meV_A": force_mae_meV_A,
            "pe_diff_vs_stored_meV_atom": pe_diff_meV_atom,
        })
        if (i + 1) % 10 == 0:
            r = rows[-1]
            print(f"{i+1}: {r['formula']}, force mae vs stored {r['force_mae_vs_stored_meV_A']:.2f} meV/A, "
                  f"rms dq {r['rms_dq_A']:.2e} A, "
                  f"rel dp {r['rms_dp_massscaled']/r['ref_p_rms_massscaled']:.2e}, "
                  f"drift {r['energy_drift_meV_per_atom']:.3f} meV/atom")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rms_dq = np.array([r["rms_dq_A"] for r in rows])
    rel_dp = np.array([r["rms_dp_massscaled"] / r["ref_p_rms_massscaled"] for r in rows])
    drift = np.array([r["energy_drift_meV_per_atom"] for r in rows])
    fmae = np.array([r["force_mae_vs_stored_meV_A"] for r in rows])
    print("\nsummary over", len(rows), "structures")
    print(f"  single-point force MAE vs stored forces: median {np.nanmedian(fmae):.3f} meV/A, "
          f"max {np.nanmax(fmae):.3f} meV/A  (near zero means same PET-MAD version)")
    print(f"  rms position difference: median {np.median(rms_dq):.2e} A, max {rms_dq.max():.2e} A")
    print(f"  relative momentum difference: median {np.median(rel_dp):.2e}, max {rel_dp.max():.2e}")
    print(f"  energy drift over stride: median {np.median(np.abs(drift)):.3f} meV/atom, "
          f"max {np.abs(drift).max():.3f} meV/atom")
    print("pass criterion: median position difference below 1e-3 A and relative momentum "
          "difference below 1e-2; if far above, the potential version, time step or units differ")
    print("written", args.out)


if __name__ == "__main__":
    main()
