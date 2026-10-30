"""
One-step evaluation of a FlashMD model on frames from a Zenodo pairs file.

Frames are sampled spread across the file (every --every frames, up to --n),
because the file is grouped by structure. For each frame the model is applied
once, with no thermostat and no energy filter, and its predicted positions and
momenta are compared with the stored future_positions / future_momenta.

Errors are reported in FlashMD's convention: mass-scaled displacement
  (q_future - q_now) * sqrt(m)   in A * sqrt(amu)
and mass-scaled momentum
  p / sqrt(m)                    in (eV * amu)^0.5
as root mean square error over all atoms and components, plus plain RMSE on
positions (A) and momenta.

Model selection:
  --model pretrained   the released universal checkpoint for this stride
  --model <path.pt>    an exported metatomic model file (from metatrain)
  --model verlet       no learned model: 32 velocity Verlet steps with PET-MAD,
                       as a sanity reference (should give ~zero error)

Usage:
  python eval/eval_onestep.py data/universal_8fs.xyz --stride_fs 8 --n 2000 --every 500 \
      --model pretrained --out results/eval/pretrained_8fs.csv
"""

import argparse
import csv
import os

import ase.io
import ase.units
import numpy as np
import torch


def load_flashmd(model_spec, stride_fs, device):
    # flashmd 0.1.2: get_universal_model(time_step) loads the PET-MAD (PBEsol)
    # universal checkpoints, the same family the Zenodo pairs were made with
    from metatomic.torch import load_atomistic_model
    if model_spec == "pretrained":
        from flashmd import get_universal_model
        return get_universal_model(int(round(stride_fs))).to(device)
    return load_atomistic_model(model_spec).to(device)


_stepper_cache = {}


def flashmd_one_step(atoms, model, stride_fs, device):
    # Call FlashMD's stepper directly: ASE's MolecularDynamics.run() demands a
    # force calculator even though FlashMD never uses forces.
    from flashmd.stepper import FlashMDStepper
    from flashmd.ase.velocity_verlet import _convert_atoms_to_system
    key = id(model)
    if key not in _stepper_cache:
        caps = model.capabilities()
        n_steps = int([k for k in caps.outputs.keys() if "mtt::delta_" in k][0].split("_")[1])
        base = float(model.module.base_time_step)
        if abs(n_steps * base - stride_fs) > 1e-6:
            raise ValueError(f"model covers {n_steps * base} fs, asked for {stride_fs} fs")
        _stepper_cache[key] = (FlashMDStepper([model], n_steps, torch.device(device)),
                               getattr(torch, caps.dtype))
    stepper, dtype = _stepper_cache[key]
    system = _convert_atoms_to_system(atoms, dtype=dtype, device=torch.device(device))
    with torch.no_grad():
        new_system = stepper.step(system)
    q = new_system.positions.detach().cpu().numpy()
    p = new_system.get_data("momenta").block().values.squeeze(-1).detach().cpu().numpy()
    return q, p


def verlet_reference(atoms, calc, stride_fs, dt_fs=0.25):
    from ase.md.verlet import VelocityVerlet
    a = atoms.copy()
    a.calc = calc
    VelocityVerlet(a, timestep=dt_fs * ase.units.fs).run(int(round(stride_fs / dt_fs)))
    return a.get_positions(), a.get_momenta()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--stride_fs", type=float, default=8.0)
    ap.add_argument("--n", type=int, default=2000, help="number of frames to evaluate")
    ap.add_argument("--every", type=int, default=500, help="take one frame every this many")
    ap.add_argument("--offset", type=int, default=7, help="first frame index to take")
    ap.add_argument("--model", default="pretrained")
    ap.add_argument("--pet_mad_version", default="1.0.0")
    ap.add_argument("--out", default="results/eval/onestep.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "verlet":
        from pet_mad.calculator import PETMADCalculator
        calc = PETMADCalculator(version=args.pet_mad_version, device=device)
        model = None
    else:
        model = load_flashmd(args.model, args.stride_fs, device)
    print(f"device {device}, model {args.model}, stride {args.stride_fs} fs, "
          f"{args.n} frames every {args.every}")

    rows = []
    sq_dq_ms = sq_dp_ms = sq_dq = sq_dp = 0.0
    sq_ref_dq_ms = sq_ref_p_ms = 0.0
    n_entries = 0
    for i, atoms in enumerate(ase.io.iread(args.xyz, index=":", format="extxyz")):
        if (i - args.offset) % args.every != 0 or i < args.offset:
            continue
        if len(rows) >= args.n:
            break
        ref_q = atoms.arrays["future_positions"]
        ref_p = atoms.arrays["future_momenta"]
        m = atoms.get_masses()[:, None]
        if model is None:
            q, p = verlet_reference(atoms, calc, args.stride_fs)
        else:
            q, p = flashmd_one_step(atoms, model, args.stride_fs, device)
        dq = q - ref_q
        dp = p - ref_p
        dq_ms = dq * np.sqrt(m)
        dp_ms = dp / np.sqrt(m)
        ref_disp_ms = (ref_q - atoms.get_positions()) * np.sqrt(m)
        ref_p_ms = ref_p / np.sqrt(m)
        k = dq.size
        sq_dq_ms += (dq_ms ** 2).sum(); sq_dp_ms += (dp_ms ** 2).sum()
        sq_dq += (dq ** 2).sum(); sq_dp += (dp ** 2).sum()
        sq_ref_dq_ms += (ref_disp_ms ** 2).sum(); sq_ref_p_ms += (ref_p_ms ** 2).sum()
        n_entries += k
        kin_T = 2.0 * atoms.get_kinetic_energy() / (3.0 * len(atoms) * ase.units.kB)
        rows.append({
            "frame": i, "natoms": len(atoms), "formula": atoms.get_chemical_formula(),
            "kinetic_T_K": float(kin_T),
            "rmse_dq_massscaled": float(np.sqrt((dq_ms ** 2).mean())),
            "rmse_dp_massscaled": float(np.sqrt((dp_ms ** 2).mean())),
            "rmse_dq_A": float(np.sqrt((dq ** 2).mean())),
            "rmse_dp": float(np.sqrt((dp ** 2).mean())),
        })
        if len(rows) % 100 == 0:
            print(f"{len(rows)} frames, running mass-scaled RMSE: "
                  f"dq {np.sqrt(sq_dq_ms/n_entries):.4f}, dp {np.sqrt(sq_dp_ms/n_entries):.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\nsummary over {len(rows)} frames, {n_entries//3} atoms")
    print(f"  mass-scaled displacement RMSE: {np.sqrt(sq_dq_ms/n_entries):.4f} "
          f"(reference displacement RMS {np.sqrt(sq_ref_dq_ms/n_entries):.4f})")
    print(f"  mass-scaled momentum RMSE:     {np.sqrt(sq_dp_ms/n_entries):.4f} "
          f"(reference momentum RMS {np.sqrt(sq_ref_p_ms/n_entries):.4f})")
    print(f"  plain position RMSE: {np.sqrt(sq_dq/n_entries):.5f} A, plain momentum RMSE: {np.sqrt(sq_dp/n_entries):.5f}")
    print("compare the mass-scaled values with Fig. 8 of the FlashMD paper for the universal 8 fs model")
    print("written", args.out)


if __name__ == "__main__":
    main()
