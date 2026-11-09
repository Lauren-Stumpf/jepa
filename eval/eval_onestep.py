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


def _is_metatrain_export(model):
    return not any("mtt::delta_" in k for k in model.capabilities().outputs.keys())


def _metatrain_one_step(atoms, model, stride_fs, device, scaled_targets, momenta_difference=False, masses=None):
    # metatrain's FlashMD export: outputs "position" and "momentum" (absolute,
    # unscaled), reads system data "momentum" and "mass", buffer "timestep".
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatomic.torch import ModelEvaluationOptions, ModelOutput, System
    from metatomic.torch import systems_to_torch
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
    key = id(model)
    if key not in _stepper_cache:
        caps = model.capabilities()
        model_dt = float(model.module.timestep)
        if abs(model_dt - stride_fs) > 1e-6:
            raise ValueError(f"model timestep {model_dt} fs, asked for {stride_fs} fs")
        _stepper_cache[key] = (None, getattr(torch, caps.dtype))
    _, dtype = _stepper_cache[key]
    dev = torch.device(device)
    system = systems_to_torch(atoms, dtype=dtype, device=dev)
    n = len(atoms)
    samples = Labels(names=["system", "atom"], values=torch.tensor([[0, j] for j in range(n)], device=dev))
    xyz = Labels(names="xyz", values=torch.tensor([[0], [1], [2]], device=dev))
    # only momenta are attached; the model attaches its own masses (verify_masses would
    # reject the file's i-PI masses against ASE's)
    for name, arr, comps in (("momentum", atoms.get_momenta(), [xyz]),):
        vals = torch.tensor(arr, dtype=dtype, device=dev).unsqueeze(-1)
        system.add_data(name, TensorMap(keys=Labels.single().to(dev), blocks=[TensorBlock(
            values=vals, samples=samples, components=comps,
            properties=Labels(names=[name], values=torch.tensor([[0]], device=dev)))]))
    system = get_system_with_neighbor_lists(system, model.requested_neighbor_lists())
    opts = ModelEvaluationOptions(length_unit="Angstrom", outputs={
        "position": ModelOutput(per_atom=True), "momentum": ModelOutput(per_atom=True)})
    with torch.no_grad():
        out = model([system], opts, check_consistency=False)
    q = out["position"].block().values.squeeze(-1).detach().cpu().numpy()
    p = out["momentum"].block().values.squeeze(-1).detach().cpu().numpy()
    if scaled_targets:
        # model was trained on q + dq*sqrt(m) and p/sqrt(m) (data/prepare_organic.py, data/water/parse_runs.py);
        # with predict_momenta_as_difference the momentum target was p0 + (p1 - p0)/sqrt(m)
        m = atoms.get_masses() if masses is None else np.array([masses[int(z)] for z in atoms.numbers])
        sm = np.sqrt(m)[:, None]
        q0 = atoms.get_positions()
        q = q0 + (q - q0) / sm
        if momenta_difference:
            p0 = atoms.get_momenta()
            p = p0 + (p - p0) * sm
        else:
            p = p * sm
    return q, p


def flashmd_one_step(atoms, model, stride_fs, device, scaled_targets=False, momenta_difference=False, masses=None):
    if _is_metatrain_export(model):
        return _metatrain_one_step(atoms, model, stride_fs, device, scaled_targets, momenta_difference, masses)
    # flashmd 0.1.2 universal checkpoints: call the package stepper directly,
    # since ASE's MolecularDynamics.run() demands a force calculator.
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
    ap.add_argument("--every", type=int, default=2500,
                    help="take one frame every this many (the Zenodo file has 500 frames per structure; "
                         "2500 spreads 2000 samples over all 10000 structures)")
    ap.add_argument("--offset", type=int, default=7, help="first frame index to take")
    ap.add_argument("--model", default="pretrained")
    ap.add_argument("--pet_mad_version", default="1.0.0")
    ap.add_argument("--out", default="results/eval/onestep.csv")
    ap.add_argument("--scaled_targets", action="store_true",
                    help="the metatrain model was trained on mass-scaled targets (default for data/organic8)")
    ap.add_argument("--momenta_difference", action="store_true",
                    help="the model was trained with predict_momenta_as_difference: true (water 1 fs set)")
    ap.add_argument("--config", default=None,
                    help="metatrain yaml the model was trained with; its masses are used to undo the scaling")
    ap.add_argument("--max_T", type=float, default=5000.0,
                    help="skip frames whose stored kinetic temperature exceeds this (K)")
    ap.add_argument("--max_disp", type=float, default=5.0,
                    help="skip frames whose stored mass-scaled displacement RMS exceeds this")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    masses = None
    if args.config:
        import yaml
        with open(args.config) as f:
            masses = {int(k): float(v) for k, v in yaml.safe_load(f)["architecture"]["training"]["masses"].items()}
    if args.model == "verlet":
        from pet_mad.calculator import PETMADCalculator
        calc = PETMADCalculator(version=args.pet_mad_version, device=device)
        model = None
    else:
        model = load_flashmd(args.model, args.stride_fs, device)
    print(f"device {device}, model {args.model}, stride {args.stride_fs} fs, "
          f"{args.n} frames every {args.every}")

    rows = []
    broken = []
    sq_dq_ms = sq_dp_ms = sq_dq = sq_dp = 0.0
    sq_ref_dq_ms = sq_ref_p_ms = 0.0
    sq_ball_dq_ms = sq_still_dp_ms = 0.0
    by_element = {}
    n_entries = 0
    for i, atoms in enumerate(ase.io.iread(args.xyz, index=":", format="extxyz")):
        if (i - args.offset) % args.every != 0 or i < args.offset:
            continue
        if len(rows) >= args.n:
            break
        m = (atoms.get_masses() if masses is None else np.array([masses[int(z)] for z in atoms.numbers]))[:, None]
        q0 = atoms.get_positions()
        p0 = atoms.get_momenta()
        ref_q = atoms.arrays["future_positions"]
        ref_p = atoms.arrays["future_momenta"]
        if args.scaled_targets:
            # the data file stores mass-scaled targets (data/prepare_organic.py, data/water/parse_runs.py):
            # recover the raw future state before comparing
            ref_q = q0 + (ref_q - q0) / np.sqrt(m)
            ref_p = p0 + (ref_p - p0) * np.sqrt(m) if args.momenta_difference else ref_p * np.sqrt(m)
        kin_T = 2.0 * atoms.get_kinetic_energy() / (3.0 * len(atoms) * ase.units.kB)
        ref_disp_ms = (ref_q - q0) * np.sqrt(m)
        ref_disp_rms = float(np.sqrt((ref_disp_ms ** 2).mean()))
        if kin_T > args.max_T or ref_disp_rms > args.max_disp:
            broken.append((i, atoms.get_chemical_formula(), float(kin_T), ref_disp_rms))
            continue
        if model is None:
            q, p = verlet_reference(atoms, calc, args.stride_fs)
        else:
            q, p = flashmd_one_step(atoms, model, args.stride_fs, device, args.scaled_targets,
                                    args.momenta_difference, masses)
        dq = q - ref_q
        dp = p - ref_p
        dq_ms = dq * np.sqrt(m)
        dp_ms = dp / np.sqrt(m)
        ref_p_ms = ref_p / np.sqrt(m)
        # trivial predictors: "no motion" (q, p unchanged) and "ballistic" (q + p/m dt, p unchanged)
        ball_q = q0 + p0 / m * (args.stride_fs * ase.units.fs)
        ball_dq_ms = (ball_q - ref_q) * np.sqrt(m)
        still_dp_ms = (p0 - ref_p) / np.sqrt(m)
        k = dq.size
        sq_dq_ms += (dq_ms ** 2).sum(); sq_dp_ms += (dp_ms ** 2).sum()
        sq_dq += (dq ** 2).sum(); sq_dp += (dp ** 2).sum()
        sq_ref_dq_ms += (ref_disp_ms ** 2).sum(); sq_ref_p_ms += (ref_p_ms ** 2).sum()
        sq_ball_dq_ms += (ball_dq_ms ** 2).sum(); sq_still_dp_ms += (still_dp_ms ** 2).sum()
        for z in np.unique(atoms.numbers):
            sel = atoms.numbers == z
            e = by_element.setdefault(int(z), {"n": 0, "dq": 0.0, "dp": 0.0, "ball": 0.0, "ref_dq": 0.0})
            e["n"] += int(sel.sum()) * 3
            e["dq"] += (dq_ms[sel] ** 2).sum(); e["dp"] += (dp_ms[sel] ** 2).sum()
            e["ball"] += (ball_dq_ms[sel] ** 2).sum(); e["ref_dq"] += (ref_disp_ms[sel] ** 2).sum()
        n_entries += k
        rows.append({
            "frame": i, "natoms": len(atoms), "formula": atoms.get_chemical_formula(),
            "kinetic_T_K": float(kin_T),
            "rmse_dq_massscaled": float(np.sqrt((dq_ms ** 2).mean())),
            "rmse_dp_massscaled": float(np.sqrt((dp_ms ** 2).mean())),
            "rmse_dq_A": float(np.sqrt((dq ** 2).mean())),
            "rmse_dp": float(np.sqrt((dp ** 2).mean())),
            "ballistic_rmse_dq_massscaled": float(np.sqrt((ball_dq_ms ** 2).mean())),
            "nomotion_rmse_dq_massscaled": float(np.sqrt((ref_disp_ms ** 2).mean())),
            "nomotion_rmse_dp_massscaled": float(np.sqrt((still_dp_ms ** 2).mean())),
        })
        if len(rows) % 100 == 0:
            print(f"{len(rows)} frames, running mass-scaled RMSE: "
                  f"dq {np.sqrt(sq_dq_ms/n_entries):.4f}, dp {np.sqrt(sq_dp_ms/n_entries):.4f}")

    if not rows:
        raise SystemExit("no frames evaluated (all excluded or --n 0); nothing written")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    med_dq = np.median([r["rmse_dq_massscaled"] for r in rows])
    med_dp = np.median([r["rmse_dp_massscaled"] for r in rows])
    print(f"\nsummary over {len(rows)} frames, {n_entries//3} atoms "
          f"({len(broken)} frames excluded as broken reference runs)")
    for b in broken:
        print(f"  excluded frame {b[0]} {b[1]}: kinetic T {b[2]:.3g} K, stored displacement RMS {b[3]:.3g}")
    print(f"  per-frame median mass-scaled RMSE: dq {med_dq:.4f}, dp {med_dp:.4f}")
    print(f"  mass-scaled displacement RMSE: {np.sqrt(sq_dq_ms/n_entries):.4f}   "
          f"[no motion {np.sqrt(sq_ref_dq_ms/n_entries):.4f}, ballistic q + p/m dt {np.sqrt(sq_ball_dq_ms/n_entries):.4f}]")
    print(f"  mass-scaled momentum RMSE:     {np.sqrt(sq_dp_ms/n_entries):.4f}   "
          f"[momentum unchanged {np.sqrt(sq_still_dp_ms/n_entries):.4f}, reference momentum RMS {np.sqrt(sq_ref_p_ms/n_entries):.4f}]")
    print("  by element (mass-scaled RMSE: model dq | ballistic dq | no-motion dq | model dp):")
    for z in sorted(by_element):
        e = by_element[z]
        print(f"    Z={z:2d}: {np.sqrt(e['dq']/e['n']):.4f} | {np.sqrt(e['ball']/e['n']):.4f} | "
              f"{np.sqrt(e['ref_dq']/e['n']):.4f} | {np.sqrt(e['dp']/e['n']):.4f}")
    print(f"  plain position RMSE: {np.sqrt(sq_dq/n_entries):.5f} A, plain momentum RMSE: {np.sqrt(sq_dp/n_entries):.5f}")
    print("a model is useful only if its dq error is below the ballistic value and its dp error below the unchanged-momentum value")
    print("written", args.out)


if __name__ == "__main__":
    main()
