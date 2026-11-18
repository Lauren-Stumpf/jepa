"""
E22: roll out a trained water FlashMD model (metatrain export, model.pt) from the first frame of
each water test run, feeding each prediction back in as the next input, with no thermostat and
no energy filter. See experiments/E22/card.md.

One step is the same computation as eval/eval_onestep.py (_metatrain_one_step with
--scaled_targets and the config masses): the model reads positions and momenta, its outputs
are the mass-scaled targets, and the raw next state is recovered as
  q_next = q + (q_out - q) / sqrt(m),   p_next = p_out * sqrt(m).
Here the 28 test runs are stepped together in one batch. --check_onestep proves the batched step
gives the same predictions as eval_onestep.py's function on the 1,120 test pairs.

A rollout stops at its first unphysical step: an O-H bond (within a molecule) over 1.5 A, two
atoms closer than 0.7 A (nearest periodic copy), or an instantaneous temperature over 600 K.

--momentum_scale f multiplies every momentum by f after each step (must-fail controls only).

Saved (npz): states at every step up to 126, at every step that is a multiple of --save_every,
and at k+1 for every k that is a multiple of 50 (for part A2); temperature at every step; the
step and reason each rollout stopped; and, for part A2, the model's one-step prediction from the
TRUE state at every 5th step of the first 2 ps.

  python eval/rollout_water.py results/runs/flashmd_water16_scratch/model.pt configs/flashmd_water16_scratch.yaml \
      --out results/eval/e22/rollout_primary.npz
  python eval/rollout_water.py <model.pt> <config> --check_onestep data/water/pairs/16fs/test.xyz
  python eval/rollout_water.py <model.pt> <config> --check_wrap 10
"""

import argparse
import json
import os
import sys
import time

import ase
import ase.io
import ase.units
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.eval_onestep import _metatrain_one_step, load_flashmd  # noqa: E402
from eval.qtip4pf_ase import config_masses, read_run  # noqa: E402

STRIDE_FS = 16.0
RAW_PER_STEP = 64           # 16 fs / 0.25 fs
MAX_OH, MIN_DIST, MAX_T = 1.5, 0.7, 600.0


def test_runs(manifest, runs_root):
    runs = json.load(open(manifest))["split"]["runs"]["test"]
    return [os.path.join(runs_root, r) for r in runs]


def batched_step(model, numbers, q, p, cell, masses, device):
    """q, p: (B, N, 3) numpy (Angstrom, ASE momentum units); returns raw next (q, p).
    Same computation as eval_onestep._metatrain_one_step with scaled_targets, for B systems."""
    from metatensor.torch import Labels, TensorBlock, TensorMap
    from metatomic.torch import ModelEvaluationOptions, ModelOutput, System
    from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists
    dtype = getattr(torch, model.capabilities().dtype)
    dev = torch.device(device)
    B, N = q.shape[:2]
    types = torch.tensor(numbers, dtype=torch.int32, device=dev)
    cellt = torch.tensor(np.diag(cell), dtype=dtype, device=dev)
    samples = Labels(names=["system", "atom"], values=torch.tensor([[0, j] for j in range(N)], device=dev))
    xyz = Labels(names="xyz", values=torch.tensor([[0], [1], [2]], device=dev))
    systems = []
    for b in range(B):
        s = System(types=types, positions=torch.tensor(q[b], dtype=dtype, device=dev), cell=cellt,
                   pbc=torch.tensor([True, True, True], device=dev))
        s.add_data("momentum", TensorMap(keys=Labels.single().to(dev), blocks=[TensorBlock(
            values=torch.tensor(p[b], dtype=dtype, device=dev).unsqueeze(-1), samples=samples, components=[xyz],
            properties=Labels(names=["momentum"], values=torch.tensor([[0]], device=dev)))]))
        systems.append(get_system_with_neighbor_lists(s, model.requested_neighbor_lists()))
    opts = ModelEvaluationOptions(length_unit="Angstrom", outputs={
        "position": ModelOutput(per_atom=True), "momentum": ModelOutput(per_atom=True)})
    with torch.no_grad():
        out = model(systems, opts, check_consistency=False)
    qo = np.zeros((B, N, 3)); po = np.zeros((B, N, 3))
    for name, arr in (("position", qo), ("momentum", po)):
        block = out[name].block()
        sys_idx = block.samples.column("system").cpu().numpy()
        atom_idx = block.samples.column("atom").cpu().numpy()
        arr[sys_idx, atom_idx] = block.values.squeeze(-1).detach().cpu().numpy()
    sm = np.sqrt(masses)[None, :, None]
    return q + (qo - q) / sm, po * sm


def temperature(p, masses):
    """Instantaneous kinetic temperature, 2 KE / (3 N k_B), per system. p: (B, N, 3)."""
    ke = 0.5 * ((p ** 2).sum(-1) / masses[None, :]).sum(-1)
    return 2.0 * ke / (3.0 * p.shape[1] * ase.units.kB)


def unphysical(q, p, numbers, cell, masses, device):
    """Reason string per system ('' if physical). Molecules are O H H in file order."""
    x = torch.tensor(q, device=device)
    L = torch.tensor(cell, device=device)
    d = x[:, :, None, :] - x[:, None, :, :]
    d = d - L * torch.round(d / L)
    r = d.norm(dim=-1)
    n = q.shape[1]
    r = r + torch.eye(n, device=device) * 1e9
    oh = torch.stack([r[:, 0::3, :][:, torch.arange(n // 3), torch.arange(1, n, 3)],
                      r[:, 0::3, :][:, torch.arange(n // 3), torch.arange(2, n, 3)]], -1)
    max_oh = oh.flatten(1).max(1).values.cpu().numpy()
    min_r = r.flatten(1).min(1).values.cpu().numpy()
    T = temperature(p, masses)
    reasons = []
    for b in range(q.shape[0]):
        why = []
        if max_oh[b] > MAX_OH:
            why.append(f"O-H bond {max_oh[b]:.2f} A")
        if min_r[b] < MIN_DIST:
            why.append(f"atoms {min_r[b]:.2f} A apart")
        if T[b] > MAX_T:
            why.append(f"temperature {T[b]:.0f} K")
        reasons.append("; ".join(why))
    return reasons


def check_order(numbers):
    assert len(numbers) % 3 == 0 and (numbers[0::3] == 8).all() and (numbers[1::3] == 1).all() \
        and (numbers[2::3] == 1).all(), "atoms are not in O H H order per molecule"


def check_onestep(model, config, test_xyz, masses_z, device, batch=28):
    """Batched step against eval_onestep._metatrain_one_step on every pair of the test file."""
    frames = list(ase.io.iread(test_xyz, index=":", format="extxyz"))
    worst_q = worst_p = 0.0
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        numbers = chunk[0].numbers
        assert all((a.numbers == numbers).all() for a in chunk)
        cell = np.diag(chunk[0].cell.array)
        assert all(np.allclose(a.cell.array, np.diag(cell)) for a in chunk)
        masses = np.array([masses_z[int(z)] for z in numbers])
        q = np.stack([a.get_positions() for a in chunk]); p = np.stack([a.get_momenta() for a in chunk])
        qb, pb = batched_step(model, numbers, q, p, cell, masses, device)
        for j, a in enumerate(chunk):
            q1, p1 = _metatrain_one_step(a, model, STRIDE_FS, device, True, False, masses_z)
            # relative to the size of the coordinate: float32 rounding grows with |x| (positions are unwrapped)
            worst_q = max(worst_q, float((np.abs(qb[j] - q1) / np.maximum(np.abs(q1), 1.0)).max()))
            worst_p = max(worst_p, float(np.abs(pb[j] - p1).max() / np.sqrt((p1 ** 2).mean())))
    print(f"batched step against eval_onestep on {len(frames)} test pairs: largest position difference "
          f"{worst_q:.2e} (relative to the coordinate, or absolute in A below 1 A), largest momentum difference {worst_p:.2e} of the momentum spread")
    ok = worst_q <= 1e-5 and worst_p <= 1e-5
    print("PASS" if ok else "FAIL")
    return ok


def check_wrap(model, runs, masses_z, device, n=10, test_error=0.0038):
    """The model's predicted displacement and momenta must not change when a frame is wrapped
    into the box. Measured the same way as the one-step error (mass-scaled root mean square over atoms and
    components), and required to be under 1% of the model's one-step test error: a real
    dependence on wrapping would be of the order of the error itself, while float32 rounding of
    unwrapped coordinates near 100 A gives about 1e-6 (first run on the cluster, 18 Nov: the
    largest single-atom momentum difference was 1.65e-4 of the momentum spread, failing an
    absolute 1e-4 limit set without this measurement; the mass-scaled differences were 1e-6 to
    4e-6)."""
    worst_q = worst_p = worst_atom = 0.0
    for k in range(n):
        r = read_run(runs[k % len(runs)])
        i = (k * 797) % len(r["q"])
        masses = np.array([masses_z[int(z)] for z in r["numbers"]])
        sm = np.sqrt(masses)[:, None]
        q0, p0 = r["q"][i][None], r["p"][i][None]
        qw = q0 - r["cell"] * np.floor(q0 / r["cell"])
        qa, pa = batched_step(model, r["numbers"], q0, p0, r["cell"], masses, device)
        qb, pb = batched_step(model, r["numbers"], qw, p0, r["cell"], masses, device)
        worst_q = max(worst_q, float(np.sqrt(((((qa - q0) - (qb - qw))[0] * sm) ** 2).mean())))
        worst_p = max(worst_p, float(np.sqrt((((pa - pb)[0] / sm) ** 2).mean())))
        worst_atom = max(worst_atom, float(np.abs(pa - pb).max() / np.sqrt((pa ** 2).mean())))
    lim = 0.01 * test_error
    print(f"wrapped against unwrapped frames ({n}): mass-scaled difference position {worst_q:.2e}, momentum {worst_p:.2e} "
          f"(limit {lim:.1e}, 1% of the one-step test error); largest single-atom momentum difference {worst_atom:.2e} of the spread (limit 1e-2)")
    ok = worst_q <= lim and worst_p <= lim and worst_atom <= 1e-2   # also no single atom badly off
    print("PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("config")
    ap.add_argument("--runs_root", default="/storage3/lauren/data/water")
    ap.add_argument("--manifest", default="data/water/pairs/16fs/manifest.json")
    ap.add_argument("--n_steps", type=int, default=6250, help="16 fs steps (6,250 = 100 ps)")
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--momentum_scale", type=float, default=1.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--check_onestep", default="", help="test pairs file: run the one-step check only")
    ap.add_argument("--check_wrap", type=int, default=0, help="run the wrap check on this many frames only")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_flashmd(args.model, STRIDE_FS, device)
    masses_z = config_masses(args.config)
    runs = test_runs(args.manifest, args.runs_root)
    if args.check_onestep:
        sys.exit(0 if check_onestep(model, args.config, args.check_onestep, masses_z, device) else 1)
    if args.check_wrap:
        sys.exit(0 if check_wrap(model, runs, masses_z, device, args.check_wrap) else 1)
    assert args.out, "--out is required for a rollout"

    starts, refs = [], []
    for d in runs:
        r = read_run(d)
        # keep only what is used (frame 0, and every 16 fs for the first 2 ps), not 8,001 frames per run
        refs.append((r["q"][::RAW_PER_STEP].copy(), r["p"][::RAW_PER_STEP].copy()))
        starts.append({"q": r["q"][:1].copy(), "p": r["p"][:1].copy(), "numbers": r["numbers"], "cell": r["cell"]})
        del r
    numbers, cell = starts[0]["numbers"], starts[0]["cell"]
    assert all((s["numbers"] == numbers).all() and np.allclose(s["cell"], cell) for s in starts)
    check_order(numbers)
    masses = np.array([masses_z[int(z)] for z in numbers])
    B = len(starts)
    n_ref = min(len(rq) for rq, _ in refs)          # 126 states: 0 .. 2 ps

    # A2, true-state half: the model's one-step prediction from the reference state at every 5th step
    a2_steps = list(range(0, n_ref - 1, 5))
    a2_true_q, a2_true_p = [], []
    for k in a2_steps:
        qt = np.stack([rq[k] for rq, _ in refs]); pt = np.stack([rp[k] for _, rp in refs])
        q1, p1 = batched_step(model, numbers, qt, pt, cell, masses, device)
        a2_true_q.append(q1); a2_true_p.append(p1)

    save = sorted(set(range(0, min(n_ref + 1, args.n_steps + 1))) | set(range(0, args.n_steps + 1, args.save_every))
                  | {k + 1 for k in range(0, args.n_steps, 50)})
    save_set = set(save)
    saved_q = np.full((len(save), B, len(numbers), 3), np.nan)  # float64: A2 and A3 restart exact dynamics from these
    saved_p = np.full_like(saved_q, np.nan)
    T = np.full((args.n_steps + 1, B), np.nan)
    stopped_at = np.full(B, -1)
    reason = [""] * B
    q = np.stack([s["q"][0] for s in starts]); p = np.stack([s["p"][0] for s in starts])
    alive = np.ones(B, bool)
    pos = {k: i for i, k in enumerate(save)}
    t0 = time.time()
    for k in range(args.n_steps + 1):
        T[k, alive] = temperature(p[alive], masses)
        if k in save_set:
            saved_q[pos[k], alive] = q[alive]; saved_p[pos[k], alive] = p[alive]
        if k == args.n_steps or not alive.any():
            break
        idx = np.where(alive)[0]
        qn, pn = batched_step(model, numbers, q[idx], p[idx], cell, masses, device)
        pn = pn * args.momentum_scale
        q[idx], p[idx] = qn, pn
        why = unphysical(qn, pn, numbers, cell, masses, device)
        for j, b in enumerate(idx):
            if why[j]:
                alive[b] = False; stopped_at[b] = k + 1; reason[b] = why[j]
                print(f"run {os.path.basename(runs[b])} stopped at step {k + 1} ({(k + 1) * STRIDE_FS / 1000:.2f} ps): {why[j]}", flush=True)
        if (k + 1) % 500 == 0:
            print(f"step {k + 1}/{args.n_steps}, {alive.sum()} of {B} rollouts running, mean T {np.nanmean(T[k]):.1f} K, "
                  f"{time.time() - t0:.0f} s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, saved_steps=np.array(save), q=saved_q, p=saved_p, temperature=T, stopped_at=stopped_at,
        reason=np.array(reason), runs=np.array([os.path.basename(r) for r in runs]), numbers=numbers, cell=cell,
        masses=masses, ref_q=np.stack([rq[:n_ref] for rq, _ in refs]),
        ref_p=np.stack([rp[:n_ref] for _, rp in refs]),
        a2_steps=np.array(a2_steps), a2_true_q=np.stack(a2_true_q),
        a2_true_p=np.stack(a2_true_p), model=args.model, config=args.config,
        momentum_scale=args.momentum_scale, stride_fs=STRIDE_FS)
    print(f"written {args.out}: {alive.sum()} of {B} rollouts reached step {args.n_steps}")


if __name__ == "__main__":
    main()
