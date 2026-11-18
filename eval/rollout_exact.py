"""
E22 parts A2, A3 and the energies of part B: everything that needs the exact q-TIP4P/f water
force field (eval/qtip4pf_ase.py). CPU only; one force-driver process per worker.

A2  error on the model's own outputs. For each saved rollout step k (every 5th step for the
    first 2 ps, then every 50th step), run 16 fs of exact dynamics from the model's own state
    at step k and compare with the model's state at step k+1. The same is done from the TRUE
    state for the first 2 ps: the model's one-step prediction from the reference state k
    (saved by rollout_water.py) against the reference state k+1.
A3  chaos reference. For each test run, two exact 2 ps simulations from the run's first frame:
    one with every position moved by a random Gaussian displacement whose per-component root
    mean square equals the model's one-step position error for that element (measured on these
    28 runs: rollout step 1 against the reference), momenta unchanged; and one with no change,
    to measure the error from restarting at the stored precision. States every 16 fs.
B   potential energy (and so total energy) of every saved rollout state that is a multiple of
    the rollout's save interval, plus each reference run's own drift of i-PI's conserved energy.

Errors are root mean square over atoms and components of mass-scaled differences, as the
one-step test error: position (q - q_ref) * sqrt(m), momentum (p - p_ref) / sqrt(m).

  python eval/rollout_exact.py results/eval/e22/rollout_primary.npz --out results/eval/e22/exact_primary.npz --workers 16
"""

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.qtip4pf_ase import QTIP4PF, exact_dynamics, potential_energy  # noqa: E402

RAW_PER_STEP = 64
_calc = None


def _init():
    global _calc
    from multiprocessing.util import Finalize
    _calc = QTIP4PF()
    Finalize(None, _calc.close, exitpriority=10)   # stop the driver and remove its socket when the worker exits


def ms_rmse(q, p, q_ref, p_ref, masses):
    sm = np.sqrt(masses)[:, None]
    return (float(np.sqrt((((q - q_ref) * sm) ** 2).mean())), float(np.sqrt((((p - p_ref) / sm) ** 2).mean())))


def _a2(job):
    numbers, cell, masses, q0, p0, q1, p1 = job
    qe, pe = exact_dynamics(_calc, numbers, q0, p0, cell, masses, RAW_PER_STEP)
    return ms_rmse(q1, p1, qe, pe, masses)


def _a3(job):
    numbers, cell, masses, q0, p0, n_steps = job
    _, _, rq, rp = exact_dynamics(_calc, numbers, q0, p0, cell, masses, n_steps * RAW_PER_STEP, record_every=RAW_PER_STEP)
    return np.array(rq), np.array(rp)


def _energy(job):
    numbers, cell, masses, q = job
    return potential_energy(_calc, numbers, q, cell, masses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs_root", default="/storage3/lauren/data/water")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_a3", action="store_true", help="for control rollouts, which need only A2 and B")
    args = ap.parse_args()
    r = dict(np.load(args.rollout))   # dict: read each array once (an npz file decompresses on every access)
    steps = list(r["saved_steps"]); pos = {int(k): i for i, k in enumerate(steps)}
    numbers, cell, masses = r["numbers"], r["cell"], r["masses"]
    q, p = r["q"], r["p"]
    ref_q, ref_p = r["ref_q"], r["ref_p"]
    B, n_ref = ref_q.shape[0], ref_q.shape[1]
    n_steps = int(max(steps))
    alive_at = lambda k, b: not np.isnan(q[pos[k], b]).any()  # noqa: E731

    # A2, own outputs
    a2_steps = [k for k in range(0, n_ref - 1, 5)] + [k for k in range(150, n_steps, 50)]
    a2_steps = [k for k in a2_steps if k in pos and k + 1 in pos]
    jobs, where = [], []
    for k in a2_steps:
        for b in range(B):
            if alive_at(k + 1, b):
                jobs.append((numbers, cell, masses, q[pos[k], b], p[pos[k], b], q[pos[k + 1], b], p[pos[k + 1], b]))
                where.append((k, b))
    # A2, true states (first 2 ps): model's prediction from the reference state against the next reference state
    a2_true = np.full((len(r["a2_steps"]), B, 2), np.nan)
    for i, k in enumerate(r["a2_steps"]):
        for b in range(B):
            a2_true[i, b] = ms_rmse(r["a2_true_q"][i, b], r["a2_true_p"][i, b], ref_q[b, k + 1], ref_p[b, k + 1], masses)

    # A3: perturbation per element from the model's error after one rollout step on these runs
    rng = np.random.default_rng(args.seed)
    per_el = {}
    for z in np.unique(numbers):
        sel = numbers == z
        d = np.stack([q[pos[1], b, sel] - ref_q[b, 1, sel] for b in range(B) if alive_at(1, b)])
        per_el[int(z)] = float(np.sqrt((d ** 2).mean()))
    sigma = np.array([per_el[int(z)] for z in numbers])[:, None]
    a3_jobs = []
    if not args.skip_a3:
        for b in range(B):
            a3_jobs.append((numbers, cell, masses, ref_q[b, 0] + rng.normal(size=ref_q[b, 0].shape) * sigma, ref_p[b, 0], n_ref - 1))
            a3_jobs.append((numbers, cell, masses, ref_q[b, 0], ref_p[b, 0], n_ref - 1))

    # B: energies of saved states at multiples of the save interval (and every step of the first 2 ps)
    e_steps = [k for k in steps if k % 10 == 0 or k < n_ref]
    e_jobs, e_where = [], []
    for k in e_steps:
        for b in range(B):
            if alive_at(k, b):
                e_jobs.append((numbers, cell, masses, q[pos[k], b])); e_where.append((k, b))

    with Pool(args.workers, initializer=_init) as pool:
        a2_res = pool.map(_a2, jobs, chunksize=4)
        a3_res = pool.map(_a3, a3_jobs, chunksize=1) if a3_jobs else []
        e_res = pool.map(_energy, e_jobs, chunksize=16)
        pool.close()
        pool.join()        # let the workers exit normally so their drivers are closed

    a2_own = np.full((len(a2_steps), B, 2), np.nan)
    ki = {k: i for i, k in enumerate(a2_steps)}
    for (k, b), v in zip(where, a2_res):
        a2_own[ki[k], b] = v
    a3_err = np.full((B, 2, n_ref, 2), np.nan)          # run, (perturbed, unperturbed), step, (dq, dp)
    for j, (rq, rp) in enumerate(a3_res):
        b, kind = divmod(j, 2)
        for k in range(n_ref):
            a3_err[b, kind, k] = ms_rmse(rq[k], rp[k], ref_q[b, k], ref_p[b, k], masses)
    epot = np.full((len(e_steps), B), np.nan)
    ei = {k: i for i, k in enumerate(e_steps)}
    for (k, b), v in zip(e_where, e_res):
        epot[ei[k], b] = v
    ekin = np.full_like(epot, np.nan)
    for i, k in enumerate(e_steps):
        ekin[i] = 0.5 * ((p[pos[k]] ** 2).sum(-1) / masses[None, :]).sum(-1)
    ref_drift = []
    for name in r["runs"]:
        c = np.loadtxt(os.path.join(args.runs_root, str(name), "nve.out"), comments="#")[:, 2]   # conserved (eV)
        ref_drift.append(c[-1] - c[0])
    np.savez_compressed(args.out, a2_steps=np.array(a2_steps), a2_own=a2_own, a2_true_steps=r["a2_steps"], a2_true=a2_true,
                        a3_sigma_by_element=np.array([[z, s] for z, s in per_el.items()]), a3_err=a3_err,
                        energy_steps=np.array(e_steps), epot=epot, ekin=ekin, ref_conserved_drift=np.array(ref_drift),
                        rollout=args.rollout)
    print(f"written {args.out}: A2 {len(jobs)} exact steps, A3 {len(a3_jobs)} simulations, B {len(e_jobs)} energies")


if __name__ == "__main__":
    main()
