"""
E28 control: exact constant-energy dynamics of q-TIP4P/f water for as long as the rollouts, from
the same starts, saved in the same layout as eval/rollout_water.py so that one analysis
(eval/rollout_scores.py) reads the model's rollouts and the exact answer in the same way.

The starts, the saved steps, the box, the masses and the order of the runs are all taken from
an existing rollout file (--like): its ref_q[:, 0] and ref_p[:, 0] are the first stored frame of
each test run, which is the state every rollout started from.

Dynamics: eval/qtip4pf_ase.exact_dynamics, unchanged (velocity Verlet, 0.25 fs, the force driver
that made the data; checked against i-PI by eval/qtip4pf_ase.py). One state every 16 fs.
CPU only; one force-driver process per worker.

Two checks, with limits fixed before the run. The file is written either way, with the verdict:
  1. reproduces the data: after 16 fs each run matches the stored frame, root mean square
     position difference within 2 times the rounding step of the stored positions and momentum
     difference within 1% of the momentum spread (the limits of eval/qtip4pf_ase.check_against_ipi);
  2. conserves energy: the total energy of each run at the end differs from its start by at most
     1 meV per molecule (the model without correction lost 178 meV per molecule in 100 ps).

  python eval/exact_reference_water.py --like results/eval/e28/rollout_energy.npz \
      --out results/eval/e28/exact_reference.npz --workers 28
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import ase.units
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.qtip4pf_ase import QTIP4PF, exact_dynamics, potential_energy, stored_quantum  # noqa: E402

RAW_PER_STEP = 64            # 16 fs / 0.25 fs
ENERGY_EVERY = 10            # potential energy at every 10th saved 16 fs step
_calc = None


def _init():
    global _calc
    from multiprocessing.util import Finalize
    _calc = QTIP4PF()
    Finalize(None, _calc.close, exitpriority=10)   # stop the driver and remove its socket when the worker exits


def _run(job):
    b, numbers, cell, masses, q0, p0, n_steps, save_steps = job
    t0 = time.time()
    _, _, rq, rp = exact_dynamics(_calc, numbers, q0, p0, cell, masses, n_steps * RAW_PER_STEP, record_every=RAW_PER_STEP)
    rq, rp = np.array(rq), np.array(rp)                       # (n_steps + 1, atoms, 3), one state every 16 fs
    assert len(rq) == n_steps + 1, (len(rq), n_steps)
    ke = 0.5 * ((rp ** 2).sum(-1) / masses[None, :]).sum(-1)  # eV, every 16 fs
    e_steps = np.arange(0, n_steps + 1, ENERGY_EVERY)
    epot = np.array([potential_energy(_calc, numbers, rq[k], cell, masses) for k in e_steps])
    return b, rq[save_steps], rp[save_steps], ke, e_steps, epot, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, help="a rollout file of eval/rollout_water.py: starts, saved steps and run order are taken from it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--n_steps", type=int, default=0, help="16 fs steps; 0 = as many as the rollout file")
    args = ap.parse_args()
    import shutil
    assert shutil.which("i-pi-driver"), "i-pi-driver is not on PATH (source env_train.sh first)"
    r = dict(np.load(args.like, allow_pickle=True))
    numbers, cell, masses = r["numbers"], r["cell"], r["masses"]
    ref_q, ref_p = r["ref_q"], r["ref_p"]                     # (runs, 126, atoms, 3): the stored true frames, every 16 fs
    B, N = ref_q.shape[0], ref_q.shape[2]
    n_steps = args.n_steps or int(r["saved_steps"].max())
    save_steps = np.array([k for k in r["saved_steps"] if k <= n_steps])
    assert float(r["stride_fs"]) == 16.0 and r["temperature"].shape[1] == B

    jobs = [(b, numbers, cell, masses, ref_q[b, 0], ref_p[b, 0], n_steps, save_steps) for b in range(B)]
    q = np.full((len(save_steps), B, N, 3), np.nan); p = np.full_like(q, np.nan)
    T = np.full((n_steps + 1, B), np.nan)
    E_total = np.full((n_steps + 1, B), np.nan)               # at every ENERGY_EVERY-th step, NaN elsewhere
    t0 = time.time()
    with Pool(min(args.workers, B), initializer=_init) as pool:
        for b, sq, sp, ke, e_steps, epot, secs in pool.imap_unordered(_run, jobs):
            q[:, b], p[:, b] = sq, sp
            T[:, b] = 2.0 * ke / (3.0 * N * ase.units.kB)
            E_total[e_steps, b] = ke[e_steps] + epot
            print(f"run {b} ({r['runs'][b]}) done in {secs:.0f} s; {time.time() - t0:.0f} s since the start", flush=True)
        pool.close()
        pool.join()        # let the workers exit normally so their drivers are closed

    # check 1: after 16 fs each run matches the stored frame
    i1 = int(np.where(save_steps == 1)[0][0])
    rms = lambda d: float(np.sqrt((d ** 2).mean()))  # noqa: E731
    worst_q = worst_p = 0.0
    ok1 = True
    for b in range(B):
        tol_q = 2 * np.sqrt((stored_quantum(ref_q[b, 1]) ** 2).mean())
        dq = rms(q[i1, b] - ref_q[b, 1]); dp = rms(p[i1, b] - ref_p[b, 1]) / rms(ref_p[b, 1])
        worst_q, worst_p = max(worst_q, dq / tol_q), max(worst_p, dp)
        ok1 = ok1 and dq <= tol_q and dp <= 1e-2
    print(f"check 1, stored frame at 16 fs: largest position difference {worst_q:.2f} of its limit, "
          f"largest momentum difference {worst_p:.2e} of the momentum spread (limit 1e-2): {'PASS' if ok1 else 'FAIL'}")
    # check 2: total energy conserved
    n_mol = N // 3
    last = max(k for k in range(0, n_steps + 1, ENERGY_EVERY))
    drift = (E_total[last] - E_total[0]) / n_mol * 1e3         # meV per molecule, per run
    ok2 = bool(np.isfinite(drift).all() and np.abs(drift).max() <= 1.0)
    print(f"check 2, total energy after {last * 16 / 1000:.1f} ps minus at the start: mean {drift.mean():+.4f}, "
          f"largest size {np.abs(drift).max():.4f} meV per molecule (limit 1): {'PASS' if ok2 else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out, saved_steps=save_steps, q=q, p=p, temperature=T, stopped_at=np.full(B, -1),
        reason=np.array([""] * B), runs=r["runs"], numbers=numbers, cell=cell, masses=masses,
        ref_q=ref_q, ref_p=ref_p, model="exact q-TIP4P/f dynamics (velocity Verlet, 0.25 fs)", like=args.like,
        momentum_scale=1.0, stride_fs=16.0, conserve_energy=False, langevin_fs=0.0, target_T=float("nan"),
        total_energy=E_total, energy_drift_meV_per_molecule=drift, check_reproduces_data=ok1, check_conserves_energy=ok2)
    print(f"written {args.out}: {B} runs of {n_steps} steps ({n_steps * 16 / 1000:.1f} ps), {time.time() - t0:.0f} s")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
