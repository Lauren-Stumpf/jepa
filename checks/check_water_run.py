"""
Interrogate one q-TIP4P/f water run (output of data/water/run_one.sh) for
correctness. Every check prints PASS or FAIL with the measured number.

Checks (the "level 2" integrator and data-format checks):
  1. energy conservation: spread and linear drift of the conserved quantity
  2. temperature: mean near 300 K, fluctuation size consistent with 192 atoms
  3. momenta are synchronous with positions: p/m at step k matches the
     central difference (q[k+1]-q[k-1]) / (2 dt) to integrator accuracy
  4. units: momenta file units convert to ASE units consistently with the
     kinetic energy column in nve.out
  5. molecular integrity: O-H bonds within 0.85-1.15 A, no O-O pair closer
     than 2.3 A (using minimum image), throughout the run
  6. equilibrium structure: O-O radial distribution function first peak near
     2.75-2.85 A (q-TIP4P/f at 300 K; Habershon et al. 2009)
  7. no net drift: total momentum stays near zero
Also writes a PNG with energy, temperature and the O-O radial distribution
function so the run can be looked at.

Usage:
  python checks/check_water_run.py /storage3/lauren/data/water/run_001
"""

import argparse
import os
import sys

import numpy as np

HARTREE_EV = 27.211386245988
BOHR_A = 0.529177210903
AMU_ME = 1822.888486209  # electron masses per atomic mass unit
# i-PI atomic unit of momentum: m_e * bohr / (atomic time). ASE momentum unit:
# amu * A / (A * sqrt(amu/eV)) = sqrt(eV*amu). Conversion factor:
AU_TIME_FS = 0.0241888432650516  # atomic unit of time in fs
# p_ase = p_au * (bohr[A] / au_time[fs]) * (m_e in amu) / ASE velocity unit
# ASE: v in A/fs * amu -> momentum in amu*A/fs; ASE internal momentum unit is
# amu*A/(A*sqrt(amu/eV)) = sqrt(amu*eV); 1 A/fs = 1/ase.units.fs internal.
import ase.units as U
P_AU_TO_ASE = (BOHR_A / AU_TIME_FS) * (1.0 / AMU_ME) * (1.0 / U.fs)  # amu*A/fs -> internal


def read_ipi_xyz(path):
    """Return (frames[nframes, natoms, 3], symbols, cell_lengths[nframes, 3])."""
    frames, cells = [], []
    symbols = None
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line)
            header = f.readline()
            toks = header.split()
            i = toks.index("CELL(abcABC):")
            cells.append([float(toks[i + 1]), float(toks[i + 2]), float(toks[i + 3])])
            syms, xyz = [], []
            for _ in range(n):
                t = f.readline().split()
                syms.append(t[0])
                xyz.append([float(t[1]), float(t[2]), float(t[3])])
            if symbols is None:
                symbols = syms
            frames.append(xyz)
    return np.array(frames), symbols, np.array(cells)


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--dt_fs", type=float, default=0.25)
    ap.add_argument("--target_T", type=float, default=300.0)
    args = ap.parse_args()
    d = args.run_dir
    ok_all = True

    out = np.loadtxt(os.path.join(d, "nve.out"), comments="#")
    step, time_fs, conserved, temp, kin, pot = out[:, :6].T
    q, syms, cells = read_ipi_xyz(os.path.join(d, "nve.pos_0.xyz"))
    p_au, _, _ = read_ipi_xyz(os.path.join(d, "nve.mom_0.xyz"))
    n_frames, n_atoms = q.shape[:2]
    L = cells[0]
    masses = np.array([15.9994 if s == "O" else 1.00794 for s in syms])
    print(f"{n_frames} frames, {n_atoms} atoms, cell {L[0]:.3f} A, dt {args.dt_fs} fs")

    # 1. energy conservation
    spread = conserved.max() - conserved.min()
    slope = np.polyfit(time_fs, conserved, 1)[0] * 1000.0  # eV/ps
    ok_all &= report("energy conservation",
                     spread / n_atoms < 1e-3 and abs(slope) / n_atoms < 1e-3,
                     f"spread {spread*1000/n_atoms:.3f} meV/atom, drift {slope*1000/n_atoms:+.3f} meV/atom/ps")

    # 2. temperature
    t_mean, t_std = temp.mean(), temp.std()
    expected_std = args.target_T * np.sqrt(2.0 / (3 * n_atoms))
    # The run-mean temperature of one constant-energy run is not fixed at 300 K: it is set by
    # the total energy at the moment the thermostat is switched off, which itself fluctuates
    # from run to run (by roughly 12 K for 64 flexible waters). So this per-run test only
    # catches gross errors (60 K); whether the runs are centred on 300 K with the right spread
    # is tested across all runs by checks/check_water_temperatures.py.
    ok_all &= report("temperature",
                     abs(t_mean - args.target_T) < 60 and 0.3 * expected_std < t_std < 3 * expected_std,
                     f"mean {t_mean:.1f} K, std {t_std:.1f} K (expected ~{expected_std:.1f} K for {n_atoms} atoms)")

    # 3. momenta synchronous with positions (central difference)
    p_ase = p_au * P_AU_TO_ASE
    v_from_p = p_ase / masses[None, :, None]            # ASE internal velocity
    v_fd = (q[2:] - q[:-2]) / (2 * args.dt_fs * U.fs)    # A per internal time
    rel = np.sqrt(((v_from_p[1:-1] - v_fd) ** 2).mean() / (v_fd ** 2).mean())
    ok_all &= report("momenta synchronous with positions", rel < 0.02,
                     f"relative rms difference between p/m and central-difference velocity: {rel:.2e}")

    # 4. units: kinetic energy from converted momenta vs nve.out column
    ke = 0.5 * (p_ase ** 2 / masses[None, :, None]).sum(axis=(1, 2))
    ke_rel = np.abs(ke - kin).mean() / kin.mean()
    ok_all &= report("momentum units", ke_rel < 1e-3,
                     f"kinetic energy from momenta vs file column: relative difference {ke_rel:.2e}")

    # 5. molecular integrity, minimum image
    o_idx = [i for i, s in enumerate(syms) if s == "O"]
    h_idx = [i for i, s in enumerate(syms) if s == "H"]
    def mic(dv):
        return dv - L * np.round(dv / L)
    oh_min, oh_max, oo_min = 1e9, 0.0, 1e9
    for k in range(0, n_frames, max(1, n_frames // 50)):
        fr = q[k]
        for j, o in enumerate(o_idx):
            for h in (o + 1, o + 2):  # O H H ordering
                r = np.linalg.norm(mic(fr[h] - fr[o]))
                oh_min, oh_max = min(oh_min, r), max(oh_max, r)
        oo = fr[o_idx]
        dv = mic(oo[:, None, :] - oo[None, :, :])
        r = np.linalg.norm(dv, axis=-1) + np.eye(len(o_idx)) * 1e9
        oo_min = min(oo_min, r.min())
    ok_all &= report("molecular integrity", 0.85 < oh_min and oh_max < 1.15 and oo_min > 2.3,
                     f"O-H bond range {oh_min:.3f} to {oh_max:.3f} A, closest O-O {oo_min:.3f} A")

    # 6. O-O radial distribution function first peak
    edges = np.arange(2.0, 6.0, 0.05)
    hist = np.zeros(len(edges) - 1)
    n_used = 0
    for k in range(0, n_frames, max(1, n_frames // 100)):
        oo = q[k][o_idx]
        dv = mic(oo[:, None, :] - oo[None, :, :])
        r = np.linalg.norm(dv, axis=-1)[np.triu_indices(len(o_idx), 1)]
        hist += np.histogram(r, bins=edges)[0]
        n_used += 1
    rho = len(o_idx) / np.prod(L)
    shell = 4 * np.pi * (edges[:-1] + 0.025) ** 2 * 0.05 * rho * len(o_idx) / 2 * n_used
    g = hist / shell
    centres = edges[:-1] + 0.025
    peak = centres[np.argmax(g[centres < 3.5])]
    ok_all &= report("O-O radial distribution peak", 2.7 < peak < 2.9,
                     f"first peak at {peak:.2f} A, height {g[centres < 3.5].max():.2f} (literature ~2.8 A, ~3)")

    # 7. net momentum
    net = np.abs(p_ase.sum(axis=1)).max()
    typical = np.sqrt((p_ase ** 2).mean())
    ok_all &= report("no net drift", net < 0.05 * typical * np.sqrt(n_atoms),
                     f"max |total momentum| {net:.3e} vs typical per-atom {typical:.3e}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
        ax[0].plot(time_fs / 1000, (conserved - conserved[0]) * 1000 / n_atoms, lw=0.6)
        ax[0].set_xlabel("time (ps)"); ax[0].set_ylabel("conserved energy change (meV/atom)")
        ax[1].plot(time_fs / 1000, temp, lw=0.6); ax[1].axhline(args.target_T, c="k", lw=0.5)
        ax[1].set_xlabel("time (ps)"); ax[1].set_ylabel("temperature (K)")
        ax[2].plot(centres, g); ax[2].set_xlabel("O-O distance (A)"); ax[2].set_ylabel("g(r)")
        fig.tight_layout()
        png = os.path.join(d, "check_water_run.png")
        fig.savefig(png, dpi=130)
        print("plot written", png)
    except Exception as e:  # matplotlib optional
        print("plot skipped:", e)

    print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
