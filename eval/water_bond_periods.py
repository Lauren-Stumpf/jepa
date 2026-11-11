"""
How fast each kind of motion in our q-TIP4P/f water is, measured from our own runs, so
the 1, 4 and 16 fs strides can be placed against the real time scales (and the
textbook values on the slides replaced by measured ones).

From the constant-energy part of a few runs (0.25 fs per frame, i-PI output; molecules
are stored as O H H triplets):
  1. the length of every O-H bond and every H-O-H angle at every frame;
  2. the spectrum of each (average power over all bonds or angles), whose main peak gives
     the period of the stretch and of the bend;
  3. the spectrum of the hydrogen velocities, which shows every motion hydrogens take
     part in (rocking of whole molecules, bend, stretch) as separate peaks.

Periods are printed in femtoseconds and as wavenumbers (cm^-1, the unit spectroscopists
use), together with how many cycles each stride covers.

  python eval/water_bond_periods.py /storage3/lauren/data/water --runs 5 \
      --out results/eval/water_bond_periods.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from checks.check_water_run_io import read_ipi_xyz  # noqa: E402

DT_FS = 0.25
C_CM_PER_FS = 2.99792458e-5   # speed of light in cm per fs: wavenumber = 1 / (period_fs * c)


def min_image(d, box):
    return d - box * np.round(d / box)


def spectrum(series):
    """Average power spectrum over the columns of series[n_frames, n_signals]."""
    x = series - series.mean(0)
    x = x * np.hanning(len(x))[:, None]
    p = (np.abs(np.fft.rfft(x, axis=0)) ** 2).mean(1)
    f = np.fft.rfftfreq(len(x), d=DT_FS)        # cycles per fs
    return f, p


def peaks(f, p, lo_cm, hi_cm):
    """Frequency of the largest peak between two wavenumbers."""
    wn = f / C_CM_PER_FS
    sel = (wn >= lo_cm) & (wn <= hi_cm)
    i = np.argmax(np.where(sel, p, -1))
    return float(wn[i]), float(1.0 / f[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(args.runs_root, "run_*")))[:args.runs]
    bonds, angles, vel_h = [], [], []
    for d in dirs:
        q, symbols, cells = read_ipi_xyz(os.path.join(d, "nve.pos_0.xyz"))
        p, _, _ = read_ipi_xyz(os.path.join(d, "nve.mom_0.xyz"))
        assert all(symbols[3 * k] == "O" and symbols[3 * k + 1] == "H" and symbols[3 * k + 2] == "H"
                   for k in range(len(symbols) // 3)), "atoms are not stored as O H H triplets"
        box = cells[0][None, None, :]
        o, h1, h2 = q[:, 0::3], q[:, 1::3], q[:, 2::3]
        v1, v2 = min_image(h1 - o, box), min_image(h2 - o, box)
        r1, r2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
        bonds.append(np.concatenate([r1, r2], 1))
        cosang = (v1 * v2).sum(-1) / (r1 * r2)
        angles.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
        hmask = np.array([s == "H" for s in symbols])
        vel_h.append(p[:, hmask, :].reshape(len(p), -1))   # momenta of H (same mass): spectrum shape = velocity's
    nf = min(len(b) for b in bonds)
    bonds = np.concatenate([b[:nf] for b in bonds], 1)
    angles = np.concatenate([a[:nf] for a in angles], 1)
    vel_h = np.concatenate([v[:nf] for v in vel_h], 1)

    res = {"runs": [os.path.basename(d) for d in dirs], "frames_per_run": int(nf), "dt_fs": DT_FS,
           "mean_OH_A": float(bonds.mean()), "sd_OH_A": float(bonds.std()),
           "mean_HOH_deg": float(angles.mean()), "sd_HOH_deg": float(angles.std())}
    f, pb = spectrum(bonds)
    res["stretch_cm"], res["stretch_period_fs"] = peaks(f, pb, 2500, 4500)
    f, pa = spectrum(angles)
    res["bend_cm"], res["bend_period_fs"] = peaks(f, pa, 1000, 2200)
    f, pv = spectrum(vel_h)
    res["hydrogen_velocity_peaks"] = {name: dict(zip(("cm", "period_fs"), peaks(f, pv, lo, hi)))
                                      for name, lo, hi in (("rocking", 300, 1100), ("bend", 1100, 2200), ("stretch", 2500, 4500))}
    for stride in (1, 4, 16):
        res[f"cycles_in_{stride}fs"] = {"stretch": stride / res["stretch_period_fs"], "bend": stride / res["bend_period_fs"],
                                        "rocking": stride / res["hydrogen_velocity_peaks"]["rocking"]["period_fs"]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    np.savez(args.out.replace(".json", "_spectra.npz"), wavenumber_cm=f / C_CM_PER_FS, hydrogen_velocity=pv)
    print(f"{len(dirs)} runs, {nf} frames each ({nf * DT_FS:.0f} fs)")
    print(f"O-H length {res['mean_OH_A']:.3f} +/- {res['sd_OH_A']:.3f} A; H-O-H angle {res['mean_HOH_deg']:.1f} +/- {res['sd_HOH_deg']:.1f} deg")
    print(f"stretch: {res['stretch_cm']:.0f} cm^-1, period {res['stretch_period_fs']:.1f} fs")
    print(f"bend:    {res['bend_cm']:.0f} cm^-1, period {res['bend_period_fs']:.1f} fs")
    for k, v in res["hydrogen_velocity_peaks"].items():
        print(f"hydrogen velocity peak, {k}: {v['cm']:.0f} cm^-1, period {v['period_fs']:.1f} fs")
    for stride in (1, 4, 16):
        c = res[f"cycles_in_{stride}fs"]
        print(f"{stride:2d} fs covers {c['stretch']:.2f} stretch cycles, {c['bend']:.2f} bend cycles, {c['rocking']:.2f} rocking cycles")


if __name__ == "__main__":
    main()
