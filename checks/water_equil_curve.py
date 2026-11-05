"""
How long does the water box take to equilibrate? Fits potential energy (averaged over runs
when several are given) against time with  E(t) = E_inf + A exp(-t / tau)  and prints the
relaxation time, how far from the plateau the energy is at chosen times, and block averages.

  python checks/water_equil_curve.py /storage3/lauren/data/water/run_*/equil.out     (existing 5 ps runs)
  python checks/water_equil_curve.py /storage3/lauren/data/water_pilot/pilot.out      (the 100 ps pilot)
"""
import sys

import numpy as np
from scipy.optimize import curve_fit

files = sys.argv[1:]
data = [np.loadtxt(f, comments="#") for f in files]
n = min(len(d) for d in data)
t = data[0][:n, 1]                                   # ps
pe = np.mean([d[:n, 5] for d in data], axis=0)       # eV per box
T = np.mean([d[:n, 3] for d in data], axis=0)
print(f"{len(files)} file(s), {t[-1]:.1f} ps")
w = max(1, int(round(1.0 / (t[1] - t[0]))))          # 1 ps blocks
step = max(1, len(t) // (w * 20)) * w                # print at most ~20 blocks
for k in range(0, len(t) - w + 1, step):
    print(f"  t {t[k]:6.1f}-{t[k + w - 1]:6.1f} ps: potential {pe[k:k + w].mean():10.3f} eV, T {T[k:k + w].mean():6.1f} K")
skip = t > 0.2                                       # skip the first 0.2 ps (velocity thermalisation)
f = lambda x, e, a, tau: e + a * np.exp(-x / tau)
p0 = (pe[skip][-1], pe[skip][0] - pe[skip][-1], 2.0)
try:
    (e_inf, a, tau), cov = curve_fit(f, t[skip], pe[skip], p0=p0, maxfev=20000,
                                     bounds=([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 5 * t[-1]]))
    err = np.sqrt(np.diag(cov))
    print(f"fit: plateau {e_inf:.3f} eV, amplitude {a * 1000:+.0f} meV, relaxation time {tau:.2f} +- {err[2]:.2f} ps")
    if tau > 0.4 * t[-1] or err[2] > tau:
        print(f"  NOTE: the window ({t[-1]:.0f} ps) is too short to see the plateau; the fitted relaxation time is "
              f"not reliable. Only the 100 ps pilot can measure it.")
        raise SystemExit
    kin_per_K = (3 * 192 - 3) / 2 * 8.617333e-5      # eV per K for 192 atoms
    for tt in (5, 10, 20, 30, 50):
        gap = a * np.exp(-tt / tau)
        print(f"  remaining gap after {tt:3d} ps: {gap * 1000:+8.1f} meV per box (~{-gap / kin_per_K:+.2f} K drop it would cause in NVE)")
except RuntimeError as e:
    print("fit failed:", e)
