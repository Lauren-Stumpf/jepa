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
# averages over consecutive blocks covering ALL the data (block length chosen to give ~20 blocks)
dt = t[1] - t[0]
block_ps = max(1.0, round(t[-1] / 20))
w = max(1, int(round(block_ps / dt)))
nb = len(t) // w
bt = np.array([t[k * w:(k + 1) * w].mean() for k in range(nb)])
bpe = np.array([pe[k * w:(k + 1) * w].mean() for k in range(nb)])
bT = np.array([T[k * w:(k + 1) * w].mean() for k in range(nb)])
print(f"averages over {block_ps:.0f} ps blocks (every data point used):")
for k in range(nb):
    print(f"  t {t[k * w]:6.1f}-{t[(k + 1) * w - 1]:6.1f} ps: potential {bpe[k]:10.3f} eV, T {bT[k]:6.1f} K")
# is the second half flat? straight-line fit to the block averages, error from the scatter
# of the blocks about the line (blocks of several ps are close to independent for water)
half = bt > t[-1] / 2
if half.sum() >= 4:
    x, y = bt[half], bpe[half]
    A = np.c_[x - x.mean(), np.ones(len(x))]
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    se = np.sqrt((resid ** 2).sum() / (len(x) - 2) / ((x - x.mean()) ** 2).sum())
    slope_meV = coef[0] * 1000
    print(f"second half ({x[0]:.0f}-{x[-1]:.0f} ps): potential energy trend {slope_meV:+.1f} +- {se * 1000:.1f} meV per ps "
          f"({abs(slope_meV) / max(se * 1000, 1e-9):.1f} standard errors from flat); "
          f"block-to-block scatter {resid.std(ddof=2) * 1000:.0f} meV")
    first = (bt > 5) & ~half
    if first.sum() >= 2:
        print(f"mean potential, 5 ps to half-way: {bpe[first].mean():.3f} eV; second half: {bpe[half].mean():.3f} eV; "
              f"difference {1000 * (bpe[half].mean() - bpe[first].mean()):+.0f} meV "
              f"(standard error about {1000 * np.sqrt(bpe[first].var(ddof=1) / first.sum() + bpe[half].var(ddof=1) / half.sum()):.0f} meV)")
skip = t > 0.2                                       # skip the first 0.2 ps (velocity thermalisation)
f = lambda x, e, a, tau: e + a * np.exp(-x / tau)
p0 = (pe[skip][-1], pe[skip][0] - pe[skip][-1], 2.0)
try:
    (e_inf, a, tau), cov = curve_fit(f, t[skip], pe[skip], p0=p0, maxfev=20000,
                                     bounds=([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 5 * t[-1]]))
    err = np.sqrt(np.diag(cov))
    print(f"fit: plateau {e_inf:.3f} eV, amplitude {a * 1000:+.0f} meV, relaxation time {tau:.2f} +- {err[2]:.2f} ps")
    if tau > 0.4 * t[-1] or err[2] > tau:
        print(f"  NOTE: the exponential fit found no clear decay within {t[-1]:.0f} ps, so its relaxation time is not "
              f"meaningful; use the block averages and the second-half trend above instead.")
        raise SystemExit
    kin_per_K = (3 * 192 - 3) / 2 * 8.617333e-5      # eV per K for 192 atoms
    for tt in (5, 10, 20, 30, 50):
        gap = a * np.exp(-tt / tau)
        print(f"  remaining gap after {tt:3d} ps: {gap * 1000:+8.1f} meV per box (~{-gap / kin_per_K:+.2f} K drop it would cause in NVE)")
except RuntimeError as e:
    print("fit failed:", e)
