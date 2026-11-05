"""
Why are the water runs centred at ~295 K rather than 300 K? Three candidate causes give
three different signatures, all readable from the i-PI property files of every run:

  A. the thermostatted stage itself does not reach 300 K (definition or thermostat issue):
     mean temperature over the last 1 ps of equilibration is below 300 K
  B. the switch from the thermostatted stage (0.5 fs) to constant energy (0.25 fs) shifts
     the temperature: equilibration ends near 300 K but the constant-energy run starts and
     stays lower
  C. the system is not yet equilibrated: the temperature drifts downwards during the
     constant-energy run (first 0.5 ps warmer than last 0.5 ps), and the potential energy
     drifts during the last ps of equilibration

  python checks/diagnose_water_temperature.py /storage3/lauren/data/water
"""
import glob
import os
import sys

import numpy as np

root = sys.argv[1]
rows = []
for d in sorted(glob.glob(os.path.join(root, "run_*"))):
    try:
        eq = np.loadtxt(os.path.join(d, "equil.out"), comments="#")   # stride 20 steps = 10 fs
        nv = np.loadtxt(os.path.join(d, "nve.out"), comments="#")     # every step = 0.25 fs
    except OSError:
        continue
    t_eq = eq[:, 1]                     # ps
    last = t_eq > t_eq[-1] - 1.0        # last 1 ps of equilibration
    first_half_eq = (t_eq > t_eq[-1] - 2.0) & ~last
    n = len(nv)
    rows.append([
        eq[last, 3].mean(),                         # T, last ps of equilibration
        nv[: n // 4, 3].mean(),                     # T, first 0.5 ps constant energy
        nv[-n // 4:, 3].mean(),                     # T, last 0.5 ps constant energy
        nv[:, 3].mean(),                            # T, whole constant-energy run
        eq[last, 5].mean() - eq[first_half_eq, 5].mean(),   # potential drift over the last 2 ps of equilibration (eV)
        nv[0, 3],                                   # T at the first constant-energy step
        eq[-1, 3],                                  # T at the last equilibration output
    ])
r = np.array(rows)
k = len(r)
se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
print(f"{k} runs (mean +- standard error over runs)")
print(f"A. T over last 1 ps of equilibration:            {r[:,0].mean():6.1f} +- {se(r[:,0]):.1f} K")
print(f"B. T last equilibration output vs first NVE step: {r[:,6].mean():6.1f} vs {r[:,5].mean():6.1f} K")
print(f"   T over whole constant-energy run:              {r[:,3].mean():6.1f} +- {se(r[:,3]):.1f} K")
print(f"C. T first 0.5 ps vs last 0.5 ps of NVE:          {r[:,1].mean():6.1f} vs {r[:,2].mean():6.1f} K "
      f"(difference {np.mean(r[:,2]-r[:,1]):+.1f} +- {se(r[:,2]-r[:,1]):.1f} K)")
print(f"   potential energy change, ps 3-4 to ps 4-5 of equilibration: {r[:,4].mean()*1000:+.1f} +- {se(r[:,4])*1000:.1f} meV per box")
