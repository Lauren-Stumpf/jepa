"""
Across all water runs: are the run-mean temperatures centred on 300 K, with a spread
that matches what the protocol produces, and is any run an outlier beyond chance?

Each constant-energy run starts from the last frame of a thermostatted run, so its total
energy is a random draw from the 300 K canonical distribution and its mean temperature
varies from run to run. Expected spread of the run means: 300 K x sqrt(k / C), where C is
the heat capacity of the box; for 64 classical flexible waters C is about 9 k per molecule
(3 translational + 3 rotational + 3 vibrational degrees of freedom, each contributing
kinetic and potential terms, minus anharmonic effects), giving about 300/sqrt(576) = 12.5 K.
The measured spread is printed next to that estimate.

  python checks/check_water_temperatures.py /storage3/lauren/data/water
"""
import glob
import os
import sys

import numpy as np

root = sys.argv[1]
target = 300.0
names, means, first, last = [], [], [], []
for d in sorted(glob.glob(os.path.join(root, "run_*"))):
    f = os.path.join(d, "nve.out")
    if not os.path.exists(f):
        continue
    out = np.loadtxt(f, comments="#")
    names.append(os.path.basename(d))
    means.append(out[:, 3].mean())   # column 3: temperature{kelvin}
    q = len(out) // 4
    first.append(out[:q, 3].mean()); last.append(out[-q:, 3].mean())
means = np.array(means)
mu, sd = means.mean(), means.std(ddof=1)
sem = sd / np.sqrt(len(means))
expected_sd = target / np.sqrt(9 * 64)
print(f"{len(means)} runs: mean of run-mean temperatures {mu:.1f} K (standard error {sem:.1f} K), "
      f"spread {sd:.1f} K (estimate {expected_sd:.1f} K)")
z = (means - mu) / sd
order = np.argsort(np.abs(z))[::-1][:5]
print("most extreme runs:", ", ".join(f"{names[i]} {means[i]:.1f} K (z = {z[i]:+.1f})" for i in order))
# with n runs drawn from a normal distribution, the largest |z| is typically about sqrt(2 ln n)
z_max_expected = np.sqrt(2 * np.log(len(means)))
ok_centre = abs(mu - target) < 4 * sem + 2.0
ok_spread = 0.5 * expected_sd < sd < 2.0 * expected_sd
ok_tail = np.abs(z).max() < z_max_expected + 1.5
# Started from equilibrium, constant-energy dynamics cannot change the average over runs:
# the run-averaged temperature must be flat in time. A drift means the starts were not equilibrated.
drift = np.array(last) - np.array(first)
drift_se = drift.std(ddof=1) / np.sqrt(len(drift))
ok_flat = abs(drift.mean()) < 3 * drift_se + 0.5
for ok, name, detail in [
    (ok_centre, "centred on 300 K", f"{mu:.1f} K"),
    (ok_spread, "spread of run means", f"{sd:.1f} K vs estimate {expected_sd:.1f} K"),
    (ok_flat, "run-averaged temperature flat in time (equilibrated starts)",
     f"last quarter minus first quarter {drift.mean():+.2f} +- {drift_se:.2f} K"),
    (ok_tail, "no outlier beyond chance", f"largest |z| {np.abs(z).max():.1f}, typical largest for {len(means)} runs {z_max_expected:.1f}"),
]:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
print("ALL PASS" if (ok_centre and ok_spread and ok_tail and ok_flat) else "SOME CHECKS FAILED")
