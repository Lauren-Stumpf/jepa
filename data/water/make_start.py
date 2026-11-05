"""
Rescale the i-PI developers' 64-water file to experimental density (0.997 g/cm3 at 298 K,
as in FlashMD App. B.3; the file is at 1.010 g/cm3). Each molecule is moved rigidly
(its oxygen position is scaled, the hydrogens move with it), so no bond is stretched.

  python data/water/make_start.py data/water/water64_init_ipi.xyz data/water/water64_rho0997_unequilibrated.xyz
"""
import sys

import numpy as np

NA = 6.02214076e23
BOHR_CM = 0.529177210903e-8
src, dst = sys.argv[1], sys.argv[2]
lines = open(src).read().splitlines()
n = int(lines[0])
header = lines[1].split()
i = header.index("CELL(abcABC):")
L0 = float(header[i + 1])
mass_g = (n // 3) * 18.01528 / NA
L1 = (mass_g / 0.997) ** (1 / 3) / BOHR_CM          # bohr
s = L1 / L0
atoms = [l.split() for l in lines[2:2 + n]]
xyz = np.array([[float(v) for v in a[1:4]] for a in atoms])
sym = [a[0] for a in atoms]
assert sym[::3] == ["O"] * (n // 3), "expected O H H ordering"
for k in range(0, n, 3):
    shift = (s - 1.0) * xyz[k]
    xyz[k:k + 3] += shift
header[i + 1:i + 4] = [f"{L1:.5f}"] * 3
with open(dst, "w") as f:
    f.write(f"{n}\n{' '.join(header)}\n")
    for a, p in zip(sym, xyz):
        f.write(f"{a:>8s} {p[0]: .6e} {p[1]: .6e} {p[2]: .6e}\n")
print(f"box {L0:.5f} -> {L1:.5f} bohr ({L1 * 0.529177210903:.4f} A), density {mass_g / (L0 * BOHR_CM) ** 3:.4f} -> "
      f"{mass_g / (L1 * BOHR_CM) ** 3:.4f} g/cm3; written {dst}")
