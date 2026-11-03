"""
Unit tests for the periodic unwrapping in data/make_organic_split.py.

Run:  python checks/test_unwrap.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from make_organic_split import nearest_image, unwrap_molecule  # noqa: E402

rng = np.random.default_rng(0)
L = 20.0
cell = np.eye(3) * L


def rigid(a, b):
    d = a - b
    return np.allclose(d - d[0], 0.0, atol=1e-9)


# 1. chain wrapped on one axis
chain = np.array([[1.0, 5.0, 5.0], [2.2, 5.0, 5.0], [3.4, 5.0, 5.0]])
w = chain.copy(); w[0] += [L, 0, 0]; w[2] -= [0, L, 0]
q, ok, cons = unwrap_molecule(w, cell)
assert ok and cons and rigid(q, chain), "single-axis wrap"

# 2. chain crossing the boundary twice (two axes)
w = chain.copy(); w[1] += [L, 0, 0]; w[2] += [L, L, 0]
q, ok, cons = unwrap_molecule(w, cell)
assert ok and cons and rigid(q, chain), "double crossing"

# 3. stretched 2.5 A bond across the boundary is followed (cutoff 3 A)
st = np.array([[1.0, 5.0, 5.0], [3.5, 5.0, 5.0], [6.0, 5.0, 5.0]]); st[1] += [0, 0, L]
q, ok, cons = unwrap_molecule(st, cell)
assert ok and cons and np.allclose(q[1] - q[0], [2.5, 0, 0]), "stretched bond"

# 4. branched molecule, random per-atom lattice shifts, random atom order
base = np.array([[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [1.4, 1.4, 0], [1.4, -1.4, 0],
                 [2.8, 0, 1.4], [-1.0, 1.0, 0], [-1.0, -1.0, 0], [4.2, 0, 0]], float) + 10.0
for _ in range(200):
    perm = rng.permutation(len(base))
    b = base[perm]
    shifted = b + rng.integers(-2, 3, size=b.shape) * L
    q, ok, cons = unwrap_molecule(shifted, cell)
    assert ok and cons and rigid(q, b), "branched random"

# 5. frame needing no unwrapping is returned unchanged
q, ok, cons = unwrap_molecule(base, cell)
assert ok and cons and np.allclose(q, base), "no-op"

# 6. future placement: independent shifts of now and future recover the true displacement
true_dq = rng.normal(0, 0.1, size=base.shape)
now_raw = base + rng.integers(-1, 2, size=base.shape) * L
fut_raw = base + true_dq + rng.integers(-1, 2, size=base.shape) * L
q_new, ok, cons = unwrap_molecule(now_raw, cell)
fut = q_new + nearest_image(fut_raw - now_raw, cell)
assert np.allclose(fut - q_new, true_dq), "future placement"

# 7. fragmented molecule reported
frag = chain.copy(); frag[2] = [9.0, 9.0, 9.0]
_, ok, _ = unwrap_molecule(frag, cell)
assert not ok, "fragmented"

# 8. molecule within bond_cut of its own image is reported inconsistent
small = np.eye(3) * 8.0
line = np.array([[0, 4, 4], [1.4, 4, 4], [2.8, 4, 4], [4.2, 4, 4], [5.6, 4, 4]], float)
_, ok, cons = unwrap_molecule(line, small)
assert ok and not cons, "self-image case must be flagged"

# 9. nearest image is exact for a non-orthogonal cell
tri = np.array([[20.0, 0, 0], [10.0, 17.32, 0], [0, 0, 20.0]])
d = np.array([[0.5 * 20 + 0.5 * 10, 0.5 * 17.32, 0.0]]) + np.array([[0.3, 0.2, 0.1]])
d_img = d + tri[0] + tri[1]
assert np.allclose(nearest_image(d_img, tri), nearest_image(d, tri)), "non-orthogonal"

print("unwrap tests passed")
