"""
Unit test for data.make_organic_split.unwrap_molecule.

Builds a 3-atom chain, wraps one atom by a box vector, and checks that
unwrapping restores it exactly and leaves already-connected atoms alone.

Run:  python checks/test_unwrap.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from make_organic_split import unwrap_molecule  # noqa: E402

L = np.array([20.0, 20.0, 20.0])
chain = np.array([[1.0, 5.0, 5.0], [2.2, 5.0, 5.0], [3.4, 5.0, 5.0]])   # 1.2 A bonds, intact
wrapped = chain.copy()
wrapped[0] += [20.0, 0.0, 0.0]                                          # atom 0 shown on far side
wrapped[2] -= [0.0, 20.0, 0.0]                                          # atom 2 shown on another side

q, shift, ok = unwrap_molecule(wrapped, L)
assert ok, "walk did not reach every atom"
# result must be the intact chain up to one common box translation
delta = q - chain
assert np.allclose(delta - delta[0], 0.0), f"not rigidly unwrapped: {delta}"
assert np.allclose(np.round(delta[0] / L) * L, delta[0]), "common offset is not a box vector"
assert np.allclose(q + 0 * shift, wrapped + shift), "shift inconsistent with positions"

# a genuinely fragmented molecule (atom 2 far from both others) must be reported
frag = chain.copy(); frag[2] = [9.0, 9.0, 9.0]
_, _, ok = unwrap_molecule(frag, L)
assert not ok, "fragmented molecule reported as complete"
print("unwrap test passed")
