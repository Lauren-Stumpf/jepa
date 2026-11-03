"""
Does metatrain's training-time rotational augmentation rotate the momentum
INPUT as well as the positions and the vector targets? If not, FlashMD
training is silently wrong with this metatrain/metatomic version.

Usage (in the training venv):
  python checks/check_augmentation.py data/organic8/val.xyz
"""

import sys

import torch
from metatrain.utils.augmentation import O3Augmenter
from metatrain.utils.data.readers.ase import read_systems

systems = read_systems(sys.argv[1])[:1]
R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=systems[0].positions.dtype)
aug = O3Augmenter(target_info_dict={}, extra_data_info_dict={})
new, _, _ = aug.apply_augmentations(systems, {}, [R])
p0 = systems[0].get_data("momentum").block().values.squeeze(-1)
p1 = new[0].get_data("momentum").block().values.squeeze(-1)
pos_ok = torch.allclose(new[0].positions, systems[0].positions @ R.T, atol=1e-8)
mom_ok = torch.allclose(p1, p0 @ R.T, atol=1e-8)
print("positions rotated:", bool(pos_ok))
print("input momenta rotated:", bool(mom_ok))
if not mom_ok:
    print("FAIL: momenta are not rotated with positions; do not train with this metatrain version")
    sys.exit(1)
print("OK")
