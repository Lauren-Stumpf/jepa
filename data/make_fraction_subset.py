"""
Fixed fraction of the organic training set, spread over every molecule: keeps every
k-th frame of train.xyz (the file is grouped by molecule, so a head-of-file cut would
drop molecules). The subset is the same for every seed and both arms.

  python data/make_fraction_subset.py data/organic8/train.xyz 10 data/organic8/train_10pct.xyz
"""
import sys

import ase.io

src, every, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
frames = [a for i, a in enumerate(ase.io.iread(src, index=":", format="extxyz")) if i % every == 0]
ase.io.write(dst, frames, format="extxyz")
print(f"{len(frames)} frames (every {every}th) -> {dst}")
