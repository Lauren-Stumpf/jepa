"""
Fixed 10% subset of a water pairs training file (data/water/parse_runs.py output),
spread over every run.

The file stores each run's pairs contiguously as (forward, time-reversed) couples:
frame 2j is forward pair j and frame 2j+1 is its time-reversed copy. Taking every
10th FRAME (as data/make_fraction_subset.py does for the organic file) would keep only
forward pairs here (with 40 frames per run, frames 0, 10, 20, 30 are all even), so the
subset would lose the time-reversed copies that the full set has. This script keeps
every 10th forward pair TOGETHER with its time-reversed copy, so the subset has the
same make-up as the full set: 2 forward + 2 reversed frames from each of the 224
training runs, 896 frames in total.

Check made on every kept couple: the reversed frame starts at the forward pair's
future positions (targets are stored mass-scaled with the config masses, so the
masses are read from the config to undo the scaling).

  python data/water/make_water_subset.py data/water/pairs/16fs/train.xyz 10 \
      data/water/pairs/16fs/train_10pct.xyz configs/flashmd_water16_scratch.yaml
"""
import sys

import ase.io
import numpy as np
import yaml

src, every, dst, config = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
masses = {int(k): float(v) for k, v in yaml.safe_load(open(config))["architecture"]["training"]["masses"].items()}
kept = [a for i, a in enumerate(ase.io.iread(src, index=":", format="extxyz")) if (i // 2) % every == 0]
assert len(kept) % 2 == 0, "odd number of kept frames: the file is not made of (forward, reversed) couples"
worst = 0.0
for fwd, rev in zip(kept[0::2], kept[1::2]):
    sm = np.sqrt([masses[int(z)] for z in fwd.numbers])[:, None]
    fwd_future = fwd.get_positions() + (fwd.arrays["future_positions"] - fwd.get_positions()) / sm
    worst = max(worst, float(np.abs(rev.get_positions() - fwd_future).max()))
assert worst < 1e-6, f"a kept couple is not (forward, reversed): largest mismatch {worst:.2e} A"
ase.io.write(dst, kept, format="extxyz")
print(f"{len(kept)} frames ({len(kept) // 2} forward pairs, each with its time-reversed copy; every {every}th pair); "
      f"couple check passed (largest mismatch {worst:.1e} A) -> {dst}")
