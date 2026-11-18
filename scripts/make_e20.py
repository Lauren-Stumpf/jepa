"""
Write the configs and run list for E20 (card in experiments/E20/card.md): the two runs that close
out the question "was the pretraining gain only extra optimiser steps?".

E20 organic: the organic 10% baseline trained for 500 epochs instead of 100 (8,400 frames, batch
  16, so 262,500 optimiser steps instead of 52,500). Copies of flashmd_organic8_scratch_10pct_s<seed>
  with only the number of epochs and how often a checkpoint is saved changed.
E20 water: the water baseline on ALL the labels (8,960 frames) trained for 140 epochs, which is
  78,400 optimiser steps: the same total as the E17 pretrained arms (22,400 pretraining steps +
  100 labelled epochs of 560 steps). Copies of flashmd_water16_scratch_10pct_s<seed> with the
  training file changed to train.xyz (as E17's baseline) and 140 epochs.

  python scripts/make_e20.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_e05_e06 import copy_config  # noqa: E402
from make_e11_e12 import write_runs  # noqa: E402
from make_e16_e17 import ALL_FRAMES  # noqa: E402

SCRIPT = "scripts/make_e20.py"
EPOCHS = r"^    num_epochs: 100 .*$"
CHECKPOINTS = r"^    checkpoint_interval: \d+ .*$"


def main():
    lines = []
    for s in (0, 1, 2):
        name = f"e20_baseline_organic8_10pct_500ep_s{s}"
        copy_config(f"flashmd_organic8_scratch_10pct_s{s}.yaml", name, "E20", "organic-baseline-500ep",
                    [(EPOCHS, "    num_epochs: 500          # E20: five times the 100-epoch baseline (262,500 steps)"),
                     (CHECKPOINTS, "    checkpoint_interval: 50  # E20: a checkpoint every 50 epochs (disk)")],
                    f"organic baseline on 10% of the labels, 500 epochs, seed {s}", script=SCRIPT)
        lines.append(f"{name}|train|20000|1|configs/{name}.yaml")
    for s in (0, 1, 2):
        name = f"e20_baseline_water16_all_140ep_s{s}"
        copy_config(f"flashmd_water16_scratch_10pct_s{s}.yaml", name, "E20", "water-all-baseline-140ep",
                    [ALL_FRAMES,
                     (r"^# Water 16 fs .*on 10% of the training pairs.*$",
                      "# Water 16 fs baseline trained on ALL the training pairs (E20)."),
                     (EPOCHS, "    num_epochs: 140          # E20: 78,400 steps, the same total as the E17 pretrained arms")],
                    f"water baseline on all the labels, 140 epochs, seed {s}", script=SCRIPT)
        lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
    write_runs("E20", lines)
    print("written: configs/e20_*.yaml, experiments/E20/runs.txt")


if __name__ == "__main__":
    main()
