"""
Write the configs and run lists for the two close-out experiments (cards in experiments/E20 and
experiments/E21): was the pretraining gain only extra optimiser steps?

E20 (organic, 10% of the labels, 8,400 frames, batch 16, so 525 steps per epoch): the baseline
  trained for 150 epochs (78,750 steps, the same total as E15's plain and no-predictor arms with
  200 pretraining epochs: 26,400 + 52,500 = 78,900) and for 500 epochs (262,500 steps, to see
  whether the baseline is still improving). Copies of flashmd_organic8_scratch_10pct_s<seed> with
  only the number of epochs (and, for 500 epochs, how often a checkpoint is saved) changed.
E21 (water, ALL the labels, 8,960 frames, batch 16, so 560 steps per epoch): the baseline trained
  for 140 epochs, which is 78,400 steps: the same total as the E17 pretrained arms (22,400
  pretraining steps + 100 labelled epochs of 560). Copies of flashmd_water16_scratch_10pct_s<seed>
  with the training file changed to train.xyz (exactly as E17's baseline) and 140 epochs.

Job types: E20 arms are "baseline-150ep" and "baseline-500ep" and are compared with E01's
100-epoch baseline; the E21 arm is "baseline", and the E17 arms are compared with it.

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
    for epochs, steps in ((150, "78,750"), (500, "262,500")):
        for s in (0, 1, 2):
            name = f"e20_baseline_organic8_10pct_{epochs}ep_s{s}"
            rules = [(EPOCHS, f"    num_epochs: {epochs}          # E20: {steps} optimiser steps")]
            if epochs == 500:
                rules.append((CHECKPOINTS, "    checkpoint_interval: 50  # E20: a checkpoint every 50 epochs (disk)"))
            copy_config(f"flashmd_organic8_scratch_10pct_s{s}.yaml", name, "E20", f"baseline-{epochs}ep", rules,
                        f"organic baseline on 10% of the labels, {epochs} epochs, seed {s}", script=SCRIPT)
            lines.append(f"{name}|train|20000|1|configs/{name}.yaml")
    write_runs("E20", lines)

    lines = []
    for s in (0, 1, 2):
        name = f"e21_baseline_water16_all_140ep_s{s}"
        copy_config(f"flashmd_water16_scratch_10pct_s{s}.yaml", name, "E21", "baseline",
                    [ALL_FRAMES,
                     (r"^# Water 16 fs .*on 10% of the training pairs.*$",
                      "# Water 16 fs baseline trained on ALL the training pairs (E21)."),
                     (EPOCHS, "    num_epochs: 140          # E21: 78,400 steps, the same total as the E17 pretrained arms")],
                    f"water baseline on all the labels, 140 epochs, seed {s}", script=SCRIPT)
        lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
    write_runs("E21", lines)
    print("written: configs/e20_*.yaml, configs/e21_*.yaml, experiments/E20/runs.txt, experiments/E21/runs.txt")


if __name__ == "__main__":
    main()
