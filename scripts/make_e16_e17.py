"""
Write the configs and run lists for E16 and E17 (cards in experiments/E16 and experiments/E17).

E16: water, 10% of the labels, baseline only, trained for as many optimiser steps as the
  pretrained arms get in total: 500 epochs (28,000 steps, to compare with E13) and 1,400 epochs
  (78,400 steps, to compare with E05 plain). Copies of flashmd_water16_scratch_10pct_s<seed>
  with only the number of epochs (and how often a checkpoint is saved) changed.
E17: water, ALL the labels (8,960 frames), 100 epochs: baseline, plain pretrained (E04's
  pretraining) and no-predictor pretrained (E11's pretraining). Copies of the 10% configs with
  the training file changed from train_10pct.xyz to train.xyz (and, for no predictor, the
  pretraining it reads).

  python scripts/make_e16_e17.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_e05_e06 import REPO, copy_config  # noqa: E402
from make_e11_e12 import write_runs  # noqa: E402

SCRIPT = "scripts/make_e16_e17.py"
ALL_FRAMES = (r"^    read_from: data/water/pairs/16fs/train_10pct\.xyz$",
              "    read_from: data/water/pairs/16fs/train.xyz   # E17: all 8,960 frames labelled")
# the source configs' comments say "10%"; replaced so the generated file does not mislead
NOT_10PCT = [(r"^# Water 16 fs .*on 10% of the training pairs.*$", "# Water 16 fs, trained on ALL the training pairs (E17)."),
             (r"^    num_epochs: 100 .*$", "    num_epochs: 100          # E17: the same length as the 10% runs")]


def main():
    lines = []
    for epochs, every in ((500, 50), (1400, 100)):
        for s in (0, 1, 2):
            name = f"e16_baseline_water16_10pct_{epochs}ep_s{s}"
            copy_config(f"flashmd_water16_scratch_10pct_s{s}.yaml", name, "E16", f"baseline-{epochs}ep",
                        [(r"^    num_epochs: 100 .*$", f"    num_epochs: {epochs}        # E16: as many optimiser steps as the pretrained arms get in total"),
                         (r"^    checkpoint_interval: \d+ .*$", f"    checkpoint_interval: {every}  # E16: a checkpoint every {every} epochs (disk)")],
                        f"baseline, {epochs} epochs, seed {s}", script=SCRIPT)
            lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
    write_runs("E16", lines)

    lines = []
    for s in (0, 1, 2):
        name = f"e17_baseline_water16_all_s{s}"
        copy_config(f"flashmd_water16_scratch_10pct_s{s}.yaml", name, "E17", "baseline", [ALL_FRAMES] + NOT_10PCT,
                    f"baseline on all the labels, 100 epochs, seed {s}", script=SCRIPT)
        lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
        name = f"e17_finetune_plain_water16_all_s{s}"
        copy_config(f"jepa_finetune_plain_water16_10pct_s{s}.yaml", name, "E17", "plain", [ALL_FRAMES] + NOT_10PCT,
                    f"plain pretrained, all the labels, 100 epochs, seed {s}", script=SCRIPT)
        lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
        name = f"e17_finetune_nopred_water16_all_s{s}"
        copy_config(f"jepa_finetune_plain_water16_10pct_s{s}.yaml", name, "E17", "nopred",
                    [ALL_FRAMES] + NOT_10PCT + [
                     (r"^      read_from: /storage3/\S+/jepa_pretrain_plain_water16/pretrained\.ckpt$",
                      f"      read_from: {REPO}/results/runs/e11_pretrain_nopred_water16/pretrained.ckpt   # no predictor, all frames (E11)")],
                    f"no-predictor pretrained, all the labels, 100 epochs, seed {s}", script=SCRIPT)
        lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
    write_runs("E17", lines)
    print("written: configs/e16_*.yaml, configs/e17_*.yaml, experiments/E16/runs.txt, experiments/E17/runs.txt")


if __name__ == "__main__":
    main()
