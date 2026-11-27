"""
Write the configs and run lists for E26 (organic part) and E27.

E26, organic part (card experiments/E26): is the organic loss of the pretrained arm an effect of the
  learning rate? Baseline and plain-pretrained (the E01 encoder), 10% of the labels, 100 epochs,
  at learning rates 1e-4 and 1e-3 (the 3e-4 runs exist: flashmd_organic8_scratch_10pct_s<seed> and
  jepa_finetune_plain_organic8_10pct_s<seed>). Only the learning rate changes.
E27 (card experiments/E27): water, 10% of the labels, longer runs at equal total optimiser steps.
  Baseline 2,400 epochs (134,400 steps) against plain-pretrained (the E04 encoder, 22,400
  pretraining steps) then 2,000 epochs (112,000 steps): 134,400 in total. Only the number of
  epochs and the checkpoint interval change against the 100-epoch configs.
  Third arm, added 26 Nov after the first result: the same, starting from the plain encoder that
  was pretrained on only the 896 labelled frames (E13, 200 epochs, also 22,400 steps). It removes
  the extra frames (8,960 seen against the baseline's 896). The pretraining objective and the
  second warm-up and decay cycle of the learning rate remain.

  Control added 27 Nov (experiments/E27/runs_restart.txt): labels throughout, with the learning
  rate rising and falling twice at the same steps as the pretrained arms. Stage 1: the baseline
  trained for 400 epochs (22,400 steps, the same number as the pretraining), seed 0. Stage 2:
  its encoder saved in the pretrained-encoder format by train/encoder_from_baseline.py (the E14
  route: new output layers, as for every pretrained encoder; scripts/run_e27_restart.sh). Stage 3:
  2,000 epochs from that encoder with the fine-tuning configs of the pretrained arms, seeds 0 to 2.
  Against the third arm only the kind of the first 22,400 steps differs (labelled steps in place
  of pretraining steps, on the same 896 frames).

  python scripts/make_e26_e27.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_e05_e06 import REPO, copy_config  # noqa: E402
from make_e11_e12 import write_runs  # noqa: E402

SCRIPT = "scripts/make_e26_e27.py"
REPO_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LR = r"^    learning_rate: 3\.0e-4 .*$"
EPOCHS = r"^    num_epochs: 100 .*$"
CHECKPOINTS = r"^    checkpoint_interval: \d+ .*$"
READ_FROM = r"^      read_from: /storage3/\S+/jepa_pretrain_plain_water16/pretrained\.ckpt$"
RESTART = (f"      read_from: {REPO}/results/runs/e27_restart_encoder_water16/pretrained.ckpt"
           "   # encoder of the baseline trained with labels for 400 epochs (E27 control with two learning-rate cycles)")
SAME10 = (f"      read_from: {REPO}/results/runs/e13_pretrain_plain_same10_ep200_water16/pretrained.ckpt"
          "   # plain, pretrained on the 896 labelled frames only, 200 epochs (E13)")


def main():
    lines = []
    for arm, src in (("baseline", "flashmd_organic8_scratch_10pct_s{s}.yaml"),
                     ("plain", "jepa_finetune_plain_organic8_10pct_s{s}.yaml")):
        for tag, lr in (("lr1e-4", "1.0e-4"), ("lr1e-3", "1.0e-3")):
            for s in (0, 1, 2):
                name = f"e26_{arm}_organic8_10pct_{tag}_s{s}"
                copy_config(src.format(s=s), name, "E26", f"{arm}-{tag}",
                            [(LR, f"    learning_rate: {lr}   # E26: the only change (earlier runs used 3.0e-4)")],
                            f"organic {arm}, 10% of the labels, learning rate {lr}, seed {s}", script=SCRIPT)
                lines.append(f"{name}|train|20000|1|configs/{name}.yaml")
    write_runs("E26", lines)

    lines = []
    for arm, src, ep, note, extra in (
            ("baseline-2400ep", "flashmd_water16_scratch_10pct_s{s}.yaml", 2400, "134,400 labelled steps", []),
            ("plain-2000ep", "jepa_finetune_plain_water16_10pct_s{s}.yaml", 2000,
             "112,000 labelled steps after 22,400 pretraining steps: 134,400 in total", []),
            ("plain-same10-2000ep", "jepa_finetune_plain_water16_10pct_s{s}.yaml", 2000,
             "112,000 labelled steps after 22,400 pretraining steps on the 896 labelled frames: 134,400 in total",
             [(READ_FROM, SAME10)])):
        for s in (0, 1, 2):
            name = f"e27_{arm.replace('-', '_')}_water16_10pct_s{s}"
            copy_config(src.format(s=s), name, "E27", arm,
                        [(EPOCHS, f"    num_epochs: {ep}         # E27: {note}"),
                         (CHECKPOINTS, "    checkpoint_interval: 200  # E27: few checkpoints (disk); results use model.pt")] + extra,
                        f"water {arm}, 10% of the labels, seed {s}", script=SCRIPT)
            lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
    write_runs("E27", lines)

    # E27 control: labels throughout, two learning-rate cycles
    s1 = "e27_restart_stage1_400ep_water16_10pct_s0"
    copy_config("flashmd_water16_scratch_10pct_s0.yaml", s1, "E27", "restart-stage1",
                [(EPOCHS, "    num_epochs: 400          # E27 control, stage 1: 22,400 labelled steps, as many as the pretraining"),
                 (CHECKPOINTS, "    checkpoint_interval: 200  # E27: few checkpoints (disk); results use model.pt")],
                "water baseline, 10% of the labels, seed 0: stage 1 of the control with two learning-rate cycles", script=SCRIPT)
    restart_lines = [f"{s1}|train|40000|1|configs/{s1}.yaml"]
    for s in (0, 1, 2):
        name = f"e27_restart_2000ep_water16_10pct_s{s}"
        copy_config(f"jepa_finetune_plain_water16_10pct_s{s}.yaml", name, "E27", "restart-2000ep",
                    [(EPOCHS, "    num_epochs: 2000         # E27 control: 112,000 labelled steps after 22,400 labelled steps (stage 1): 134,400 in total"),
                     (CHECKPOINTS, "    checkpoint_interval: 200  # E27: few checkpoints (disk); results use model.pt"),
                     (READ_FROM, RESTART)],
                    f"water, 10% of the labels, seed {s}: 2,000 epochs from the encoder of the 400-epoch baseline", script=SCRIPT)
        restart_lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
    with open(os.path.join(REPO_LOCAL, "experiments/E27/runs_restart.txt"), "w") as f:
        f.write("# name|kind|min GPU memory (MB)|runs per GPU|spec   (scripts/launch_runs.sh)\n"
                "# stage 2 (the encoder of stage 1) is written by scripts/run_e27_restart.sh; the stage-3 runs wait for it\n"
                + "\n".join(restart_lines) + "\n")
    print("written: configs/e26_*.yaml, configs/e27_*.yaml, experiments/E26/runs.txt, experiments/E27/runs.txt")


if __name__ == "__main__":
    main()
