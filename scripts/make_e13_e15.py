"""
Write the configs and run lists for E13, E14 and E15 (cards in experiments/E13 to E15;
instructions in experiments/INSTRUCTIONS_E13_E15.md).

E13: water, pretraining on only the 896 frames the baseline trains on (train_10pct.xyz).
  Four pretrainings: predictor plain or none, 20 or 200 epochs; otherwise as E04 plain / E11
  (batch 8 frames, weights 25/25/1, seed 0). Each followed by three training runs that are
  copies of jepa_finetune_plain_water16_10pct_s<seed> reading that pretraining.
E14: organic control that must work. Three training runs that are copies of
  jepa_finetune_plain_organic8_10pct_s<seed> reading the encoder of the full-data organic
  baseline (results/runs/supervised_encoder_organic8, made by train/encoder_from_baseline.py).
E15: organic, pretraining on only the 8,400 frames the baseline trains on (train_10pct.xyz).
  Four pretrainings as in E13 (batch 64, the E01 default); training runs copied from
  jepa_finetune_plain_organic8_10pct_s<seed>.

  python scripts/make_e13_e15.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_e05_e06 import PROJECT, REPO, copy_config  # noqa: E402
from make_e11_e12 import write_runs  # noqa: E402

SCRIPT = "scripts/make_e13_e15.py"
WATER_10 = ("CONFIG=configs/flashmd_water16_scratch.yaml TRAIN=data/water/pairs/16fs/train_10pct.xyz "
            "VAL=data/water/pairs/16fs/val.xyz")
ORGANIC_10 = ("CONFIG=configs/flashmd_organic8_scratch.yaml TRAIN=data/organic8/train_10pct.xyz "
              "VAL=data/organic8/val.xyz")
ARMS = [("plain", 20), ("plain", 200), ("none", 20), ("none", 200)]


def short(predictor):
    return "nopred" if predictor == "none" else predictor


def same_frames(exp, system, data, batch_flag, src_tpl, src_read_from, mem_train):
    lines = []
    for predictor, epochs in ARMS:
        p = short(predictor)
        pre = f"{exp.lower()}_pretrain_{p}_same10_ep{epochs}_{system}"
        lines.append(f"{pre}|pretrain|20000|1|{data} -- {batch_flag}--predictor {predictor} --epochs {epochs} "
                     f"--wandb {PROJECT} --wandb_group {exp} --wandb_job_type pretrain-{p}-ep{epochs}")
        for s in (0, 1, 2):
            name = f"{exp.lower()}_finetune_{p}_same10_ep{epochs}_{system}_10pct_s{s}"
            copy_config(src_tpl.format(s=s), name, exp, f"{p}-ep{epochs}",
                        [(src_read_from, f"      read_from: {REPO}/results/runs/{pre}/pretrained.ckpt"
                                         f"   # {p}, pretrained on the 10% frames only, {epochs} epochs")],
                        f"training from the {p} pretraining on the 10% frames only ({epochs} epochs), seed {s}",
                        script=SCRIPT)
            lines.append(f"{name}|train|{mem_train}|1|configs/{name}.yaml")
    write_runs(exp, lines)


def main():
    same_frames("E13", "water16", WATER_10, "--batch_size 8 ", "jepa_finetune_plain_water16_10pct_s{s}.yaml",
                r"^      read_from: /storage3/\S+/jepa_pretrain_plain_water16/pretrained\.ckpt$", 40000)
    same_frames("E15", "organic8", ORGANIC_10, "", "jepa_finetune_plain_organic8_10pct_s{s}.yaml",
                r"^      read_from: /storage3/\S+/jepa_pretrain_plain_organic8/pretrained\.ckpt(\s+#.*)?$", 20000)

    lines = []
    for s in (0, 1, 2):
        name = f"e14_finetune_supervised_organic8_10pct_s{s}"
        copy_config(f"jepa_finetune_plain_organic8_10pct_s{s}.yaml", name, "E14", "supervised-encoder",
                    [(r"^      read_from: /storage3/\S+/jepa_pretrain_plain_organic8/pretrained\.ckpt(\s+#.*)?$",
                      f"      read_from: {REPO}/results/runs/supervised_encoder_organic8/pretrained.ckpt"
                      f"   # encoder of the baseline trained on all organic labels (control)")],
                    f"training from the encoder of the full-data organic baseline, seed {s}", script=SCRIPT)
        lines.append(f"{name}|train|20000|1|configs/{name}.yaml")
    write_runs("E14", lines)
    print("written: configs/e13_*.yaml, configs/e14_*.yaml, configs/e15_*.yaml, experiments/E13..E15/runs.txt")


if __name__ == "__main__":
    main()
