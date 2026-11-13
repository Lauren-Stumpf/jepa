"""
Write the configs and run lists for two experiments (cards in experiments/E11 and experiments/E12).

E11, hypothesis H5: does the water gain come from pulling the features now and after the gap
  together, without any learned predictor?
  One pretraining identical to jepa_pretrain_plain_water16 except --predictor none, then
  fine-tunes that are copies of jepa_finetune_plain_water16_10pct_s<seed> reading it.

E12, hypothesis H2: does organic pretraining help when the gap is long enough for the
  features to have to change?
  One pretraining identical to jepa_pretrain_plain_organic8 except that its pairs are 100 fs
  apart (data/organic8_gap100, from data/make_gap_pairs.py) instead of 8 fs, then fine-tunes
  that are copies of jepa_finetune_plain_organic8_10pct_s<seed> reading it. The fine-tuning
  task (8 fs, 10% of the labels) is unchanged.

  python scripts/make_e11_e12.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from make_e05_e06 import HERE, PROJECT, REPO, WATER, copy_config  # noqa: E402

ORGANIC_GAP = ("CONFIG=configs/flashmd_organic8_scratch.yaml TRAIN=data/organic8_gap100/train.xyz "
               "VAL=data/organic8_gap100/val.xyz")


def write_runs(exp, lines):
    os.makedirs(os.path.join(HERE, "experiments", exp), exist_ok=True)
    with open(os.path.join(HERE, "experiments", exp, "runs.txt"), "w") as f:
        f.write("# name|kind|min GPU memory (MB)|runs per GPU|spec   (scripts/launch_runs.sh)\n" + "\n".join(lines) + "\n")


def main():
    # E11: no predictor, water 16 fs
    pre = "e11_pretrain_nopred_water16"
    lines = [f"{pre}|pretrain|20000|1|{WATER} -- --batch_size 8 --predictor none "
             f"--wandb {PROJECT} --wandb_group E11 --wandb_job_type pretrain-nopred"]
    for s in (0, 1, 2):
        name = f"e11_finetune_nopred_water16_10pct_s{s}"
        copy_config(f"jepa_finetune_plain_water16_10pct_s{s}.yaml", name, "E11", "nopred",
                    [(r"^      read_from: /storage3/\S+/jepa_pretrain_plain_water16/pretrained\.ckpt$",
                      f"      read_from: {REPO}/results/runs/{pre}/pretrained.ckpt")],
                    f"fine-tune of the no-predictor pretraining, seed {s}", script="scripts/make_e11_e12.py")
        lines.append(f"{name}|train|40000|1|configs/{name}.yaml")
    write_runs("E11", lines)

    # E12: plain, organic, 100 fs gap in pretraining only
    pre = "e12_pretrain_plain_gap100_organic8"
    lines = [f"{pre}|pretrain|20000|1|{ORGANIC_GAP} -- --predictor plain "
             f"--wandb {PROJECT} --wandb_group E12 --wandb_job_type pretrain-gap100"]
    for s in (0, 1, 2):
        name = f"e12_finetune_plain_gap100_organic8_10pct_s{s}"
        copy_config(f"jepa_finetune_plain_organic8_10pct_s{s}.yaml", name, "E12", "gap100",
                    [(r"^      read_from: /storage3/\S+/jepa_pretrain_plain_organic8/pretrained\.ckpt(\s+#.*)?$",
                      f"      read_from: {REPO}/results/runs/{pre}/pretrained.ckpt   # plain, pretraining pairs 100 fs apart")],
                    f"fine-tune of the 100 fs-gap plain pretraining, seed {s}", script="scripts/make_e11_e12.py")
        lines.append(f"{name}|train|20000|1|configs/{name}.yaml")
    write_runs("E12", lines)
    print("written: configs/e11_*.yaml, configs/e12_*.yaml, experiments/E11/runs.txt, experiments/E12/runs.txt")


if __name__ == "__main__":
    main()
