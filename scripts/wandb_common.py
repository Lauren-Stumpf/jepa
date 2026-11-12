"""
Shared definitions for the Weights and Biases project that holds the structured experiments.

Every run belongs to one experiment (wandb group, e.g. "E04") and one arm (wandb job_type,
e.g. "baseline", "plain"). Metric names are the ones metatrain itself logs, so runs logged
live by metatrain and runs imported from logs plot on the same axes.
"""

import os
import re

PROJECT = "jepa-flashmd-experiments"
ENTITY = "lauren-stumpf-oxford-university"
REPO = "/storage3/lauren/code/jepa"

# metatrain's own wandb keys (step = epoch); validation/training errors are in mass-scaled units
VAL_Q = "validation/position RMSE (per atom) [A]"
VAL_P = "validation/momentum RMSE (per atom) [(eV*u)^(1 per 2)]"
TRAIN_Q = "training/position RMSE (per atom) [A]"
TRAIN_P = "training/momentum RMSE (per atom) [(eV*u)^(1 per 2)]"
LR = "learning rate"
VAL_LOSS, TRAIN_LOSS = "validation/loss", "training/loss"
# added after evaluation (eval/eval_onestep.py on the test set)
TEST_Q = "test/position RMSE (per atom, mass-scaled)"
TEST_P = "test/momentum RMSE (per atom, mass-scaled)"

EXPERIMENTS = {
    "E00": "FlashMD baselines trained on all the labels (organic 8 fs; water 1, 4 and 16 fs)",
    "E01": "Organic, 10% of the labels: five pretraining variants against the baseline",
    "E02": "Organic, all the labels: normalised-target pretraining against the baseline",
    "E03": "Organic, 10% of the labels: freezing the encoder at the start of fine-tuning",
    "E04": "Water 16 fs, 10% of the labels: five pretraining variants against the baseline",
    "E05": "Hypothesis A: does the water gain last with ten times more training?",
    "E06": "Hypothesis B: does the water gain come from the regulariser rather than prediction?",
}

VARIANTS = ("plain", "normtarget", "resid", "proj", "residrel")

EPOCH_RE = re.compile(
    r"Epoch:\s+(\d+) \| learning rate: ([\d.e+-]+) \| training loss: ([\d.e+-]+) \| "
    r"training momentum RMSE \(per atom\): ([\d.e+-]+).*?training position RMSE \(per atom\): ([\d.e+-]+).*?"
    r"validation loss: ([\d.e+-]+) \| validation momentum RMSE \(per atom\): ([\d.e+-]+).*?"
    r"validation position RMSE \(per atom\): ([\d.e+-]+)")


def classify(name):
    """(experiment, arm, seed) for a run folder name from before the experiment project
    existed, or None for folders that are not imported (smoke tests, untrained encoders,
    stage 2 of heads-first, which is merged into its stage 1 run)."""
    m = re.search(r"_s(\d)$", name)
    seed = int(m.group(1)) if m else 0
    if name in ("flashmd_organic8_scratch", "flashmd_organic8_scratch_300"):
        return "E00", ("organic8-100ep" if name.endswith("scratch") else "organic8-300ep"), 0
    m = re.fullmatch(r"flashmd_water(1|4|16)_scratch", name)
    if m:
        return "E00", f"water{m.group(1)}fs", 0
    if re.fullmatch(r"flashmd_organic8_scratch_10pct_s\d", name):
        return "E01", "baseline", seed
    m = re.fullmatch(r"jepa_finetune_([a-z]+)_organic8_10pct_s\d", name)
    if m and m.group(1) in VARIANTS:
        return "E01", m.group(1), seed
    m = re.fullmatch(r"jepa_pretrain_([a-z]+)_organic8", name)
    if m and m.group(1) in VARIANTS:
        return "E01", f"pretrain-{m.group(1)}", 0
    if name == "jepa_finetune_organic8":
        return "E02", "normtarget", 0
    m = re.fullmatch(r"frozen_(random|plain)_organic8_10pct_s0", name)
    if m:
        return "E03", f"{m.group(1)}-frozen", 0
    m = re.fullmatch(r"headsfirst1_(random|plain)_organic8_10pct_s\d", name)
    if m:
        return "E03", f"{m.group(1)}-headsfirst", seed
    if re.fullmatch(r"flashmd_water16_scratch_10pct_s\d", name):
        return "E04", "baseline", seed
    m = re.fullmatch(r"jepa_finetune_([a-z]+)_water16_10pct_s\d", name)
    if m and m.group(1) in VARIANTS:
        return "E04", m.group(1), seed
    m = re.fullmatch(r"jepa_pretrain_([a-z]+)_water16", name)
    if m and m.group(1) in VARIANTS:
        return "E04", f"pretrain-{m.group(1)}", 0
    return None


def latest_train_log(run_dir):
    import glob
    logs = sorted(glob.glob(os.path.join(run_dir, "outputs", "*", "*", "train.log")), key=os.path.getmtime)
    return logs[-1] if logs else None


def epochs_from_log(path):
    """Per-epoch rows from a metatrain train.log, in metatrain's wandb key names."""
    rows = []
    for line in open(path, errors="replace"):
        m = EPOCH_RE.search(line)
        if m:
            e, lr, tl, tp, tq, vl, vp, vq = m.groups()
            rows.append({"epoch": int(e), LR: float(lr), TRAIN_LOSS: float(tl), TRAIN_P: float(tp), TRAIN_Q: float(tq),
                         VAL_LOSS: float(vl), VAL_P: float(vp), VAL_Q: float(vq)})
    return rows


def test_metrics(name, repo=REPO):
    """Test-set errors of a run from its eval/eval_onestep.py log, or {} if not evaluated."""
    path = os.path.join(repo, "results", "eval", "test", f"{name}.log")
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\s*mass-scaled displacement RMSE:\s+([\d.e+-]+)", line)
        if m:
            out[TEST_Q] = float(m.group(1))
        m = re.match(r"\s*mass-scaled momentum RMSE:\s+([\d.e+-]+)", line)
        if m:
            out[TEST_P] = float(m.group(1))
    return out
