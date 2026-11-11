"""
Save an UNTRAINED FlashMD encoder as a checkpoint in exactly the format the JEPA
pretraining writes (results/runs/<name>/pretrained.ckpt plus manifest.json), so the
frozen-encoder baseline can be fine-tuned through the same metatrain path as the
pretrained encoders.

Why this is needed: metatrain's "heads" fine-tuning (encoder frozen, only the output
heads train) can only start from a checkpoint. For the frozen-encoder experiment to
change one thing at a time, the baseline arm must go through the same fine-tuning
code as the pretrained arm, differing only in the encoder weights it starts from.

The model is built with the same function and the same empty target list as
train/pretrain_jepa.py, so fine-tuning adds the heads and fits the target scaler
exactly as it does for a pretrained encoder. The weights are PyTorch's default
initialisation under torch.manual_seed(seed), the same initialisation metatrain uses
when it trains a baseline from scratch.

  python train/make_random_checkpoint.py configs/flashmd_organic8_scratch.yaml \
      --seed 0 --out results/runs/random_encoder_organic8_s0
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.pretrain_jepa import build_model, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.out, "manifest.json")):
        print(f"{args.out} already has manifest.json; nothing to do")
        return
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    cfg, model_hypers, masses, timestep, dataset_info = load_config(args.config)
    model = build_model(model_hypers, dataset_info, masses, timestep)
    ckpt = model.get_checkpoint()
    ckpt["epoch"] = 0
    ckpt["best_epoch"] = 0
    torch.save(ckpt, os.path.join(args.out, "pretrained.ckpt"))

    n_params = sum(p.numel() for p in model.parameters())
    first = next(model.parameters()).detach().flatten()[:5].tolist()
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"kind": "untrained encoder (no pretraining)", "config": args.config, "seed": args.seed,
                   "parameters": n_params, "first_weights": first, "masses": masses,
                   "timestep_fs": timestep, "model_hypers": model_hypers}, f, indent=1)
    print(f"untrained encoder, seed {args.seed}, {n_params / 1e6:.1f}M parameters -> {args.out}/pretrained.ckpt")


if __name__ == "__main__":
    main()
