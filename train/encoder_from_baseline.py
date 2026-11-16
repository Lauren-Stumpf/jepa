"""
Save the ENCODER of a model trained with labels as an encoder-only checkpoint, in exactly the
format the JEPA pretraining writes (<out>/pretrained.ckpt plus manifest.json), so it can be
trained from through the same metatrain route as every pretrained encoder (E14).

E14 is a control that must work: an encoder trained on all 84,000 labelled organic frames
certainly holds useful information, so training on the 10% subset from it should clearly beat
the 10% baseline. If it does not, the route (checkpoint loading, new heads, target scaler) is
broken and no pretraining result can be trusted.

The model is built as in train/make_random_checkpoint.py and train/pretrain_jepa.py: the same
function, the same config, NO targets. So training from it adds the heads and fits the target
scaler exactly as for a pretrained encoder; the source model's heads and scaler are not carried
over. Weights are copied from the source checkpoint's best_model_state_dict (the model metatrain
exported as model.pt), with the copying rule of load_encoder(kind="baseline") in
eval/feature_change_vs_jump.py (every tensor whose name and shape exist in the target-free
model), except that the scaler and additive models are left at their defaults.

Checks, all of which stop the script if they fail:
  1. every copied tensor has the same precision as the model's and is bitwise equal to the source
     tensor after copying; the scaler and additive models are NOT copied and are checked to keep
     the target-free defaults (as for a pretrained encoder, whose scaler is fitted in training);
  2. the only names of the target-free model not filled from the source are the scaler and
     additive models (as in load_encoder);
  3. the saved checkpoint, reloaded from the key metatrain fine-tuning reads
     (best_model_state_dict, which must equal model_state_dict), gives the same per-atom features
     as the SOURCE model itself (rebuilt from the source checkpoint's own model_data with its heads,
     loaded strictly), on a few frames of the training file.

The source checkpoint is read with torch.load directly (no metatrain version upgrade), which is
right for checkpoints written by the training environment's metatrain (2026.4.1), as all of ours are.

  python train/encoder_from_baseline.py configs/flashmd_organic8_scratch.yaml \
      --source results/runs/flashmd_organic8_scratch/model.ckpt --out results/runs/supervised_encoder_organic8
"""

import argparse
import json
import os
import sys

import ase.io
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.feature_change_vs_jump import features  # noqa: E402
from train.pretrain_jepa import build_model, load_config  # noqa: E402


def same(x, y):
    """Bitwise equality, counting NaN as equal to NaN (the masses buffer holds NaN for atomic
    numbers that are not used), as in train/last_epoch_checkpoint.py."""
    x, y = x.cpu(), y.cpu()
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.is_floating_point():
        return torch.allclose(x, y, rtol=0, atol=0, equal_nan=True)
    return torch.equal(x, y)


def target_free(model_hypers, dataset_info, masses, timestep):
    model = build_model(model_hypers, dataset_info, masses, timestep).to(dtype=torch.float32)
    for additive in model.additive_models:
        additive.to(dtype=torch.float64)
    model.scaler.to(dtype=torch.float64)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--source", required=True, help="a metatrain checkpoint of a model trained with labels (model.ckpt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check_frames", default="data/organic8/train.xyz")
    ap.add_argument("--n_check", type=int, default=4)
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.out, "manifest.json")):
        print(f"{args.out} already has manifest.json; nothing to do")
        return

    cfg, model_hypers, masses, timestep, dataset_info = load_config(args.config)
    model = target_free(model_hypers, dataset_info, masses, timestep)

    src = torch.load(args.source, map_location="cpu", weights_only=False)
    sd = src.get("best_model_state_dict") or src["model_state_dict"]
    own = model.state_dict()
    # encoder tensors only: the scaler and additive models are left at the target-free defaults,
    # so nothing fitted to the source's targets is carried over
    head_free = lambda k: not k.startswith(("scaler", "additive_models"))  # noqa: E731
    shared = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape and head_free(k)}
    missing = [k for k in own if k not in shared]
    bad_missing = [k for k in missing if head_free(k)]
    assert not bad_missing, f"encoder weights missing from {args.source}: {bad_missing[:5]}"
    for k, v in shared.items():
        assert v.dtype == own[k].dtype, f"tensor {k}: source dtype {v.dtype}, model dtype {own[k].dtype}"
    model.load_state_dict(shared, strict=False)
    after = model.state_dict()
    for k in shared:
        assert same(after[k], sd[k]), f"tensor {k} differs after copying"
    for k in missing:
        assert same(after[k], own[k]), f"tensor {k} was changed although it should keep its default"
    print(f"copied {len(shared)} encoder tensors from {args.source}; left at their defaults: "
          f"{len(missing)} scaler and additive-model tensors")

    os.makedirs(args.out, exist_ok=True)
    ckpt = model.get_checkpoint()
    ckpt["epoch"] = 0
    ckpt["best_epoch"] = 0
    path = os.path.join(args.out, "pretrained.ckpt")
    torch.save(ckpt, path)

    # 1. reload from the key metatrain fine-tuning reads (best_model_state_dict, else model_state_dict)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("best_model_state_dict") is not None:
        # get_checkpoint() puts finetune_config into model_state_dict only, so compare without it
        a = {k: v for k, v in saved["best_model_state_dict"].items() if k != "finetune_config"}
        b = {k: v for k, v in saved["model_state_dict"].items() if k != "finetune_config"}
        assert a.keys() == b.keys() and all(same(a[k], b[k]) for k in a), \
            "best_model_state_dict and model_state_dict of the saved checkpoint differ"
        reload_sd = a
    else:
        reload_sd = saved["model_state_dict"]
    re = target_free(model_hypers, dataset_info, masses, timestep)
    re.load_state_dict({k: v for k, v in reload_sd.items() if k != "finetune_config"})

    # 2. the SOURCE model itself, with its own heads and targets, loaded strictly
    from metatrain.experimental.flashmd import FlashMD
    md = src["model_data"]
    source = FlashMD(md["model_hypers"], md["dataset_info"])
    source.set_masses(masses)
    source.set_timestep(timestep)
    # metatrain's trainer stores finetune_config inside the state dict; its own load_checkpoint
    # removes it before loading, and so do we
    source.load_state_dict({k: v for k, v in sd.items() if k != "finetune_config"}, strict=True)
    source = source.to(dtype=torch.float32)
    for additive in source.additive_models:
        additive.to(dtype=torch.float64)
    source.scaler.to(dtype=torch.float64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = []
    for i, a in enumerate(ase.io.iread(args.check_frames, index=":", format="extxyz")):
        if i >= args.n_check:
            break
        frames.append((torch.tensor(a.numbers, dtype=torch.int32), torch.tensor(a.get_positions()),
                       torch.tensor(a.get_momenta()), torch.tensor(a.cell.array)))
    h_src = features(source.to(device), frames, device)
    h_re = features(re.to(device), frames, device)
    scale = max(float(abs(h).max()) for h in h_src)
    diff = max(float(abs(x - y).max()) for x, y in zip(h_src, h_re))
    assert diff <= 1e-5 * max(scale, 1.0), f"saved encoder's features differ from the source model's by up to {diff} (largest feature {scale})"
    print(f"saved encoder reproduces the source model's features on {len(frames)} frames (largest difference {diff:.2e})")

    first = next(model.parameters()).detach().flatten()[:5].cpu().tolist()
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"kind": "encoder of a model trained with labels (E14 control)", "config": args.config,
                   "source": args.source, "source_epoch": src.get("best_epoch", src.get("epoch")),
                   "encoder_tensors_copied": len(shared), "not_copied": missing,
                   "first_weights": first, "masses": masses, "timestep_fs": timestep,
                   "model_hypers": model_hypers}, f, indent=1)
    print(f"written {path} and manifest.json")


if __name__ == "__main__":
    main()
