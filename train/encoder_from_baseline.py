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
eval/feature_change_vs_jump.py: every tensor whose name and shape exist in the target-free model.

Checks, all of which stop the script if they fail:
  1. every encoder tensor of the saved model is bitwise equal to the source tensor of that name;
  2. the only names of the target-free model not filled from the source are the scaler and
     additive models (as in load_encoder);
  3. reloading the saved checkpoint the way pretrained encoders are loaded gives the same
     per-atom features as the model in memory, on a few frames of the training file.

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
    shared = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in shared]
    bad_missing = [k for k in missing if not k.startswith(("scaler", "additive_models"))]
    assert not bad_missing, f"encoder weights missing from {args.source}: {bad_missing[:5]}"
    model.load_state_dict(shared, strict=False)
    copied = [k for k in shared if not k.startswith(("scaler", "additive_models"))]
    after = model.state_dict()
    for k in copied:
        assert torch.equal(after[k].cpu(), sd[k].to(after[k].dtype).cpu()), f"tensor {k} differs after copying"
    print(f"copied {len(copied)} encoder tensors from {args.source}; not copied (as for a pretrained encoder): "
          f"{len(missing)} scaler and additive-model tensors")

    os.makedirs(args.out, exist_ok=True)
    ckpt = model.get_checkpoint()
    ckpt["epoch"] = 0
    ckpt["best_epoch"] = 0
    path = os.path.join(args.out, "pretrained.ckpt")
    torch.save(ckpt, path)

    # reload the way pretrained encoders are loaded (load_encoder(kind="pretrained")) and compare features
    re = target_free(model_hypers, dataset_info, masses, timestep)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    re.load_state_dict({k: v for k, v in saved["model_state_dict"].items() if k != "finetune_config"})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device); re.to(device)
    frames = []
    for i, a in enumerate(ase.io.iread(args.check_frames, index=":", format="extxyz")):
        if i >= args.n_check:
            break
        frames.append((torch.tensor(a.numbers, dtype=torch.int32), torch.tensor(a.get_positions()),
                       torch.tensor(a.get_momenta()), torch.tensor(a.cell.array)))
    h_mem = features(model, frames, device)
    h_re = features(re, frames, device)
    diff = max(float(abs(a - b).max()) for a, b in zip(h_mem, h_re))
    assert diff == 0.0, f"features of the reloaded checkpoint differ by up to {diff}"
    print(f"reloaded checkpoint gives identical features on {len(frames)} frames")

    first = next(model.parameters()).detach().flatten()[:5].cpu().tolist()
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump({"kind": "encoder of a model trained with labels (E14 control)", "config": args.config,
                   "source": args.source, "source_epoch": src.get("best_epoch", src.get("epoch")),
                   "encoder_tensors_copied": len(copied), "not_copied": missing,
                   "first_weights": first, "masses": masses, "timestep_fs": timestep,
                   "model_hypers": model_hypers}, f, indent=1)
    print(f"written {path} and manifest.json")


if __name__ == "__main__":
    main()
