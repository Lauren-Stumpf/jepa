"""
Recompute a finished pretraining run's validation diagnostics twice from its saved final
state: once with validation batches in file order (as the run logged them) and once with
the fixed-seed shuffled order that train/pretrain_jepa.py now uses.

Why: the validation file is grouped by molecule, so a file-order batch of 64 pairs holds
one molecule, and the per-batch diagnostics (variance and covariance terms, effective
rank, dead fraction) then measure the spread within one molecule. Training batches are
shuffled across molecules, so the two were not comparable. Reporting both orders side by
side shows how much of the logged train/validation gap came from batch makeup.

The loss terms are computed with the same functions and the same code path as the
validation loop in train/pretrain_jepa.py (no augmentation, encoder through encode()).
The file-order column should reproduce the last epoch of the run's history.json; the
script prints both so this can be checked.

  python eval/recompute_val_diagnostics.py results/runs/jepa_pretrain_plain_organic8 \
      --val data/organic8/val.xyz --out results/eval/val_diagnostics_plain.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.pretrain_jepa import (VAL_SHUFFLE_SEED, PairData, Predictor, Projector, build_model,  # noqa: E402
                                 encode, jepa_loss, load_config, make_batch, normalise_embeddings,
                                 vicreg_terms)


def evaluate(frames, model, pred, proj, embed, a, dtype, device):
    agg, nb = {}, 0
    with torch.no_grad():
        for b in range(0, len(frames), a["batch_size"]):
            now, fut = make_batch(frames[b:b + a["batch_size"]], model, dtype, device, augment=False)
            h_now = encode(model, now)
            loss, parts = jepa_loss(embed(proj(h_now)), embed(proj(encode(model, fut))), pred,
                                    a["w_inv"], a["w_var"], a["w_cov"], a["relative_loss"])
            _, _, de = vicreg_terms(h_now)
            parts.update({"enc_rank": de["eff_rank"], "enc_dead": de["dead_frac"]})
            parts["loss"] = loss.item()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--val", required=True)
    ap.add_argument("--max_frames", type=int, default=0, help="first N validation pairs only (smoke tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.run_dir, "manifest.json")))
    a = man["args"]
    variant = man["predictor_variant"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    cfg, model_hypers, masses, timestep, dataset_info = load_config(a["config"])
    model = build_model(model_hypers, dataset_info, masses, timestep)
    state = torch.load(os.path.join(args.run_dir, "state.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model = model.to(device=device, dtype=dtype)
    for additive in model.additive_models:
        additive.to(dtype=torch.float64)
    model.scaler.to(dtype=torch.float64)
    d = man["feature_dim"]
    if a["projector_dim"] > 0:
        proj = Projector(d, a["projector_dim"]); d_loss = a["projector_dim"]
    else:
        proj = torch.nn.Identity(); d_loss = d
    proj.load_state_dict(state["proj"])
    pred = Predictor(d_loss, a["pred_hidden"], variant=variant)
    pred.load_state_dict(state["pred"])
    proj, pred = proj.to(device=device, dtype=dtype), pred.to(device=device, dtype=dtype)
    model.eval(); pred.eval(); proj.eval()
    embed = normalise_embeddings if variant == "norm_target" else (lambda z: z)

    val = PairData(args.val, masses, args.max_frames)
    file_order = list(val.frames)
    order = np.random.default_rng(VAL_SHUFFLE_SEED).permutation(len(file_order))
    shuffled = [file_order[i] for i in order]
    torch.manual_seed(0)
    res = {"run": os.path.basename(os.path.normpath(args.run_dir)), "variant": variant,
           "epoch_of_state": state.get("epoch"), "val_pairs": len(file_order), "val_shuffle_seed": VAL_SHUFFLE_SEED,
           "file_order": evaluate(file_order, model, pred, proj, embed, a, dtype, device),
           "shuffled": evaluate(shuffled, model, pred, proj, embed, a, dtype, device)}
    hist = json.load(open(os.path.join(args.run_dir, "history.json")))
    res["logged_last_epoch"] = {k[4:]: v for k, v in hist[-1].items() if k.startswith("val_")}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    keys = ["inv_ratio", "var", "cov", "std_min", "dead_frac", "eff_rank", "enc_rank", "enc_dead"]
    print(f"{res['run']} ({variant}), state from epoch {res['epoch_of_state']}")
    print(f"{'diagnostic':12s} {'logged':>10s} {'file order':>11s} {'shuffled':>10s}")
    for k in keys:
        print(f"{k:12s} {res['logged_last_epoch'].get(k, float('nan')):10.4f} "
              f"{res['file_order'].get(k, float('nan')):11.4f} {res['shuffled'].get(k, float('nan')):10.4f}")


if __name__ == "__main__":
    main()
