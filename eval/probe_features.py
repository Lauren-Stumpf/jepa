"""
Does JEPA pretraining put information about motion into the encoder features?

Linear probes on the frozen per-atom feature vector of the current state, fitted by
ridge regression and scored by R^2 on held-out frames, for the pretrained encoder and
for an untrained encoder of the same architecture (the control):

  dynamic targets   mass-scaled displacement magnitude over the stride; mass-scaled future
                    momentum magnitude; current momentum magnitude (an encoder input, sanity)
  static targets    element (one-hot, four classes); number of bonded neighbours
                    (within 1.25 x covalent radii)
  static fraction   the reverse regression: how much of the variance of the feature vector
                    is explained by the static descriptors alone (element one-hot and
                    neighbour count). This is "how static is the vector".

If pretraining raises the static fraction and the static probes but not the dynamic
probes, it encoded what each atom is, not how it moves.

  python eval/probe_features.py configs/flashmd_organic8_scratch.yaml data/organic8/val.xyz \
      --pretrained results/runs/jepa_pretrain_organic8/pretrained.ckpt --n 3000 --out results/eval/probe.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train.pretrain_jepa import PairData, build_model, encode, load_config, make_batch  # noqa: E402


ELEMENTS = [1, 6, 7, 8]


def static_descriptors(t, q, cell, pbc):
    """One-hot element (4) and bonded-neighbour count (1) per atom."""
    from ase import Atoms
    from ase.neighborlist import natural_cutoffs, neighbor_list
    a = Atoms(numbers=t.numpy(), positions=q.numpy(), cell=cell.numpy(), pbc=pbc.numpy())
    i, _ = neighbor_list("ij", a, natural_cutoffs(a, mult=1.25))
    counts = np.bincount(i, minlength=len(a)).astype(float)
    onehot = (t.numpy()[:, None] == np.array(ELEMENTS)[None, :]).astype(float)
    return np.c_[onehot, counts]


def features_and_targets(model, data, masses, device, batch=64):
    X, Y, S = [], [], []
    with torch.no_grad():
        for b in range(0, len(data), batch):
            frames = data.frames[b:b + batch]
            now, _ = make_batch(frames, model, torch.float32, device, augment=False)
            X.append(encode(model, now).cpu().numpy())
            for t, q, p, qf, pf, cell, pbc in frames:
                m = np.array([masses[int(z)] for z in t.numpy()])
                sm = np.sqrt(m)[:, None]
                dq_ms = np.linalg.norm((qf - q).numpy() * sm, axis=1)
                pf_ms = np.linalg.norm(pf.numpy() / sm, axis=1)
                p_ms = np.linalg.norm(p.numpy() / sm, axis=1)
                Y.append(np.stack([dq_ms, pf_ms, p_ms], 1))
                S.append(static_descriptors(t, q, cell, pbc))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(S)


def pooled_r2(X, Y, lam=1e-3, train_frac=0.7, seed=0):
    """Fraction of the total variance of Y (all columns pooled) explained by a ridge fit on X;
    robust to near-constant columns of Y, unlike a per-column R^2."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); ntr = int(train_frac * len(X)); tr, te = idx[:ntr], idx[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    A = np.c_[(X - mu) / sd, np.ones(len(X))]
    W = np.linalg.solve(A[tr].T @ A[tr] + lam * len(tr) * np.eye(A.shape[1]), A[tr].T @ Y[tr])
    res = ((Y[te] - A[te] @ W) ** 2).sum(); tot = ((Y[te] - Y[te].mean(0)) ** 2).sum()
    return float(1 - res / tot)


def ridge_r2(X, Y, lam=1e-2, train_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    ntr = int(train_frac * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xs = (X - mu) / sd
    A = np.c_[Xs, np.ones(len(X))]
    W = np.linalg.solve(A[tr].T @ A[tr] + lam * len(tr) * np.eye(A.shape[1]), A[tr].T @ Y[tr])
    pred = A[te] @ W
    ss_res = ((Y[te] - pred) ** 2).sum(0)
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("xyz")
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--n", type=int, default=3000, help="frames to use (spread over the file)")
    ap.add_argument("--out", default="results/eval/probe.csv")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, hypers, masses, timestep, info = load_config(args.config)
    data = PairData(args.xyz, masses)
    step = max(1, len(data) // args.n)
    data.frames = data.frames[::step][:args.n]
    print(f"{len(data)} frames, {sum(len(f[0]) for f in data.frames)} atoms")

    torch.manual_seed(0)
    untrained = build_model(hypers, info, masses, timestep).to(device)
    pretrained = build_model(hypers, info, masses, timestep)
    ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in ckpt["model_state_dict"].items() if k != "finetune_config"}
    pretrained.load_state_dict(sd)
    pretrained = pretrained.to(device)

    dyn_names = ["dynamic: mass-scaled displacement magnitude", "dynamic: mass-scaled future momentum magnitude",
                 "dynamic: current momentum magnitude (encoder input)"]
    rows = []
    for label, model in (("untrained", untrained), ("pretrained", pretrained)):
        X, Y, S = features_and_targets(model, data, masses, device)
        for n, r in zip(dyn_names, ridge_r2(X, Y)):
            rows.append({"encoder": label, "target": n, "r2": float(r)})
            print(f"{label:10s} {n:55s} R^2 = {r:.3f}")
        r_el = ridge_r2(X, S[:, :4]).mean()            # element one-hot, mean over the four classes
        r_nb = ridge_r2(X, S[:, 4:5])[0]               # neighbour count
        # static fraction: variance of the features explained by the static descriptors
        static_frac = pooled_r2(S, X)
        for n, r in (("static: element (mean over 4 classes)", r_el), ("static: bonded neighbour count", r_nb),
                     ("static fraction of the feature vector", static_frac)):
            rows.append({"encoder": label, "target": n, "r2": float(r)})
            print(f"{label:10s} {n:55s} R^2 = {r:.3f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["encoder", "target", "r2"])
        w.writeheader(); w.writerows(rows)
    print("written", args.out)


if __name__ == "__main__":
    main()
