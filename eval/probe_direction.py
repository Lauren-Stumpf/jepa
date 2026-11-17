"""
Do an encoder's features depend on direction, and can directions be read out of them? (E19)

Two measurements on the same frames and encoders as eval/probe_compare.py (E18), whose
quantities were all unchanged by rotating the frame and so could not see direction.

1. Rotation sensitivity. Each frame is encoded as stored and again after a random rotation of
   the whole frame (positions, momenta and cell together, with the same routine the pretraining
   uses for its augmentation, rotations only). Reported: mean over atoms of
   |h(rotated) - h(original)|^2 divided by the spread of the features across the atoms of the
   frame, the same normalisation as the "change over the gap" in E08. 0 means the features
   ignore direction; values near or above 1 mean a rotated atom looks as different from itself
   as from another atom. As a self-check, the same frame encoded twice without rotation must
   give exactly 0.

2. Direction probes. Ridge probes exactly as in eval/probe_compare.py (per element, split by
   run, at most --max_train_rows training atoms, strength chosen on an inner split) for vector
   quantities in the frame of the simulation box, each scored as the mean R^2 of its three
   components:
     direction to the nearest neighbour      unit vector
     momentum direction                      unit vector of the current momentum (an encoder input)
     momentum (mass-scaled)                  vector, an encoder input
     displacement over the stride            mass-scaled vector (what the task predicts)
     momentum change over the stride         mass-scaled vector (what the task predicts)
   Control that must fail: a random unit vector per atom.

  python eval/probe_direction.py configs/flashmd_water16_scratch.yaml data/water/pairs/16fs/val.xyz \
      --every 2 --group_frames 80 --encoders untrained=untrained: plain=pretrained:<ckpt> --out results/eval/direction_water.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from metatomic.torch.o3 import random_transformations
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eval.probe_compare as pc  # noqa: E402
from eval.feature_change_vs_jump import load_encoder  # noqa: E402
from train.pretrain_jepa import PairData, encode, load_config, make_system  # noqa: E402

VECTORS = ["direction to nearest neighbour", "momentum direction", "momentum (mass-scaled)",
           "displacement over the stride", "momentum change over the stride"]


def vector_targets(t, q, p, qf, pf, cell, pbc, masses, cutoff):
    """[n_atoms, len(VECTORS), 3] array for one frame (nan where undefined)."""
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    z = t.numpy()
    a = Atoms(numbers=z, positions=q.numpy(), cell=cell.numpy(), pbc=pbc.numpy())
    n = len(a)
    sm = np.sqrt(np.array([masses[int(x)] for x in z]))[:, None]
    p_, dq, dp = p.numpy(), (qf - q).numpy(), (pf - p).numpy()
    i, j, d, D = neighbor_list("ijdD", a, cutoff)
    nearest = np.full(n, np.inf)
    unit = np.full((n, 3), np.nan)
    for a_i, dist, vec in zip(i.tolist(), d.tolist(), D):
        if dist < nearest[a_i]:
            nearest[a_i] = dist
            unit[a_i] = vec / dist
    pn = np.linalg.norm(p_, axis=1, keepdims=True)
    p_dir = np.where(pn > 1e-12, p_ / np.maximum(pn, 1e-12), np.nan)
    return np.stack([unit, p_dir, p_ / sm, dq * sm, dp / sm], 1)


def encode_frames(model, frames, device, rotate_seed=None):
    """Per-frame feature arrays; with rotate_seed, each frame is first rotated at random."""
    nl = model.requested_neighbor_lists()
    out = []
    if rotate_seed is not None:
        torch.manual_seed(rotate_seed)
    with torch.no_grad():
        for b in range(0, len(frames), 32):
            chunk = frames[b:b + 32]
            systems = [make_system(t, q, p, c, pb, torch.float32, device) for t, q, p, _, _, c, pb in chunk]
            if rotate_seed is not None:
                ops = random_transformations(len(chunk), 0, device=torch.device(device), dtype=torch.float32,
                                             add_inversions=False)
                systems = ops.transform_systems(systems)
            systems = [get_system_with_neighbor_lists(s, nl) for s in systems]
            h = encode(model, systems).cpu().numpy().astype(np.float64)
            k = 0
            for f in chunk:
                out.append(h[k:k + len(f[0])])
                k += len(f[0])
            assert k == len(h)
    return out


def relative_change(h0, h1):
    spread = float(np.mean([((h - h.mean(0)) ** 2).sum(1).mean() for h in h0]))
    return float(np.mean([((b - a) ** 2).sum(1).mean() for a, b in zip(h0, h1)])) / spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("xyz")
    ap.add_argument("--encoders", nargs="+", required=True, help="name=kind:path")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--group_frames", type=int, required=True)
    ap.add_argument("--cutoff", type=float, default=4.5)
    ap.add_argument("--max_train_rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pc.MAX_TRAIN_ROWS = args.max_train_rows
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, hypers, masses, timestep, info = load_config(args.config)
    data = PairData(args.xyz, masses)
    assert len(data) % args.group_frames == 0, \
        f"{len(data)} frames is not a whole number of groups of {args.group_frames}: wrong --group_frames?"
    keep = list(range(0, len(data), args.every))
    frames = [data.frames[k] for k in keep]
    groups = np.array([k // args.group_frames for k in keep])
    uniq = np.unique(groups)
    # the same held-out groups as eval/probe_compare.py (same seed, same rule)
    rng = np.random.default_rng(args.seed)
    test_groups = set(rng.permutation(uniq)[: max(1, int(round(0.3 * len(uniq))))].tolist())
    print(f"{len(frames)} frames in {len(uniq)} groups ({len(test_groups)} held out)", flush=True)

    Z = np.concatenate([f[0].numpy() for f in frames])
    atom_group = np.concatenate([np.full(len(f[0]), g) for f, g in zip(frames, groups)])
    is_test = np.isin(atom_group, list(test_groups))
    V = np.concatenate([vector_targets(*f, masses, args.cutoff) for f in frames])     # atoms x vectors x 3
    noise = np.random.default_rng(args.seed + 2).standard_normal((len(Z), 3))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    elements = sorted(np.unique(Z).tolist())

    def vector_score(X, Y3):
        """Mean over the three components of the atom-weighted mean over elements of R^2."""
        comps = []
        for c in range(3):
            by, w = {}, {}
            for z in elements:
                tr, te = (Z == z) & ~is_test, (Z == z) & is_test
                by[z], w[z] = pc.ridge_fit_score(X[tr], Y3[tr, c], X[te], Y3[te, c], None)
            good = [k for k in by if np.isfinite(by[k])]
            comps.append(sum(by[k] * w[k] for k in good) / sum(w[k] for k in good) if good else float("nan"))
        return float(np.mean(comps))

    res = {"file": args.xyz, "frames": len(frames), "elements": elements, "max_train_rows": args.max_train_rows,
           "vectors": VECTORS, "encoders": {}}
    for spec in args.encoders:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        model = load_encoder(kind, path, hypers, info, masses, timestep, device)
        h0 = encode_frames(model, frames, device)
        again = relative_change(h0, encode_frames(model, frames, device))
        assert again < 1e-8, f"{name}: the same frames encoded twice differ ({again}); the measurement is not repeatable"
        rot = [relative_change(h0, encode_frames(model, frames, device, rotate_seed=args.seed + 100 + r)) for r in range(2)]
        X = np.concatenate(h0)
        enc = {"kind": kind, "path": path, "change_under_rotation": float(np.mean(rot)), "change_under_rotation_draws": rot,
               "vectors": {v: vector_score(X, V[:, vi, :]) for vi, v in enumerate(VECTORS)},
               "random_vector_control": vector_score(X, noise)}
        res["encoders"][name] = enc
        print(f"{name:12s} change under rotation {enc['change_under_rotation']:.3f} | control {enc['random_vector_control']:+.3f} | "
              + " | ".join(f"{v[:26]} {enc['vectors'][v]:.2f}" for v in VECTORS), flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
