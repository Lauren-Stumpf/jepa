"""
Do an encoder's features depend on direction, and can directions be read out of them? (E19)

Two measurements on the same frames and encoders as eval/probe_compare.py (E18), whose
quantities were all unchanged by rotating the frame and so could not see direction.

1. Rotation sensitivity. Each frame is encoded as stored and again after a random rotation of
   the whole frame (positions, momenta and cell together, with the same routine the pretraining
   uses for its augmentation, rotations only). Reported: mean over atoms of
   |h(rotated) - h(original)|^2 divided by the spread of the features across atoms of one frame,
   for all atoms and within each element (the all-atom spread includes the difference between
   elements). Scale: 0 means the features ignore direction; about 2 means the rotated features
   are unrelated to the original ones. A network that is not built to be rotation-symmetric
   changes under rotation even when untrained, so values are read AGAINST the untrained encoder
   and the encoder trained with labels, not against a fixed threshold. Self-check: the same
   frames encoded twice without rotation must agree (the script stops otherwise).

2. Direction probes. Ridge probes as in eval/probe_compare.py (per element, split by run, at
   most --max_train_rows training atoms, strength chosen on an inner split) for vector quantities
   in the frame of the simulation box, each scored as the mean R^2 of its three components:
     direction to the nearest neighbour      unit vector            (geometry: informative)
     momentum change over the stride         mass-scaled vector     (what the task predicts: informative)
     momentum direction, momentum            encoder inputs         (reference)
     displacement over the stride            mass-scaled vector     (close to momentum x time, so it
                                                                     mostly re-reads the input momentum)
   Each is scored twice on the held-out runs: on the frames as stored, and on the same frames
   after a random rotation (rotated features against rotated targets, probe still fitted on
   stored frames). A real code for direction keeps its score on rotated frames; a probe that
   has only memorised what each atom's surroundings look like in the stored orientation loses it.
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

VECTORS = ["direction to nearest neighbour", "momentum change over the stride", "momentum direction",
           "momentum (mass-scaled)", "displacement over the stride"]


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
    return np.stack([unit, dp / sm, p_dir, p_ / sm, dq * sm], 1)


def momentum_of(system):
    return system.get_data("momentum").block().values[:, :, 0]


def encode_frames(model, frames, device, rotate_seed=None):
    """(per-frame feature arrays, frames as encoded). With rotate_seed, each frame and its future
    are first turned by one random rotation, and the returned frames are the rotated ones."""
    nl = model.requested_neighbor_lists()
    feats, used = [], []
    if rotate_seed is not None:
        torch.manual_seed(rotate_seed)
    with torch.no_grad():
        for b in range(0, len(frames), 32):
            chunk = frames[b:b + 32]
            now = [make_system(t, q, p, c, pb, torch.float64, device) for t, q, p, _, _, c, pb in chunk]
            if rotate_seed is not None:
                fut = [make_system(t, qf, pf, c, pb, torch.float64, device) for t, _, _, qf, pf, c, pb in chunk]
                ops = random_transformations(len(chunk), 0, device=torch.device(device), dtype=torch.float64,
                                             add_inversions=False)
                now, fut = ops.transform_systems(now), ops.transform_systems(fut)
                for f, s, sf in zip(chunk, now, fut):
                    used.append((f[0], s.positions.cpu(), momentum_of(s).cpu(), sf.positions.cpu(),
                                 momentum_of(sf).cpu(), s.cell.cpu(), f[6]))
            else:
                used.extend(chunk)
            # encode in single precision, as everywhere else
            enc_in = [make_system(u[0], u[1], u[2], u[5], u[6], torch.float32, device) for u in used[-len(chunk):]]
            enc_in = [get_system_with_neighbor_lists(s, nl) for s in enc_in]
            h = encode(model, enc_in).cpu().numpy().astype(np.float64)
            k = 0
            for f in chunk:
                feats.append(h[k:k + len(f[0])])
                k += len(f[0])
            assert k == len(h)
    return feats, used


def relative_change(h0, h1, types=None, z=None):
    """Mean squared change divided by the spread across atoms of one frame; with z, for atoms of
    element z against the spread among atoms of that element."""
    num, den = [], []
    for k, (a, b) in enumerate(zip(h0, h1)):
        if z is not None:
            m = types[k] == z
            if m.sum() < 2:
                continue
            a, b = a[m], b[m]
        num.append(((b - a) ** 2).sum(1).mean())
        den.append(((a - a.mean(0)) ** 2).sum(1).mean())
    return float(np.mean(num) / np.mean(den))


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

    types = [f[0].numpy() for f in frames]
    Z = np.concatenate(types)
    atom_group = np.concatenate([np.full(len(t), g) for t, g in zip(types, groups)])
    is_test = np.isin(atom_group, list(test_groups))
    V = np.concatenate([vector_targets(*f, masses, args.cutoff) for f in frames])     # atoms x vectors x 3
    noise = np.random.default_rng(args.seed + 2).standard_normal((len(Z), 3))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    elements = sorted(np.unique(Z).tolist())

    def vector_score(X, Y3, X_test=None, Y3_test=None):
        """Mean over the three components of the atom-weighted mean over elements of R^2 on the
        held-out runs; with X_test, the held-out rows are taken from X_test / Y3_test instead."""
        Xt = X if X_test is None else X_test
        Yt = Y3 if Y3_test is None else Y3_test
        comps, by_el = [], {str(z): [] for z in elements}
        for c in range(3):
            by, w = {}, {}
            for z in elements:
                tr, te = (Z == z) & ~is_test, (Z == z) & is_test
                by[z], w[z] = pc.ridge_fit_score(X[tr], Y3[tr, c], Xt[te], Yt[te, c], None)
                by_el[str(z)].append(by[z])
            good = [k for k in by if np.isfinite(by[k])]
            comps.append(sum(by[k] * w[k] for k in good) / sum(w[k] for k in good) if good else float("nan"))
        return float(np.mean(comps)), {k: float(np.mean(v)) for k, v in by_el.items()}

    res = {"file": args.xyz, "frames": len(frames), "elements": elements, "max_train_rows": args.max_train_rows,
           "vectors": VECTORS, "encoders": {}}
    V_rot = None
    for spec in args.encoders:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        model = load_encoder(kind, path, hypers, info, masses, timestep, device)
        h0, _ = encode_frames(model, frames, device)
        h0b, _ = encode_frames(model, frames, device)
        again = relative_change(h0, h0b)
        assert again < 1e-6, f"{name}: the same frames encoded twice differ ({again}); the measurement is not repeatable"
        h_rot, rot_frames = encode_frames(model, frames, device, rotate_seed=args.seed + 100)
        h_rot2, _ = encode_frames(model, frames, device, rotate_seed=args.seed + 101)
        if V_rot is None:      # the rotations depend only on the seed, so the rotated targets are shared by all encoders
            V_rot = np.concatenate([vector_targets(*f, masses, args.cutoff) for f in rot_frames])
        X, Xr = np.concatenate(h0), np.concatenate(h_rot)
        enc = {"kind": kind, "path": path, "repeat_change": again,
               "change_under_rotation": {"all": float(np.mean([relative_change(h0, h_rot), relative_change(h0, h_rot2)])),
                                         **{f"element_{z}": float(np.mean([relative_change(h0, h_rot, types, z),
                                                                           relative_change(h0, h_rot2, types, z)]))
                                            for z in elements}},
               "vectors": {}}
        for vi, v in enumerate(VECTORS):
            stored, stored_el = vector_score(X, V[:, vi, :])
            rotated, rotated_el = vector_score(X, V[:, vi, :], Xr, V_rot[:, vi, :])
            enc["vectors"][v] = {"stored_frames": stored, "rotated_frames": rotated,
                                 "stored_by_element": stored_el, "rotated_by_element": rotated_el}
        enc["random_vector_control"] = vector_score(X, noise)[0]
        res["encoders"][name] = enc
        cr = enc["change_under_rotation"]
        print(f"{name:12s} change under rotation: all {cr['all']:.2f} " + " ".join(f"el{z} {cr[f'element_{z}']:.2f}" for z in elements)
              + f" | control {enc['random_vector_control']:+.3f} | "
              + " | ".join(f"{v[:22]} {enc['vectors'][v]['stored_frames']:.2f}/{enc['vectors'][v]['rotated_frames']:.2f}" for v in VECTORS)
              + "   (stored/rotated)", flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
