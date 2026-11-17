"""
What can be read out of an encoder's frozen per-atom features by a linear map? The same list of
quantities for organic and for water encoders (E18).

Quantities, per atom (none depends on how the frame is rotated):
  what the atom is     element (probe across all atoms); number of bonded neighbours
  current geometry     mean bond length to bonded neighbours; distance to the nearest non-bonded
                       atom (capped at --cutoff); number of atoms within 3 A
  current motion       size of the mass-scaled momentum; mass-scaled momentum along the unit vector
                       to the nearest neighbour (signed)
  the frame            kinetic temperature of the whole frame (same value for all its atoms)
  what the task needs  size of the mass-scaled displacement over the stride; that displacement along
                       the unit vector to the nearest neighbour (signed); size of the mass-scaled
                       momentum change over the stride

Probe: ridge regression from the standardised 640 features (plus a constant) to each quantity,
fitted SEPARATELY for each element so that knowing the element cannot raise the score, on the
training groups and scored on held-out groups. Groups are blocks of --group_frames consecutive
frames of the file (organic: 500 = one molecule; water: 80 = one run with its reversed copies);
70% of the groups train the probe, 30% score it, so no frame, molecule or run is in both.
Score: R^2 = 1 - (squared error) / (variance of the quantity among held-out atoms of that
element). Reported per element and as the average weighted by the number of held-out atoms.
A quantity that is constant within an element (for example bonded neighbours of water hydrogens)
has no variance to explain and is reported as "nan" for that element and left out of the average.

Control that must fail: the same probe with the feature rows shuffled among atoms of the same
element in the training groups (features paired with the wrong atoms) should score about zero.

Encoders: name=kind:path with kind untrained | baseline | pretrained (loading as in
eval/feature_change_vs_jump.py).

  python eval/probe_compare.py configs/flashmd_water16_scratch.yaml data/water/pairs/16fs/val.xyz \
      --every 2 --group_frames 80 --encoders untrained=untrained: plain=pretrained:<ckpt> --out results/eval/probe_water.json
"""

import argparse
import json
import os
import sys

import ase.units
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.feature_change_vs_jump import load_encoder  # noqa: E402
from train.pretrain_jepa import PairData, encode, load_config, make_batch  # noqa: E402

TARGETS = ["bonded neighbours", "mean bond length", "nearest non-bonded distance", "atoms within 3 A",
           "momentum size", "momentum along nearest neighbour", "frame temperature",
           "displacement size", "displacement along nearest neighbour", "momentum change size"]


def atom_targets(t, q, p, qf, pf, cell, pbc, masses, cutoff):
    """[n_atoms, len(TARGETS)] array for one frame."""
    from ase import Atoms
    from ase.neighborlist import natural_cutoffs, neighbor_list
    z = t.numpy()
    a = Atoms(numbers=z, positions=q.numpy(), cell=cell.numpy(), pbc=pbc.numpy())
    n = len(a)
    m = np.array([masses[int(x)] for x in z])
    sm = np.sqrt(m)
    p_, dq, dp = p.numpy(), (qf - q).numpy(), (pf - p).numpy()

    bi, bj, bd = neighbor_list("ijd", a, natural_cutoffs(a, mult=1.25))
    n_bond = np.bincount(bi, minlength=n).astype(float)
    bond_sum = np.bincount(bi, weights=bd, minlength=n)
    mean_bond = np.where(n_bond > 0, bond_sum / np.maximum(n_bond, 1), np.nan)

    i, j, d, D = neighbor_list("ijdD", a, cutoff)
    bonded = set(zip(bi.tolist(), bj.tolist()))
    nearest = np.full(n, np.inf)
    unit = np.zeros((n, 3))
    nonbond = np.full(n, cutoff)
    within3 = np.zeros(n)
    for a_i, a_j, dist, vec in zip(i, j, d, D):
        if dist < nearest[a_i]:
            nearest[a_i] = dist
            unit[a_i] = vec / dist
        if (a_i, a_j) not in bonded and dist < nonbond[a_i]:
            nonbond[a_i] = dist
        if dist < 3.0:
            within3[a_i] += 1
    has_nb = np.isfinite(nearest)

    ke = float(((p_ ** 2).sum(1) / (2 * m)).sum())
    temp = 2 * ke / (3 * n * ase.units.kB)
    out = np.stack([
        n_bond, mean_bond, nonbond, within3,
        np.linalg.norm(p_, axis=1) / sm,
        np.where(has_nb, (p_ * unit).sum(1) / sm, np.nan),
        np.full(n, temp),
        np.linalg.norm(dq, axis=1) * sm,
        np.where(has_nb, (dq * unit).sum(1) * sm, np.nan),
        np.linalg.norm(dp, axis=1) / sm,
    ], 1)
    return out


def ridge_fit_score(Xtr, Ytr, Xte, Yte, lam):
    """R^2 on the held-out rows of a ridge fit; nan if the quantity has no variance or too few rows."""
    ok_tr, ok_te = np.isfinite(Ytr), np.isfinite(Yte)
    if ok_tr.sum() < 50 or ok_te.sum() < 20:
        return float("nan"), int(ok_te.sum())
    Xtr, Ytr, Xte, Yte = Xtr[ok_tr], Ytr[ok_tr], Xte[ok_te], Yte[ok_te]
    tot = ((Yte - Yte.mean()) ** 2).sum()
    if Ytr.std() < 1e-9 or tot < 1e-12:
        return float("nan"), int(ok_te.sum())
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    ymu = Ytr.mean()
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - ymu))
    res = ((Yte - (B @ W + ymu)) ** 2).sum()
    return float(1 - res / tot), int(ok_te.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("xyz")
    ap.add_argument("--encoders", nargs="+", required=True, help="name=kind:path")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--group_frames", type=int, required=True)
    ap.add_argument("--cutoff", type=float, default=4.5)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, hypers, masses, timestep, info = load_config(args.config)
    data = PairData(args.xyz, masses)
    keep = list(range(0, len(data), args.every))
    frames = [data.frames[k] for k in keep]
    groups = np.array([k // args.group_frames for k in keep])
    uniq = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    test_groups = set(rng.permutation(uniq)[: max(1, int(round(0.3 * len(uniq))))].tolist())
    print(f"{len(frames)} frames in {len(uniq)} groups ({len(test_groups)} held out)", flush=True)

    Z = np.concatenate([f[0].numpy() for f in frames])
    atom_group = np.concatenate([np.full(len(f[0]), g) for f, g in zip(frames, groups)])
    is_test = np.isin(atom_group, list(test_groups))
    Y = np.concatenate([atom_targets(*f, masses, args.cutoff) for f in frames])
    elements = sorted(np.unique(Z).tolist())
    onehot = (Z[:, None] == np.array(elements)[None, :]).astype(float)

    res = {"file": args.xyz, "frames": len(frames), "groups": int(len(uniq)), "held_out_groups": len(test_groups),
           "atoms": int(len(Z)), "held_out_atoms": int(is_test.sum()), "elements": elements, "targets": TARGETS, "encoders": {}}
    for spec in args.encoders:
        name, rest = spec.split("=", 1)
        kind, path = rest.split(":", 1)
        model = load_encoder(kind, path, hypers, info, masses, timestep, device)
        X = []
        with torch.no_grad():
            for b in range(0, len(frames), 32):
                now, _ = make_batch(frames[b:b + 32], model, torch.float32, device, augment=False)
                X.append(encode(model, now).cpu().numpy())
        X = np.concatenate(X).astype(np.float64)
        assert len(X) == len(Z)
        enc = {"kind": kind, "path": path, "targets": {}}
        # element: one probe across all atoms, mean R^2 over the one-hot columns
        el = [ridge_fit_score(X[~is_test], onehot[~is_test, c], X[is_test], onehot[is_test, c], args.lam)[0]
              for c in range(len(elements))]
        enc["targets"]["element"] = {"mean": float(np.nanmean(el)), "by_element": dict(zip(map(str, elements), el))}
        shuffle_rng = np.random.default_rng(args.seed + 1)
        for ti, tname in enumerate(TARGETS):
            by, w, ctrl = {}, {}, {}
            for z in elements:
                tr, te = (Z == z) & ~is_test, (Z == z) & is_test
                r2, n_te = ridge_fit_score(X[tr], Y[tr, ti], X[te], Y[te, ti], args.lam)
                by[str(z)], w[str(z)] = r2, n_te
                Xs = X[tr][shuffle_rng.permutation(int(tr.sum()))]
                ctrl[str(z)] = ridge_fit_score(Xs, Y[tr, ti], X[te], Y[te, ti], args.lam)[0]
            good = [k for k in by if np.isfinite(by[k])]
            mean = float(sum(by[k] * w[k] for k in good) / sum(w[k] for k in good)) if good else float("nan")
            cgood = [k for k in ctrl if np.isfinite(ctrl[k])]
            cmean = float(sum(ctrl[k] * w[k] for k in cgood) / sum(w[k] for k in cgood)) if cgood else float("nan")
            enc["targets"][tname] = {"mean": mean, "by_element": by, "shuffled_control_mean": cmean}
        res["encoders"][name] = enc
        print(f"{name:12s} element {enc['targets']['element']['mean']:.2f} | "
              + " | ".join(f"{t[:22]} {enc['targets'][t]['mean']:.2f}" for t in TARGETS), flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
