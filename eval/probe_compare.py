"""
What can be read out of an encoder's frozen per-atom features by a linear map? The same list of
quantities for organic and for water encoders (E18).

Quantities, per atom (none depends on how the frame is rotated):
  what the atom is     element (probe across all atoms); number of bonded neighbours
  current geometry     mean bond length to bonded neighbours; distance to the nearest atom that is
                       more than two bonds away (so not a bonded neighbour and not a neighbour's
                       neighbour: for a water hydrogen this is an atom of another molecule;
                       capped at --cutoff); number of atoms within 3 A
  current motion       size of the mass-scaled momentum; mass-scaled momentum along the unit vector
                       to the nearest neighbour (signed)
  the frame            kinetic temperature of the whole frame (same value for all its atoms)
  what the task needs  size of the mass-scaled displacement over the stride; that displacement along
                       the unit vector to the nearest neighbour (signed); size of the mass-scaled
                       momentum change over the stride

Probe: ridge regression (strength chosen per probe on an inner split of its training rows) from
the standardised 640 features (plus a constant) to each quantity,
fitted SEPARATELY for each element so that knowing the element cannot raise the score, on the
training groups and scored on held-out groups. Groups are blocks of --group_frames consecutive
frames of the file, chosen so that a group is one simulation run in both systems (organic: 50 =
one constant-energy run of one molecule; water: 80 = one run with its reversed copies). 70% of
the runs train the probe and 30% score it, so no frame or run is in both; in both systems the
held-out runs are of systems the probe has seen (the same liquid; the same molecules), which
keeps the two like for like. The script stops if the file is not a whole number of groups.
Score: R^2 = 1 - (squared error) / (variance of the quantity among held-out atoms of that
element). Reported per element and as the average weighted by the number of held-out atoms.
A quantity that is constant within an element (for example bonded neighbours of water hydrogens)
has no variance to explain and is reported as "nan" for that element and left out of the average.

Controls: a random number per atom as the quantity must score about zero (it cannot be read out
of anything); and each score is also computed with atoms split at random instead of by run
("atom_split_mean"), which shares frames between fit and score: the gap between the two shows
how much a leak would inflate a score.

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

TARGETS = ["bonded neighbours", "mean bond length", "nearest atom beyond two bonds", "atoms within 3 A",
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
    nbrs = [set() for _ in range(n)]
    for x, y in zip(bi.tolist(), bj.tolist()):
        nbrs[x].add(y)
    close = []                      # per atom: itself, its bonded neighbours and their bonded neighbours
    for x in range(n):
        c = {x} | nbrs[x]
        for y in nbrs[x]:
            c |= nbrs[y]
        close.append(c)
    nearest = np.full(n, np.inf)
    unit = np.zeros((n, 3))
    nonbond = np.full(n, cutoff)
    within3 = np.zeros(n)
    for a_i, a_j, dist, vec in zip(i.tolist(), j.tolist(), d.tolist(), D):
        if dist < nearest[a_i]:
            nearest[a_i] = dist
            unit[a_i] = vec / dist
        if a_j not in close[a_i] and dist < nonbond[a_i]:
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


def mode_share(y):
    """Share of the values equal to the most common one (after rounding): a quantity that is the
    same for more than 98% of the atoms has too little variation for a stable R^2."""
    _, counts = np.unique(np.round(y, 6), return_counts=True)
    return counts.max() / len(y)


MAX_TRAIN_ROWS = 2000   # set from --max_train_rows in main()
LAM_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0)


def ridge_fit_score(Xtr, Ytr, Xte, Yte, lam):
    """R^2 on the held-out rows of a ridge fit; nan if the quantity has no variance or too few rows.
    At most MAX_TRAIN_ROWS training rows are used (a fixed random subset), so that every element in
    both systems is fitted on the same number of rows: with 640 features, a probe fitted on fewer
    rows overfits more and scores lower for that reason alone."""
    ok_tr, ok_te = np.isfinite(Ytr), np.isfinite(Yte)
    if ok_tr.sum() < 50 or ok_te.sum() < 20:
        return float("nan"), int(ok_te.sum())
    Xtr, Ytr, Xte, Yte = Xtr[ok_tr], Ytr[ok_tr], Xte[ok_te], Yte[ok_te]
    if len(Xtr) > MAX_TRAIN_ROWS:
        pick = np.random.default_rng(len(Xtr)).choice(len(Xtr), MAX_TRAIN_ROWS, replace=False)
        Xtr, Ytr = Xtr[pick], Ytr[pick]
    tot = ((Yte - Yte.mean()) ** 2).sum()
    if Ytr.std() < 1e-9 or tot < 1e-12 or mode_share(Ytr) > 0.98 or mode_share(Yte) > 0.98:
        return float("nan"), int(ok_te.sum())
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    B = np.c_[(Xte - mu) / sd, np.ones(len(Xte))]
    ymu = Ytr.mean()

    def solve(Am, ym, strength):
        return np.linalg.solve(Am.T @ Am + strength * len(Am) * np.eye(Am.shape[1]), Am.T @ ym)

    if lam is None:
        # choose the regularisation strength on an inner split of the TRAINING rows only (the
        # held-out rows play no part), so a probe with few rows per feature is not left to overfit
        inner = np.random.default_rng(len(A) + 1).random(len(A)) < 0.8
        best, lam = np.inf, LAM_GRID[0]
        for cand in LAM_GRID:
            err = ((Ytr[~inner] - ymu - A[~inner] @ solve(A[inner], Ytr[inner] - ymu, cand)) ** 2).sum()
            if err < best:
                best, lam = err, cand
    W = solve(A, Ytr - ymu, lam)
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
    ap.add_argument("--lam", type=float, default=None,
                    help="ridge strength; default: chosen per probe from 0.01 to 100 on an inner split of the training rows")
    ap.add_argument("--max_train_rows", type=int, default=2000,
                    help="training rows per probe; use the same value for organic and water")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    global MAX_TRAIN_ROWS
    MAX_TRAIN_ROWS = args.max_train_rows
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, hypers, masses, timestep, info = load_config(args.config)
    data = PairData(args.xyz, masses)
    assert len(data) % args.group_frames == 0, \
        f"{len(data)} frames is not a whole number of groups of {args.group_frames}: wrong --group_frames?"
    keep = list(range(0, len(data), args.every))
    frames = [data.frames[k] for k in keep]
    groups = np.array([k // args.group_frames for k in keep])
    uniq = np.unique(groups)
    rng = np.random.default_rng(args.seed)
    test_groups = set(rng.permutation(uniq)[: max(1, int(round(0.3 * len(uniq))))].tolist())
    print(f"{len(frames)} frames in {len(uniq)} groups of {args.group_frames} file frames "
          f"({len(frames) / len(uniq):.1f} used per group; {len(test_groups)} groups held out)", flush=True)

    Z = np.concatenate([f[0].numpy() for f in frames])
    atom_group = np.concatenate([np.full(len(f[0]), g) for f, g in zip(frames, groups)])
    is_test = np.isin(atom_group, list(test_groups))
    Y = np.concatenate([atom_targets(*f, masses, args.cutoff) for f in frames])
    elements = sorted(np.unique(Z).tolist())
    atom_test = np.random.default_rng(args.seed + 1).random(len(Z)) < 0.3     # atom-level split, for comparison only
    noise = np.random.default_rng(args.seed + 2).standard_normal(len(Z))      # random target, must score about 0
    onehot = (Z[:, None] == np.array(elements)[None, :]).astype(float)

    res = {"file": args.xyz, "frames": len(frames), "groups": int(len(uniq)), "held_out_groups": len(test_groups),
           "atoms": int(len(Z)), "held_out_atoms": int(is_test.sum()), "elements": elements, "targets": TARGETS,
           "max_train_rows": MAX_TRAIN_ROWS,
           "train_atoms_by_element": {str(z): int(((Z == z) & ~is_test).sum()) for z in elements},
           "held_out_atoms_by_element": {str(z): int(((Z == z) & is_test).sum()) for z in elements}, "encoders": {}}
    print("training atoms per element (capped at", MAX_TRAIN_ROWS, "per probe):", res["train_atoms_by_element"], flush=True)
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
        def weighted(by, w):
            good = [k for k in by if np.isfinite(by[k])]
            return float(sum(by[k] * w[k] for k in good) / sum(w[k] for k in good)) if good else float("nan")

        for ti, tname in enumerate(TARGETS):
            by, w, by_atom, w_atom = {}, {}, {}, {}
            for z in elements:
                tr, te = (Z == z) & ~is_test, (Z == z) & is_test
                by[str(z)], w[str(z)] = ridge_fit_score(X[tr], Y[tr, ti], X[te], Y[te, ti], args.lam)
                # the same probe with atoms split at random (frames shared between fit and score):
                # how much higher this is than the group split shows what a leak would add
                tr2, te2 = (Z == z) & ~atom_test, (Z == z) & atom_test
                by_atom[str(z)], w_atom[str(z)] = ridge_fit_score(X[tr2], Y[tr2, ti], X[te2], Y[te2, ti], args.lam)
            enc["targets"][tname] = {"mean": weighted(by, w), "by_element": by,
                                     "atom_split_mean": weighted(by_atom, w_atom)}
        # control that must fail: a random number per atom cannot be read out of anything
        by, w = {}, {}
        for z in elements:
            tr, te = (Z == z) & ~is_test, (Z == z) & is_test
            by[str(z)], w[str(z)] = ridge_fit_score(X[tr], noise[tr], X[te], noise[te], args.lam)
        enc["random_target_control"] = weighted(by, w)
        enc["random_target_control_by_element"] = by
        res["encoders"][name] = enc
        print(f"{name:12s} random-target control {enc['random_target_control']:+.3f} | element {enc['targets']['element']['mean']:.2f} | "
              + " | ".join(f"{t[:22]} {enc['targets'][t]['mean']:.2f}" for t in TARGETS), flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("written", args.out)


if __name__ == "__main__":
    main()
