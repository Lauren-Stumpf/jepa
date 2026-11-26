"""
Scores of a water rollout against exact dynamics from the same starts (E28).

Reads files written by eval/rollout_water.py (a model's rollout) and by
eval/exact_reference_water.py (the exact answer), which share one layout, and reports for each:

  energy     for rollouts made with --conserve_energy: the size of the correction, and from it the
             model's own energy error at each step before the correction,
               E' - E = K_after * (1 / alpha^2 - 1)
             (K_after: kinetic energy after the correction; only when there is no thermostat,
             because with one the saved temperature is taken after the second thermostat half step).
  elements   temperature of hydrogens and of oxygens, and hydrogen minus oxygen, in three windows
             (0 to 2 ps, 2 to 50 ps, 50 to 100 ps): mean over runs; the standard error over runs
             is given for hydrogen minus oxygen only.
  shape      O-H bond length and H-O-H angle of the molecules, mean and spread, 50 to 100 ps.
  structure  how often two atoms of different molecules are found at each distance (the radial
             distribution function), for O-O, O-H and H-H, 50 to 100 ps, distances up to half the
             box. Normalised so that 1 means "as often as if the atoms were placed at random".
             Score: mean absolute difference from the exact curve over distances from 1.5 A to half
             the box. Noise level of that score: the same difference between two halves of the
             exact runs (14 runs against 14), divided by the square root of 2 to match a
             comparison of 28 runs against 28. One split only.
  movement   mean square displacement of the oxygens (after removing the motion of the whole box)
             at lags of about 1, 2, 5, 10 and 20 ps, over the whole run, and the diffusion
             coefficient from the slope between the 5 and 20 ps lags, per run. Whole run, not
             50 to 100 ps. A thermostat lowers diffusion by itself, so a low value in a
             thermostat arm is not evidence of a model error.

Tested on: whatever the files hold. For E28 that is the 28 water test runs (the same liquid as
training, runs never seen in training).

  python eval/rollout_scores.py --exact results/eval/e28/exact_reference.npz \
      --rollouts "no correction=results/eval/e22/rollout_primary.npz" "energy correction=results/eval/e28/rollout_energy.npz" \
      --out results/eval/e28/scores.json
"""

import argparse
import json
import os

import numpy as np

KB = 8.617333262e-5          # eV / K
DR = 0.05                    # A, width of a distance bin
PAIRS = [("O-O", 8, 8), ("O-H", 8, 1), ("H-H", 1, 1)]
LAGS = [60, 120, 310, 620, 1250]   # 16 fs steps: 0.96, 1.92, 4.96, 9.92, 20 ps
WINDOWS = [("0 to 2 ps", 0, 125), ("2 to 50 ps", 126, 3125), ("50 to 100 ps", 3126, 6250)]


def load(path):
    r = dict(np.load(path, allow_pickle=True))
    z = r["numbers"]
    assert len(z) % 3 == 0 and (z[0::3] == 8).all() and (z[1::3] == 1).all() and (z[2::3] == 1).all(), "not O H H order"
    return r


def frames_in(r, lo, hi, every=10):
    """Indices into the saved states for steps lo..hi. From step 126 on only multiples of
    `every` are evenly spaced in the file, so only those are used there."""
    return [i for i, k in enumerate(r["saved_steps"]) if lo <= k <= hi and (k <= 125 or k % every == 0)]


def element_temperatures(r):
    z, m = r["numbers"], r["masses"]
    out = {}
    for name, lo, hi in WINDOWS:
        idx = frames_in(r, lo, hi)
        if not idx:
            continue
        ke = (r["p"][idx] ** 2).sum(-1) / (2 * m)                 # (frames, runs, atoms)
        th = np.nanmean(2 * ke[..., z == 1].mean(-1) / (3 * KB), 0)   # per run
        to = np.nanmean(2 * ke[..., z == 8].mean(-1) / (3 * KB), 0)
        d = th - to
        n = int(np.isfinite(d).sum())
        out[name] = {"hydrogen_K": float(np.nanmean(th)), "oxygen_K": float(np.nanmean(to)),
                     "hydrogen_minus_oxygen_K": float(np.nanmean(d)),
                     "hydrogen_minus_oxygen_standard_error_K": float(np.nanstd(d, ddof=1) / np.sqrt(n)), "runs": n}
    return out


def energy(r):
    a = r.get("alpha")
    if a is None or not bool(r.get("conserve_energy", False)) or not np.isfinite(a).any():
        return None
    out = {"mean_alpha_minus_1": float(np.nanmean(a - 1)), "mean_size_of_alpha_minus_1": float(np.nanmean(np.abs(a - 1)))}
    if float(r.get("langevin_fs", 0.0)) == 0.0:
        n_atoms = len(r["numbers"]); n_mol = n_atoms // 3
        k_after = 1.5 * n_atoms * KB * r["temperature"][1:]       # eV per box, after the correction
        de = k_after * (1.0 / a ** 2 - 1.0) / n_mol * 1e3          # meV per molecule, the model's own energy change at each step
        out.update({"model_energy_error_per_step_meV_per_molecule_mean": float(np.nanmean(de)),
                    "model_energy_error_per_step_meV_per_molecule_root_mean_square": float(np.sqrt(np.nanmean(de ** 2))),
                    "sum_over_the_run_meV_per_molecule": float(np.nanmean(np.nansum(de, 0)))})
    return out


def exact_energy(r):
    if "energy_drift_meV_per_molecule" not in r:
        return None
    d = r["energy_drift_meV_per_molecule"]
    return {"total_energy_end_minus_start_meV_per_molecule_mean": float(d.mean()), "largest_size": float(np.abs(d).max())}


def min_image(d, L):
    return d - L * np.round(d / L)


def shape(r):
    """O-H bond length (A) and H-O-H angle (degrees), 50 to 100 ps."""
    idx = frames_in(r, 3126, 6250)
    q = r["q"][idx]; L = r["cell"]
    o, h1, h2 = q[..., 0::3, :], q[..., 1::3, :], q[..., 2::3, :]
    v1, v2 = min_image(h1 - o, L), min_image(h2 - o, L)
    b = np.concatenate([np.linalg.norm(v1, axis=-1).ravel(), np.linalg.norm(v2, axis=-1).ravel()])
    cosang = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1))).ravel()
    return {"bond_mean_A": float(np.nanmean(b)), "bond_spread_A": float(np.nanstd(b)),
            "angle_mean_deg": float(np.nanmean(ang)), "angle_spread_deg": float(np.nanstd(ang))}


def rdf(r, runs=None):
    """Radial distribution functions between atoms of different molecules, 50 to 100 ps.
    Returns the bin centres and {pair: g}. runs: which runs to pool (default all)."""
    idx = frames_in(r, 3126, 6250)
    z, L = r["numbers"], r["cell"]
    assert np.allclose(L, L[0]), "box is not cubic"
    L0 = float(L[0]); V = L0 ** 3
    edges = np.arange(0.0, L0 / 2 + 1e-9, DR)
    centres = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    mol = np.arange(len(z)) // 3
    runs = list(range(r["q"].shape[1])) if runs is None else list(runs)
    out = {}
    for name, za, zb in PAIRS:
        ia, ib = np.where(z == za)[0], np.where(z == zb)[0]
        keep = mol[ia][:, None] != mol[ib][None, :]               # different molecules only
        if za == zb:
            keep &= ia[:, None] < ib[None, :]                     # each pair once
        n_pairs = int(keep.sum())
        counts = np.zeros(len(centres)); n_frames = 0
        for b in runs:
            qb = r["q"][idx, b]                                    # (frames, atoms, 3)
            good = np.isfinite(qb).all((1, 2))
            for f in np.where(good)[0]:
                d = min_image(qb[f, ia][:, None, :] - qb[f, ib][None, :, :], L)
                dist = np.sqrt((d ** 2).sum(-1))[keep]
                counts += np.histogram(dist, bins=edges)[0]
                n_frames += 1
        out[name] = counts / (n_frames * n_pairs * shell / V)
    return centres, out


def structure_score(g, g_ref, centres):
    sel = centres >= 1.5
    return {name: float(np.abs(g[name][sel] - g_ref[name][sel]).mean()) for name, _, _ in PAIRS}


def first_peak(centres, g, lo=2.4, hi=3.4):
    sel = (centres >= lo) & (centres <= hi)
    i = int(np.argmax(g[sel]))
    return {"position_A": float(centres[sel][i]), "height": float(g[sel][i])}


def movement(r):
    """Mean square displacement of the oxygens (A^2) at each lag, per run, and the diffusion
    coefficient (A^2 per ps). Positions must be unwrapped (checked)."""
    ss = r["saved_steps"]; m = r["masses"]; z = r["numbers"]; L0 = float(r["cell"][0])
    idx = [i for i, k in enumerate(ss) if k % 10 == 0]
    steps = ss[idx]
    assert (np.diff(steps) == 10).all(), "saved states are not every 10 steps"
    q = r["q"][idx]                                               # (frames, runs, atoms, 3)
    jump = np.nanmax(np.abs(np.diff(q, axis=0)))
    assert not jump > L0 / 4, f"an atom moves {jump:.2f} A in 160 fs: positions look wrapped"
    com = (q * m[None, None, :, None]).sum(2) / m.sum()           # (frames, runs, 3)
    x = (q - com[:, :, None, :])[:, :, z == 8]                    # oxygens, box motion removed
    msd = {}
    for lag in LAGS:
        n = lag // 10
        if n >= len(x):
            continue
        d = x[n:] - x[:-n]
        msd[lag] = np.nanmean((d ** 2).sum(-1), axis=(0, 2))      # per run
    out = {"lag_ps": [lag * 0.016 for lag in msd],
           "mean_square_displacement_A2": [float(np.nanmean(v)) for v in msd.values()]}
    if 310 in msd and 1250 in msd:
        D = (msd[1250] - msd[310]) / (6.0 * (1250 - 310) * 0.016)  # A^2 / ps, per run
        n = int(np.isfinite(D).sum())
        out.update({"diffusion_A2_per_ps": float(np.nanmean(D)),
                    "diffusion_standard_error": float(np.nanstd(D, ddof=1) / np.sqrt(n)), "runs": n})
    return out


def score(r, centres_ref=None, g_ref=None):
    centres, g = rdf(r)
    S = {"runs": int(r["q"].shape[1]), "runs_reaching_the_end": int((r["stopped_at"] < 0).sum()),
         "energy": energy(r), "exact_energy": exact_energy(r), "elements": element_temperatures(r),
         "shape": shape(r), "movement": movement(r),
         "structure": {"first_O-O_peak": first_peak(centres, g["O-O"]),
                       "distance_A": [round(float(c), 3) for c in centres],
                       "curves": {k: [round(float(v), 4) for v in g[k]] for k in g}}}
    if g_ref is not None:
        assert np.allclose(centres, centres_ref)
        S["structure"]["mean_absolute_difference_from_exact"] = structure_score(g, g_ref, centres)
    return S, centres, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", required=True)
    ap.add_argument("--rollouts", nargs="+", required=True, help="name=path, one per rollout")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ex = load(args.exact)
    assert bool(ex.get("check_reproduces_data", False)) and bool(ex.get("check_conserves_energy", False)), \
        "the exact reference did not pass its own checks"
    S = {"exact": None, "rollouts": {}}
    S["exact"], centres, g_ex = score(ex)
    B = ex["q"].shape[1]
    _, ga = rdf(ex, range(0, B // 2)); _, gb = rdf(ex, range(B // 2, B))
    # 14 runs against 14 is noisier than 28 against 28 by the square root of 2: divide so the level matches the scores
    S["structure_noise_level"] = {k: v / np.sqrt(2.0) for k, v in structure_score(ga, gb, centres).items()}
    for spec in args.rollouts:
        name, path = spec.split("=", 1)
        r = load(path)
        assert (r["runs"] == ex["runs"]).all() and np.array_equal(r["ref_q"][:, 0], ex["ref_q"][:, 0]) \
            and np.array_equal(r["ref_p"][:, 0], ex["ref_p"][:, 0]), f"{name}: not the same starts"
        S["rollouts"][name], _, _ = score(r, centres, g_ex)
        S["rollouts"][name]["path"] = path
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(S, open(args.out, "w"), indent=1)

    def line(name, s):
        el = s["elements"].get("50 to 100 ps", {})
        st = s["structure"]
        diff = st.get("mean_absolute_difference_from_exact", {})
        mv = s["movement"]
        print(f"{name:34s} H {el.get('hydrogen_K', float('nan')):6.1f} K  O {el.get('oxygen_K', float('nan')):6.1f} K  "
              f"H-O {el.get('hydrogen_minus_oxygen_K', float('nan')):+6.1f} +- {el.get('hydrogen_minus_oxygen_standard_error_K', float('nan')):.1f}  "
              f"bond {s['shape']['bond_mean_A']:.4f} +- {s['shape']['bond_spread_A']:.4f} A  angle {s['shape']['angle_mean_deg']:.2f} +- {s['shape']['angle_spread_deg']:.2f}  "
              f"O-O peak {st['first_O-O_peak']['height']:.2f} at {st['first_O-O_peak']['position_A']:.2f} A  "
              f"structure difference {', '.join(f'{k} {v:.4f}' for k, v in diff.items()) or '(reference)'}  "
              f"diffusion {mv.get('diffusion_A2_per_ps', float('nan')):.4f} +- {mv.get('diffusion_standard_error', float('nan')):.4f} A^2/ps")
    line("exact", S["exact"])
    print(f"{'noise level of structure difference (two halves of the exact runs, divided by root 2)':34s} {', '.join(f'{k} {v:.4f}' for k, v in S['structure_noise_level'].items())}")
    for name, s in S["rollouts"].items():
        line(name, s)
        if s["energy"]:
            print(f"{'':34s} energy: {s['energy']}")
    print(f"exact energy: {S['exact']['exact_energy']}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
