"""
E22 analysis: every measurement, prediction verdict and control of experiments/E22/card.md
from the outputs of eval/rollout_water.py and eval/rollout_exact.py. Writes summary.json and
figures (PNG) into --out_dir.

Errors are mass-scaled root mean square errors over atoms and components, as the one-step test
error. Momentum error is the primary measure (as in every earlier experiment); position error is
reported beside it.

Definitions (fixed before the results, as on the card):
  A1 doubling time: the first 16 fs step at which the error, averaged over the runs, reaches
     twice its value at step 1. Computed for the model and for the chaos reference (A3).
     Prediction: the model's doubling time is shorter, within the first 0.5 ps (31 steps).
  A2 first 2 ps: per run, the mean over the A2 steps of (error on own outputs - error on true
     states); "larger" if the mean over runs exceeds 3 standard errors. After 2 ps: per run,
     the mean error on own outputs; "larger" if the mean over runs exceeds the model's mean
     one-step error on the true states of the first 2 ps (on these runs) by more than 3 standard
     errors.
  B  temperature: per run, the mean over each 10-ps window (625 steps); a run counts in a window
     only if it is running for the whole window. Drift if the mean over runs of (last window -
     first window) is over 10 K and over 3 standard errors. Energy reported on the same scale
     (3.9 meV per molecule = 10 K of kinetic energy).
  C  per element and 10-ps window: distance between the mean feature vector of the predicted
     states (pooled over runs) and of the true states (pooled over the 28 reference runs, every
     5th step of the first 2 ps), divided by the square root of the reference spread (mean over
     atoms of that element of the squared distance to the reference mean). Noise level: root mean
     square of the same distance between the reference mean and 200 bootstrap resamples of the
     reference runs. Drift if the distance exceeds 3 times the noise level.

  python eval/rollout_analysis.py --rollout results/eval/e22/rollout_primary.npz --exact results/eval/e22/exact_primary.npz \
      --worse results/eval/e22/rollout_e04.npz --drift results/eval/e22/rollout_drift.npz \
      --fail results/eval/e22/rollout_fail.npz --checkpoint results/runs/flashmd_water16_scratch/model.ckpt \
      --config configs/flashmd_water16_scratch.yaml --out_dir results/eval/e22
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

STRIDE_FS = 16.0
WINDOW = 625                  # 10 ps of 16 fs steps
HALF_PS_STEPS = 31            # 0.5 ps
KB_MEV = 8.617333262e-2       # Boltzmann constant in meV per K


def ms_err(q, p, qr, pr, masses):
    sm = np.sqrt(masses)[:, None]
    return np.sqrt((((q - qr) * sm) ** 2).mean()), np.sqrt((((p - pr) / sm) ** 2).mean())


def a1_curves(r):
    """(n_ref, B, 2) error of the rollout against the reference, NaN once a rollout stopped."""
    pos = {int(k): i for i, k in enumerate(r["saved_steps"])}
    B, n_ref = r["ref_q"].shape[:2]
    out = np.full((n_ref, B, 2), np.nan)
    for k in range(n_ref):
        if k not in pos:
            continue
        for b in range(B):
            q = r["q"][pos[k], b]
            if not np.isnan(q).any():
                out[k, b] = ms_err(q, r["p"][pos[k], b], r["ref_q"][b, k], r["ref_p"][b, k], r["masses"])
    return out


def doubling_step(curve):
    """curve: mean error per step (index = step). First step >= 2 x step-1 value, or None."""
    base = curve[1]
    hit = np.where(curve[1:] >= 2 * base)[0]
    return int(hit[0] + 1) if len(hit) else None


def window_means(T, stopped_at):
    """T: (n_steps + 1, B). Mean per 10-ps window per run; NaN where the run was not running for
    the whole window."""
    n = (T.shape[0] - 1) // WINDOW
    out = np.full((n, T.shape[1]), np.nan)
    for w in range(n):
        a, b = w * WINDOW, (w + 1) * WINDOW
        for r in range(T.shape[1]):
            if stopped_at[r] == -1 or stopped_at[r] >= b:   # stopped at step b: temperatures valid for all of [a, b)
                out[w, r] = T[a:b, r].mean()
    return out


def drift_rule(win, threshold):
    d = win[-1] - win[0]
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return {"runs": int(len(d)), "flagged": None}
    m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    return {"runs": int(len(d)), "mean_change": m, "standard_error": se, "threshold": threshold,
            "flagged": bool(abs(m) > threshold and abs(m) > 3 * se)}


def geometry(qs, cell):
    """O-H lengths, H-O-H angles (degrees) and O-O distances (nearest periodic copy) for states
    (n, N, 3) in O H H order."""
    L = np.asarray(cell)
    o, h1, h2 = qs[:, 0::3], qs[:, 1::3], qs[:, 2::3]
    mi = lambda d: d - L * np.round(d / L)  # noqa: E731
    v1, v2 = mi(h1 - o), mi(h2 - o)
    oh = np.concatenate([np.linalg.norm(v1, axis=-1).ravel(), np.linalg.norm(v2, axis=-1).ravel()])
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1))).ravel()
    d = mi(o[:, :, None] - o[:, None, :])
    r = np.linalg.norm(d, axis=-1)
    iu = np.triu_indices(o.shape[1], 1)
    oo = r[:, iu[0], iu[1]].ravel()
    return oh, ang, oo


def features_c(args, r, device):
    import torch
    from eval.feature_change_vs_jump import features, load_encoder
    from train.pretrain_jepa import load_config
    cfg, hypers, masses_z, timestep, info = load_config(args.config)
    enc = load_encoder("baseline", args.checkpoint, hypers, info, masses_z, timestep, device)
    # the encoder read from model.ckpt must be the one inside the rolled-out model.pt
    from metatomic.torch import load_atomistic_model
    exported = dict(load_atomistic_model(str(r["model"])).module.named_parameters())
    common = [k for k, _ in enc.named_parameters() if k in exported]
    assert common, "no shared parameter names between model.ckpt and model.pt"
    worst = max(float((dict(enc.named_parameters())[k].detach().cpu().double() - exported[k].detach().cpu().double()).abs().max()) for k in common)
    assert worst < 1e-6, f"model.ckpt encoder differs from model.pt by {worst}"
    print(f"encoder in model.ckpt equals model.pt on {len(common)} parameters")
    numbers = r["numbers"]; types = torch.tensor(numbers, dtype=torch.int32)
    cell = torch.tensor(np.diag(r["cell"]))
    elements = sorted(set(numbers.tolist()))
    B, n_ref = r["ref_q"].shape[:2]

    def per_run_sums(qs_ps):
        """list over runs of {z: (sum of features, sum of squared norms, count)}"""
        out = []
        for frames in qs_ps:
            if not frames:
                out.append(None); continue
            h = features(enc, [(types, torch.tensor(q), torch.tensor(p), cell) for q, p in frames], device)
            acc = {}
            for z in elements:
                hz = np.concatenate([x[numbers == z] for x in h])
                acc[z] = (hz.sum(0), (hz ** 2).sum(), len(hz))
            out.append(acc)
        return out

    ref = per_run_sums([[(r["ref_q"][b, k], r["ref_p"][b, k]) for k in range(0, n_ref, 5)] for b in range(B)])

    def pooled_mean(sums, idx, z):
        s = sum(sums[i][z][0] for i in idx if sums[i] is not None)
        n = sum(sums[i][z][2] for i in idx if sums[i] is not None)
        return s / n

    all_idx = list(range(B))
    ref_mean = {z: pooled_mean(ref, all_idx, z) for z in elements}
    # spread: mean over reference atoms of |h - ref_mean|^2 = E|h|^2 - |ref_mean|^2
    spread = {z: sum(ref[i][z][1] for i in all_idx) / sum(ref[i][z][2] for i in all_idx) - float(ref_mean[z] @ ref_mean[z])
              for z in elements}
    rng = np.random.default_rng(0)
    noise = {}
    for z in elements:
        ds = [np.linalg.norm(pooled_mean(ref, rng.integers(0, B, B), z) - ref_mean[z]) / np.sqrt(spread[z]) for _ in range(200)]
        noise[z] = float(np.sqrt(np.mean(np.square(ds))))
    pos = {int(k): i for i, k in enumerate(r["saved_steps"])}
    n_steps = int(r["temperature"].shape[0] - 1)
    res = []
    for w in range(n_steps // WINDOW):
        ks = [k for k in range(w * WINDOW, (w + 1) * WINDOW) if k % 50 == 0 and k in pos]
        frames = [[(r["q"][pos[k], b], r["p"][pos[k], b]) for k in ks if not np.isnan(r["q"][pos[k], b]).any()] for b in range(B)]
        pred = per_run_sums(frames)
        row = {"window_ps": [w * 10, (w + 1) * 10], "runs": int(sum(1 for f in frames if f))}
        for z in elements:
            if row["runs"] == 0:
                continue
            d = float(np.linalg.norm(pooled_mean(pred, all_idx, z) - ref_mean[z]) / np.sqrt(spread[z]))
            row[str(z)] = {"distance": d, "noise": noise[z], "drift": bool(d > 3 * noise[z])}
        res.append(row)
        print(f"C window {w}: " + ", ".join(f"Z={z}: {row[str(z)]['distance']:.4f} (noise {noise[z]:.4f})" for z in elements if str(z) in row), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--worse", default="", help="control: rollout of a weaker model (must be worse)")
    ap.add_argument("--drift", default="", help="control: this model's rollout with momenta x1.00002 per step")
    ap.add_argument("--fail", default="", help="control: this model's rollout with momenta x1.01 per step")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skip_features", action="store_true")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(args.out_dir, exist_ok=True)
    r = dict(np.load(args.rollout)); x = dict(np.load(args.exact))   # dict: read each array once
    masses = r["masses"]; B = r["ref_q"].shape[0]; n_mol = len(masses) // 3
    S = {"tested_on": "the 28 water test runs (same liquid at 300 K, runs never seen in training)", "model": str(r["model"])}

    # ---- A1 and A3
    a1 = a1_curves(r)
    a1_mean = np.nanmean(a1, axis=1)                                     # (n_ref, 2)
    a3 = x["a3_err"]                                                     # (B, 2, n_ref, 2)
    chaos_mean = a3[:, 0].mean(0); restart_mean = a3[:, 1].mean(0)
    worse_mean = np.nanmean(a1_curves(dict(np.load(args.worse))), axis=1) if args.worse else None
    t = np.arange(len(a1_mean)) * STRIDE_FS / 1000
    S["A1"] = {}
    for j, name in ((1, "momentum"), (0, "position")):
        dm, dc = doubling_step(a1_mean[:, j]), doubling_step(chaos_mean[:, j])
        S["A1"][name] = {"model_doubling_fs": None if dm is None else dm * STRIDE_FS,
                         "chaos_doubling_fs": None if dc is None else dc * STRIDE_FS,
                         "model_error_at_2ps": float(a1_mean[-1, j]), "chaos_error_at_2ps": float(chaos_mean[-1, j]),
                         "restart_error_max": float(np.nanmax(restart_mean[:, j]))}
    dm, dc = S["A1"]["momentum"]["model_doubling_fs"], S["A1"]["momentum"]["chaos_doubling_fs"]
    lim = HALF_PS_STEPS * STRIDE_FS
    S["A1"]["by_element"] = {}
    for z, lab in ((8, "oxygen"), (1, "hydrogen")):
        sel = r["numbers"] == z
        sub = dict(r); sub["masses"] = r["masses"][sel]; sub["q"] = r["q"][:, :, sel]; sub["p"] = r["p"][:, :, sel]
        sub["ref_q"] = r["ref_q"][:, :, sel]; sub["ref_p"] = r["ref_p"][:, :, sel]
        cz = np.nanmean(a1_curves(sub), axis=1)
        S["A1"]["by_element"][lab] = {"momentum_error_at_steps_1_31_125": [float(cz[k, 1]) for k in (1, HALF_PS_STEPS, len(cz) - 1)],
                                      "position_error_at_steps_1_31_125": [float(cz[k, 0]) for k in (1, HALF_PS_STEPS, len(cz) - 1)]}
    S["A1"]["prediction_model_doubles_faster_within_0.5ps"] = bool(dm is not None and dm <= lim and (dc is None or dm < dc))
    S["controls"] = {}
    if worse_mean is not None:
        # both position and momentum error must be larger at every step of the first 0.5 ps
        S["controls"]["must_be_worse"] = bool(np.all(worse_mean[1:HALF_PS_STEPS + 1] > a1_mean[1:HALF_PS_STEPS + 1]))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    for j, lab in ((1, "momentum"), (0, "position")):
        a = ax[0 if j == 1 else 1]
        a.plot(t, a1_mean[:, j], label="model rollout (E00)", color="C0")
        a.plot(t, chaos_mean[:, j], label="exact, start moved by the model's one-step error", color="C1")
        a.plot(t, restart_mean[:, j], label="exact, unchanged start (restart precision)", color="0.5", ls=":")
        if worse_mean is not None:
            a.plot(t, worse_mean[:, j], label="weaker model (control)", color="C3", ls="--")
        a.set_yscale("log"); a.set_xlabel("time (ps)"); a.set_ylabel(f"{lab} error (mass-scaled)")
        a.set_title(f"A1: {lab} error against the exact trajectory\nmean over the 28 test runs", fontsize=9)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "a1_error_growth.png"), bbox_inches="tight"); plt.close(fig)

    # ---- A2
    own, true = x["a2_own"], x["a2_true"]          # (steps, B, 2)
    own_steps, true_steps = list(x["a2_steps"]), list(x["a2_true_steps"])
    early = [i for i, k in enumerate(own_steps) if k in true_steps]
    late = [i for i, k in enumerate(own_steps) if k >= 150]
    S["A2"] = {}
    for j, name in ((1, "momentum"), (0, "position")):
        diff = np.nanmean(np.stack([own[i, :, j] - true[true_steps.index(own_steps[i]), :, j] for i in early]), axis=0)
        diff = diff[~np.isnan(diff)]
        m, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))
        base = float(np.nanmean(true[:, :, j]))
        lr = np.nanmean(own[late, :, j], axis=0); lr = lr[~np.isnan(lr)]
        lm, lse = (float(lr.mean()), float(lr.std(ddof=1) / np.sqrt(len(lr)))) if len(lr) > 1 else (None, None)
        S["A2"][name] = {"first_2ps": {"mean_own_minus_true": m, "standard_error": se, "larger": bool(m > 3 * se)},
                         "after_2ps": {"mean_own": lm, "standard_error": lse, "true_state_error": base,
                                       "larger": None if lm is None else bool(lm - base > 3 * lse)}}
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    ax.plot(np.array(own_steps) * STRIDE_FS / 1000, np.nanmean(own[:, :, 1], axis=1), "o-", ms=3, label="one-step error on the model's own outputs")
    ax.plot(np.array(true_steps) * STRIDE_FS / 1000, np.nanmean(true[:, :, 1], axis=1), "s-", ms=3, label="one-step error on true states")
    ax.set_xscale("symlog", linthresh=2); ax.set_xlabel("rollout time (ps)"); ax.set_ylabel("momentum error (mass-scaled)")
    ax.set_title("A2: is the model worse on its own outputs?\nmean over the 28 test runs", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "a2_own_outputs.png"), bbox_inches="tight"); plt.close(fig)

    # ---- B: temperature, energy, failures, geometry
    T = r["temperature"]; stopped = r["stopped_at"]
    win = window_means(T, stopped)
    S["B"] = {"temperature": drift_rule(win, 10.0),
              "stopped": {str(r["runs"][b]): {"step": int(stopped[b]), "ps": float(stopped[b] * STRIDE_FS / 1000), "reason": str(r["reason"][b])}
                          for b in range(B) if stopped[b] != -1},
              "any_unphysical_within_100ps": bool((stopped != -1).any())}
    es = list(x["energy_steps"]); etot = (x["epot"] + x["ekin"]) / n_mol * 1000      # meV per molecule
    first = [i for i, k in enumerate(es) if k < WINDOW and k % 10 == 0]; last = [i for i, k in enumerate(es) if k >= len(T) - 1 - WINDOW]
    de = np.nanmean(etot[last], axis=0) - np.nanmean(etot[first], axis=0)
    ok_last = [b for b in range(B) if stopped[b] == -1]
    de = de[ok_last]
    S["B"]["energy"] = {"mean_change_meV_per_molecule": float(np.mean(de)) if len(de) else None,
                        "standard_error": float(np.std(de, ddof=1) / np.sqrt(len(de))) if len(de) > 1 else None,
                        "scale_10K_meV_per_molecule": 4.5 * KB_MEV * 10,
                        "reference_conserved_drift_meV_per_molecule_over_2ps": float(np.mean(np.abs(x["ref_conserved_drift"])) / n_mol * 1000)}
    if args.drift:
        drift_r = dict(np.load(args.drift))
        # a run counts in a window only if both its control rollout and its unchanged rollout are running
        sa, sb = drift_r["stopped_at"], stopped
        both = np.where((sa == -1) & (sb == -1), -1, np.where(sa == -1, sb, np.where(sb == -1, sa, np.minimum(sa, sb))))
        S["controls"]["must_fail_drift_flagged"] = drift_rule(window_means(drift_r["temperature"] - T, both), 10.0)
    if args.fail:
        fail_r = dict(np.load(args.fail))
        S["controls"]["must_fail_detector"] = {"all_stopped": bool((fail_r["stopped_at"] != -1).all()),
                                               "latest_stop_step": int(fail_r["stopped_at"].max())}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    tt = np.arange(len(T)) * STRIDE_FS / 1000
    for b in range(B):
        ax[0].plot(tt, T[:, b], color="0.8", lw=0.3)
    wc = (np.arange(win.shape[0]) + 0.5) * 10
    ax[0].plot(wc, np.nanmean(win, axis=1), "o-", color="C0", label="10-ps window mean over runs")
    ax[0].set_xlabel("time (ps)"); ax[0].set_ylabel("temperature (K)"); ax[0].legend(fontsize=8)
    ax[0].set_title("B: temperature of the 28 rollouts (grey: each run)", fontsize=9)
    et = np.array(es) * STRIDE_FS / 1000
    ax[1].plot(et, np.nanmean(etot - etot[0], axis=1), color="C0")
    ax[1].axhline(3.88, color="0.5", ls=":"); ax[1].axhline(-3.88, color="0.5", ls=":")
    ax[1].set_xlabel("time (ps)"); ax[1].set_ylabel("total energy change (meV per molecule)")
    ax[1].set_title("B: total energy, mean over runs\ndotted: the energy of 10 K", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "b_drift.png"), bbox_inches="tight"); plt.close(fig)

    pos = {int(k): i for i, k in enumerate(r["saved_steps"])}
    ref_states = r["ref_q"].reshape(-1, *r["ref_q"].shape[2:])
    lw = [k for k in range(len(T) - 1 - WINDOW, len(T)) if k % 10 == 0 and k in pos]
    last_states = np.concatenate([r["q"][pos[k]][~np.isnan(r["q"][pos[k]]).any(axis=(1, 2))] for k in lw]) if lw else np.zeros((0,) + ref_states.shape[1:])
    g_ref = geometry(ref_states, r["cell"]); g_last = geometry(last_states, r["cell"]) if len(last_states) else None
    S["B"]["geometry"] = {"reference": {"O-H mean A": float(g_ref[0].mean()), "H-O-H mean deg": float(g_ref[1].mean())}}
    if g_last is not None:
        S["B"]["geometry"]["last_window"] = {"O-H mean A": float(g_last[0].mean()), "H-O-H mean deg": float(g_last[1].mean())}
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=130)
        for a, gr, gl, lab, rng_ in zip(ax, g_ref, g_last, ("O-H bond length (A)", "H-O-H angle (degrees)", "O-O distance (A)"),
                                        ((0.8, 1.2), (80, 130), (2.2, 6.0))):
            a.hist(gr, bins=80, range=rng_, density=True, histtype="step", label="exact (28 reference runs)")
            a.hist(gl, bins=80, range=rng_, density=True, histtype="step", label="rollout, last 10 ps")
            a.set_xlabel(lab)
        ax[0].legend(fontsize=7)
        fig.suptitle("B: structure of the liquid, exact against rollout", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "b_structure.png"), bbox_inches="tight"); plt.close(fig)

    # ---- C
    if not args.skip_features:
        import torch
        S["C"] = features_c(args, r, "cuda" if torch.cuda.is_available() else "cpu")
        fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
        for z, lab in ((1, "hydrogen"), (8, "oxygen")):
            rows = [w for w in S["C"] if str(z) in w]
            if not rows:
                continue
            xs = [np.mean(w["window_ps"]) for w in rows]
            ax.plot(xs, [w[str(z)]["distance"] for w in rows], "o-", label=lab)
            ax.axhline(3 * rows[0][str(z)]["noise"], ls=":", color="C0" if z == 1 else "C1")
        ax.set_xlabel("rollout time (ps)"); ax.set_ylabel("distance of mean features from true states\n(in units of the feature spread)")
        ax.set_title("C: do the features of the model's own outputs drift?\ndotted: 3 times the noise level", fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "c_feature_drift.png"), bbox_inches="tight"); plt.close(fig)
        S["C_drift_any"] = bool(any(w[k]["drift"] for w in S["C"] for k in w if k not in ("window_ps", "runs")))

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
