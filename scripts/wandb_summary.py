"""
Make (or remake) the summary of one experiment in the Weights and Biases project: a run in the
experiment's group with job_type "summary", holding
  - the experiment card (experiments/<E>/card.md) as the run's notes, so it shows on the run's page
  - one bar chart per measure (validation and test momentum and position errors): one bar per
    arm, the mean over seeds, a line for the spread between seeds, each seed as a dot, and the
    percentage against the baseline arm
  - learning curves: validation momentum error against epoch, the mean over seeds per arm with
    a band for the spread
  - a table of the numbers in the charts
Before drawing, test errors from results/eval/test/<run name>.log are written into each run's
summary (so the project's own table shows them too).

Arms whose name starts with "pretrain-" are pretrainings and are not drawn in the bar charts.
The previous summary run of the experiment is deleted first, so there is one per experiment.

  python scripts/wandb_summary.py E04
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd   # needed for learning curves; imported first so a missing package stops the script before anything is deleted

sys.path.insert(0, os.path.dirname(__file__))
from wandb_common import (ENTITY, EXPERIMENTS, PROJECT, REPO, TEST_P, TEST_Q, TRAIN_P, VAL_P, VAL_Q,  # noqa: E402
                          test_metrics)

# experiments whose comparison arms live in another experiment's group: (group, job type, label here)
REFERENCES = {
    "E02": [("E00", "organic8-100ep", "baseline")],
    "E06": [("E04", "baseline", "baseline"), ("E04", "plain", "plain (E04)")],
}

MEASURES = [(VAL_P, "Validation momentum error"), (TEST_P, "Test momentum error"),
            (VAL_Q, "Validation position error"), (TEST_Q, "Test position error"),
            (TRAIN_P, "Training momentum error")]


def order_arms(arms):
    base = [a for a in arms if a == "baseline" or a.startswith("random")]
    return base + sorted(a for a in arms if a not in base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    args = ap.parse_args()
    exp = args.experiment
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import wandb
    from PIL import Image
    api = wandb.Api(timeout=60)
    entity = ENTITY
    runs = list(api.runs(f"{entity}/{PROJECT}", filters={"group": exp}))
    for r in runs:
        if r.job_type == "summary":
            r.delete()
    runs = [r for r in runs if r.job_type != "summary" and r.state == "finished"]
    train_runs = [r for r in runs if not (r.job_type or "").startswith("pretrain-")]
    # reference arms filed under another experiment, shown here as extra arms
    arm_of = {r.id: r.job_type for r in train_runs}
    for ref_exp, ref_arm, label in REFERENCES.get(exp, []):
        for r in api.runs(f"{entity}/{PROJECT}", filters={"group": ref_exp}):
            if r.state == "finished" and r.job_type == ref_arm:
                arm_of[r.id] = label
                train_runs.append(r)
    if not train_runs:
        print(f"{exp}: no finished training runs yet")
        return

    # test errors into each run's summary
    for r in train_runs:
        t = test_metrics(r.name)
        if t and any(r.summary.get(k) != v for k, v in t.items()):
            for k, v in t.items():
                r.summary[k] = v
            r.summary.update()

    arms = order_arms(sorted({arm_of[r.id] for r in train_runs}))

    def base_of(a):
        """The arm each arm is compared with: "baseline" in experiments that have one; in the
        freezing experiment, the random-start arm with the same procedure (plain-headsfirst is
        compared with random-headsfirst)."""
        if a == "baseline" or a.startswith("random"):
            return None
        if "-" in a and "random-" + a.split("-", 1)[1] in arms:
            return "random-" + a.split("-", 1)[1]
        return "baseline" if "baseline" in arms else None
    vals = {a: {k: [r.summary.get(k) for r in train_runs if arm_of[r.id] == a and r.summary.get(k) is not None]
                for k, _ in MEASURES} for a in arms}
    seeds = {a: sum(1 for r in train_runs if arm_of[r.id] == a) for a in arms}

    def mean(a, k):
        v = vals[a][k]
        return float(np.mean(v)) if v else None

    table_rows = []
    for a in arms:
        row = [a, seeds[a]]
        for k, _ in MEASURES:
            m = mean(a, k)
            sd = float(np.std(vals[a][k], ddof=1)) if len(vals[a][k]) > 1 else None
            ba = base_of(a)
            b = mean(ba, k) if ba else None
            pct = 100 * (m / b - 1) if (m is not None and b) else None
            row += [m, sd, pct]
        table_rows.append(row)
    cols = ["arm", "seeds"] + [c for _, lab in MEASURES for c in (lab, lab + " spread", lab + " vs baseline %")]

    images = {}
    for k, lab in MEASURES:
        present = [a for a in arms if vals[a][k]]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(7, 0.6 * len(present) + 1.4), dpi=130)
        y = np.arange(len(present))[::-1]
        for yi, a in zip(y, present):
            v = vals[a][k]
            m = float(np.mean(v))
            ba = base_of(a)
            b = mean(ba, k) if ba in present else None
            ax.barh(yi, m, color="#DD8452" if ba else "#4C72B0", height=0.6)
            if len(v) > 1:
                sd = float(np.std(v, ddof=1))
                ax.plot([m - sd, m + sd], [yi, yi], color="black", lw=1.5)
            ax.scatter(v, [yi] * len(v), color="black", s=10, zorder=3)
            txt = f"{m:.4g}"
            if b is not None:
                txt += f"   {100 * (m / b - 1):+.1f}% vs {ba}"
            ax.text(max(max(v), m) * 1.02, yi, txt, va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a} ({len(vals[a][k])})" for a in present])
        ax.set_xlabel(f"{lab} (mass-scaled, per atom; lower is better)")
        ax.set_xlim(0, max(max(vals[a][k]) for a in present) * 1.45)
        ax.set_title(f"{exp}: {lab}\n(blue = reference arm, orange = compared with it; bar = mean over seeds, dots = seeds, line = spread)", fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        images[lab] = wandb.Image(Image.open(buf))

    # learning curves of validation momentum error
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    for i, a in enumerate(arms):
        curves = []
        for r in train_runs:
            if arm_of[r.id] != a:
                continue
            h = r.history(keys=[VAL_P], pandas=True, samples=5000)
            if len(h):
                curves.append(h.set_index("_step")[VAL_P])
        if not curves:
            continue
        df = pd.concat(curves, axis=1).sort_index()
        m, s = df.mean(axis=1), df.std(axis=1)
        ax.plot(df.index, m, label=f"{a} ({df.shape[1]})", color=f"C{i}")
        if df.shape[1] > 1:
            ax.fill_between(df.index, m - s, m + s, color=f"C{i}", alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation momentum error (log scale)")
    ax.set_title(f"{exp}: learning curves (mean over seeds, band = spread)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
    images["Learning curves"] = wandb.Image(Image.open(buf))

    card = os.path.join(REPO, "experiments", exp, "card.md")
    notes = open(card).read() if os.path.exists(card) else EXPERIMENTS.get(exp, "")
    local = os.path.join(REPO, "results", "wandb_import")
    os.makedirs(local, exist_ok=True)
    run = wandb.init(entity=ENTITY, project=PROJECT, group=exp, job_type="summary", name=f"{exp} summary", notes=notes,
                     config={"experiment": exp, "description": EXPERIMENTS.get(exp, "")}, dir=local)
    log = {f"summary/{lab}": img for lab, img in images.items()}
    log["summary/table"] = wandb.Table(columns=cols, data=table_rows)
    run.log(log)
    run.finish()
    print(f"{exp}: summary written ({len(train_runs)} runs, arms {arms})")


if __name__ == "__main__":
    main()
