"""
Copy the valid finished runs made before the experiment project existed into it, filed by
experiment (wandb group) and arm (wandb job_type), with the same metric names metatrain logs.
The old project is not touched.

What is imported, from the files on the cluster:
  training runs   every epoch of the latest train.log (training and validation errors, loss,
                  learning rate), plus test errors if eval/eval_onestep.py has been run on them
  heads-first     stage 1 (epochs 0-9) and stage 2 (logged as epochs 10-99) as ONE run, so its
                  curve is comparable with 100-epoch runs
  pretrainings    every epoch of history.json (prediction error, ratio to copying, regulariser
                  terms, rank), under the keys train/pretrain_jepa.py logs

Only finished runs are imported (model.pt, or manifest.json for a pretraining). An import is
found again by its source run name (config.source_run), so running this again skips what is
already there; --replace deletes it and imports it again under a new id (wandb does not reuse
a deleted run's id).
Runs that failed, were cancelled or moved to results/runs_invalid are never imported.

  python scripts/wandb_import.py [--only E04] [--replace] [--dry_run]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from wandb_common import ENTITY, PROJECT, REPO, classify, epochs_from_log, latest_train_log, test_metrics  # noqa: E402


def new_import_id(name):
    import uuid
    return f"imp-{name.replace('_', '-')}-{uuid.uuid4().hex[:6]}"[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="import one experiment only, e.g. E04")
    ap.add_argument("--replace", action="store_true", help="delete and re-import runs already imported")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    import wandb
    api = wandb.Api(timeout=60)
    entity = ENTITY
    runs_root = os.path.join(REPO, "results", "runs")

    for name in sorted(os.listdir(runs_root)):
        c = classify(name)
        if c is None:
            continue
        exp, arm, seed = c
        if args.only and exp != args.only:
            continue
        d = os.path.join(runs_root, name)
        is_pre = arm.startswith("pretrain-")
        headsfirst = name.startswith("headsfirst1_")
        stage2 = os.path.join(runs_root, name.replace("headsfirst1_", "headsfirst2_")) if headsfirst else None
        done = (os.path.exists(os.path.join(d, "manifest.json")) if is_pre else
                os.path.exists(os.path.join(stage2 if headsfirst else d, "model.pt")))
        if not done:
            print(f"skip (not finished): {name}")
            continue
        # an error here (no network) stops the script rather than risk importing twice
        found = list(api.runs(f"{entity}/{PROJECT}", filters={"config.source_run": name}))
        old = found[0] if found else None
        if old is not None and not args.replace:
            print(f"skip (already imported): {name}")
            continue

        # ---- assemble history and summary
        test_name = name.replace("headsfirst1_", "headsfirst2_") if headsfirst else name
        summary = test_metrics(test_name)
        if is_pre:
            hist = json.load(open(os.path.join(d, "history.json")))
            rows = [dict(r, epoch=int(r["epoch"])) for r in hist]
            config = {"experiment": exp, "arm": arm, "seed": seed, "source_run": name, "imported": True,
                      **json.load(open(os.path.join(d, "manifest.json"))).get("args", {})}
            config.update({"experiment": exp, "arm": arm, "source_run": name, "imported": True})
        else:
            rows = epochs_from_log(latest_train_log(d))
            n1 = len(rows)
            if headsfirst:
                rows2 = epochs_from_log(latest_train_log(stage2))
                for r in rows2:
                    r["epoch"] += n1
                rows += rows2
            config = {"experiment": exp, "arm": arm, "seed": seed, "source_run": name, "imported": True,
                      "epochs": len(rows)}
            if headsfirst:
                config.update({"stage1_epochs_heads_only": n1, "stage2_run": os.path.basename(stage2)})
        if not rows:
            print(f"skip (no epochs found in its log): {name}")
            continue
        label = test_name if headsfirst else name
        print(f"{'would import' if args.dry_run else 'importing'} {name} -> {exp} / {arm} / seed {seed}: "
              f"{len(rows)} epochs, test {'yes' if summary else 'no'}")
        if args.dry_run:
            continue
        for o in found:
            o.delete()
        rid = new_import_id(name)
        local = os.path.join(REPO, "results", "wandb_import")
        os.makedirs(local, exist_ok=True)
        run = wandb.init(entity=ENTITY, project=PROJECT, id=rid, resume="never", name=label, group=exp, job_type=arm, config=config,
                         notes=f"Imported from results/runs/{name} (logged before the experiment project existed).",
                         dir=local)
        for r in rows:
            step = r.pop("epoch")
            run.log({k: v for k, v in r.items() if isinstance(v, (int, float))}, step=step)
        for k, v in summary.items():
            run.summary[k] = v
        run.finish()


if __name__ == "__main__":
    main()
