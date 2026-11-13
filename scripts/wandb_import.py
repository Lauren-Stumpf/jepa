"""
Copy valid finished runs into the experiment project from their files on the cluster, filed by
experiment (wandb group) and arm (wandb job_type), with the same metric names metatrain logs.
The old project is not touched.

Two kinds of run are imported:
  - runs made before the experiment project existed (they logged to the old project only)
  - runs that logged live to the experiment project but are marked "failed" there although they
    finished (metatrain 2026.4.1 crashed after saving the final model, in its own final
    evaluation; fixed on 13 Nov 2026 by scripts/patch_metatrain.py). These are only replaced
    with --replace_failed_live: the failed live run is deleted and the run imported from its
    files. A live run marked "finished" is the real record and is always left alone.

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

  python scripts/wandb_import.py [--only E04] [--replace] [--replace_failed_live] [--dry_run]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from wandb_common import ENTITY, PROJECT, REPO, classify, epochs_from_log, latest_train_log, test_metrics  # noqa: E402

FAILED_STATES = ("failed", "crashed")


def new_import_id(name):
    import uuid
    return f"imp-{name.replace('_', '-')}-{uuid.uuid4().hex[:6]}"[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="import one experiment only, e.g. E04")
    ap.add_argument("--replace", action="store_true", help="delete and re-import runs already imported")
    ap.add_argument("--replace_failed_live", action="store_true",
                    help="delete live runs marked failed/crashed whose run finished, and import them from their files")
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
        test_name = name.replace("headsfirst1_", "headsfirst2_") if headsfirst else name
        label = test_name if headsfirst else name

        # an error here (no network) stops the script rather than risk importing twice
        found = list(api.runs(f"{entity}/{PROJECT}", filters={"config.source_run": name}))
        if found and not args.replace:
            print(f"skip (already imported): {name}")
            continue
        # runs that logged live to this project under the same name (no source_run in their config)
        live = [r for r in api.runs(f"{entity}/{PROJECT}", filters={"display_name": label})
                if r.name == label and "source_run" not in r.config]
        if any(r.state not in FAILED_STATES for r in live):
            states = sorted({r.state for r in live})
            print(f"skip (logged live, state {', '.join(states)}): {name}")
            continue
        if live and not args.replace_failed_live:
            print(f"skip (logged live but marked failed; use --replace_failed_live): {name}")
            continue
        to_delete = found + live

        # ---- assemble history and summary
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
        why = "replacing a live run marked failed" if live else ("replacing an earlier import" if found else "new")
        print(f"{'would import' if args.dry_run else 'importing'} {name} -> {exp} / {arm} / seed {seed}: "
              f"{len(rows)} epochs, test {'yes' if summary else 'no'} ({why})")
        if args.dry_run:
            continue
        for o in to_delete:
            o.delete()
        if live:
            notes = (f"Imported from results/runs/{name}. It logged live here but was marked failed: "
                     "metatrain crashed after saving the final model (fixed 13 Nov 2026). Same numbers, from its log.")
        else:
            notes = f"Imported from results/runs/{name} (logged before the experiment project existed)."
        rid = new_import_id(name)
        local = os.path.join(REPO, "results", "wandb_import")
        os.makedirs(local, exist_ok=True)
        run = wandb.init(entity=ENTITY, project=PROJECT, id=rid, resume="never", name=label, group=exp, job_type=arm,
                         config=config, notes=notes, dir=local)
        for r in rows:
            step = r.pop("epoch")
            run.log({k: v for k, v in r.items() if isinstance(v, (int, float))}, step=step)
        for k, v in summary.items():
            run.summary[k] = v
        run.finish()


if __name__ == "__main__":
    main()
