"""
Delete from Weights and Biases the runs whose local files are under a folder, so that a run
that failed, was interrupted and restarted from scratch, or was abandoned does not stay in
the project. Called by scripts/train_supervisor.sh and scripts/pretrain_supervisor.sh.

A wandb run writes a local folder named run-<date>_<time>-<run id> (metatrain writes it under
the run's working directory, results/runs/<name>/wandb/; train/pretrain_jepa.py under
results/runs/<name>/wandb/ too). The project is read from that folder's saved config
(files/config.yaml: wandb.project for metatrain runs, the "wandb" argument for pretrainings).

Only projects listed in DELETABLE are touched, so the old project is never changed.
Deleting is permanent. Errors (no network, run already gone) are printed and ignored, so a
supervisor never stops because of this script.

  python scripts/wandb_delete_runs.py results/runs/<name>/wandb [--dry_run]
"""

import argparse
import glob
import os
import re

DELETABLE = {"jepa-flashmd-experiments"}
ENTITY = "lauren-stumpf-oxford-university"


def project_of(run_dir):
    import yaml
    cfg_path = os.path.join(run_dir, "files", "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    cfg = yaml.safe_load(open(cfg_path)) or {}

    def val(key):   # wandb stores each config entry as {"value": ...}
        v = cfg.get(key)
        return v.get("value") if isinstance(v, dict) and "value" in v else v
    w = val("wandb")
    if isinstance(w, dict):          # metatrain: the whole options, including the wandb block
        return w.get("project")
    if isinstance(w, str):           # train/pretrain_jepa.py: --wandb <project>
        return w
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wandb_dir")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(args.wandb_dir, "run-*"))) + sorted(glob.glob(os.path.join(args.wandb_dir, "offline-run-*")))
    if not dirs:
        print(f"no wandb runs under {args.wandb_dir}")
        return
    try:
        import wandb
        api = wandb.Api(timeout=30)
        entity = ENTITY
    except Exception as e:   # noqa: BLE001
        print(f"wandb not reachable, nothing deleted: {e}")
        return
    done = set()
    for d in dirs:
        m = re.search(r"run-\d{8}_\d{6}-([A-Za-z0-9_\-]+)$", os.path.basename(d))
        if not m:
            continue
        run_id, project = m.group(1), project_of(d)
        if (project, run_id) in done:
            continue
        done.add((project, run_id))
        if project not in DELETABLE:
            print(f"left alone: run {run_id} in project {project!r} (only {sorted(DELETABLE)} are cleaned)")
            continue
        path = f"{entity}/{project}/{run_id}"
        if args.dry_run:
            print(f"would delete {path}")
            continue
        try:
            api.run(path).delete()
            print(f"deleted {path}")
        except Exception as e:   # noqa: BLE001
            print(f"could not delete {path}: {e}")


if __name__ == "__main__":
    main()
