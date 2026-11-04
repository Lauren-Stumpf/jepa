#!/usr/bin/env bash
# Organic track, from a complete catalogue to a passed smoke test, in one go.
# Stops at the first failing step and says which one. Run from the repo root:
#   bash scripts/organic_setup.sh
set -euo pipefail
XYZ=${XYZ:-/storage3/lauren/data/flashmd/universal_8fs.xyz}
CAT=results/data/catalogue_8fs.csv
CFG=configs/flashmd_organic8_scratch.yaml
step() { echo; echo "=== $1"; }

step "0. catalogue present"
[ "$(wc -l < $CAT)" -eq 5000001 ] || { echo "catalogue has $(wc -l < $CAT) lines, expected 5000001 (10,000 structures x 500 frames); rerun data/catalogue.py"; exit 1; }

step "1. unit tests"
python checks/test_prepare.py | tail -1

step "2. prepare the organic subset (no frame excluded: the distributions show no explosions)"
python data/prepare_organic.py "$XYZ" $CAT $CFG --out_dir data/organic8 --seed 0 --max_T 1e9 --max_disp 1e9 2>&1 | tee results/data/prepare_organic.log
grep -q "'train': 84000" results/data/prepare_organic.log || { echo "unexpected frame counts, see the log"; exit 1; }

step "3. smoke subset"
python data/make_smoke_subset.py data/organic8 --every 20

step "4. packages"
uv pip install -q metatrain wandb plotly
python -c "import metatrain, wandb; print('metatrain', metatrain.__version__)"

step "5. wandb login (skipped if already logged in)"
wandb login 2>&1 | tail -1

step "6. augmentation check"
python checks/check_augmentation.py | tee /tmp/aug.txt
grep -q "OK" /tmp/aug.txt || { echo "augmentation check did not print OK"; exit 1; }

step "7. smoke training (2 epochs on the subset)"
mkdir -p results/models results/eval
mtt train configs/flashmd_organic8_smoke.yaml -o results/models/smoke.pt

step "8. smoke evaluation"
python eval/eval_onestep.py data/organic8/test.xyz --model results/models/smoke.pt --scaled_targets --config $CFG \
    --n 200 --every 50 --out results/eval/smoke.csv

step "done: everything passed; commit the log and manifest, then launch the full baseline:"
echo "  git add results/data/prepare_organic.log data/organic8/manifest.json && git commit -m 'organic subset prepared' && git push"
echo "  tmux new -s organic_train"
echo "  mtt train $CFG -o results/models/flashmd_organic8_scratch.pt"
