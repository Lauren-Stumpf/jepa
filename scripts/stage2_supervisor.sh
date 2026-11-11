#!/usr/bin/env bash
# Stage 2 of the heads-first procedure: make stage 1's last-epoch checkpoint
# (train/last_epoch_checkpoint.py; metatrain would otherwise start from stage 1's
# best-validation epoch), then run stage 2 to completion with train_supervisor.sh.
#   bash scripts/stage2_supervisor.sh <stage1 name> <stage2 name> <gpu index>
set -uo pipefail
S1=${1:?stage 1 name}; S2=${2:?stage 2 name}; GPU=${3:?gpu index}
REPO=/storage3/lauren/code/jepa
cd "$REPO"
[ -f "results/runs/$S1/model.pt" ] || { echo "stage 1 ($S1) has not finished"; exit 1; }
if [ ! -f "results/runs/$S1/model_last.ckpt" ]; then
  ( source env_train.sh && python train/last_epoch_checkpoint.py "results/runs/$S1/model.ckpt" \
      "results/runs/$S1/model_last.ckpt" ) >> "results/runs/$S1/supervisor.log" 2>&1 \
    || { echo "could not write results/runs/$S1/model_last.ckpt" | tee -a "results/runs/$S1/supervisor.log"; exit 1; }
fi
exec bash "$REPO/scripts/train_supervisor.sh" "$S2" "configs/$S2.yaml" "$GPU"
