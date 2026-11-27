#!/bin/bash
# E27 control with two learning-rate cycles, stage 2: when stage 1 (the 400-epoch baseline) has
# finished, save its encoder in the pretrained-encoder format (train/encoder_from_baseline.py, the
# E14 route). The stage-3 runs in experiments/E27/runs_restart.txt wait for its manifest.json.
# Run on the node of the launcher, from the repository root:
#   tmux new-session -d -s e27_restart_stage2 "bash scripts/run_e27_restart.sh >> results/launch_E27_restart_stage2.log 2>&1"
set -euo pipefail
source env_train.sh >/dev/null 2>&1
S1=e27_restart_stage1_400ep_water16_10pct_s0
ENC=results/runs/e27_restart_encoder_water16
echo "$(date '+%F %T') waiting for results/runs/$S1/model.pt"
until [ -f results/runs/$S1/model.pt ]; do
  if grep -q "giving up" results/runs/$S1/supervisor.log 2>/dev/null; then echo "stage 1 was abandoned by its supervisor"; exit 1; fi
  sleep 600
done
if [ -f $ENC/manifest.json ]; then echo "$ENC already written"; exit 0; fi
echo "$(date '+%F %T') stage 1 finished; writing its encoder"
python train/encoder_from_baseline.py configs/$S1.yaml --source results/runs/$S1/model.ckpt --out $ENC \
    --check_frames data/water/pairs/16fs/train_10pct.xyz
ls -la $ENC
echo "$(date '+%F %T') done"
