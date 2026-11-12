#!/usr/bin/env bash
# Run the JEPA pretraining to completion, resuming from its state file after any
# interruption (the script itself resumes from state.pt).
#   [CONFIG=... TRAIN=... VAL=...] bash scripts/pretrain_supervisor.sh <name> <gpu index> [extra args for train/pretrain_jepa.py]
# Finished when results/runs/<name>/manifest.json exists.
set -uo pipefail
NAME=${1:?name}; GPU=${2:?gpu index}; shift 2
REPO=/storage3/lauren/code/jepa
RUN=$REPO/results/runs/$NAME
MAX_FAILS=${MAX_FAILS:-5}
# dataset: organic 8 fs by default; the water launcher sets these three
CONFIG=${CONFIG:-configs/flashmd_organic8_scratch.yaml}
TRAIN=${TRAIN:-data/organic8/train.xyz}
VAL=${VAL:-data/organic8/val.xyz}
mkdir -p "$RUN"
cd "$REPO"
source env_train.sh
export CUDA_VISIBLE_DEVICES=$GPU
echo "$GPU" > "$RUN/gpu"
hostname -s > "$RUN/host"   # which node runs this
fails=0
while [ ! -f "$RUN/manifest.json" ]; do
  echo "$(date '+%F %T') starting/resuming $NAME on GPU $GPU" | tee -a "$RUN/supervisor.log"
  python train/pretrain_jepa.py "$CONFIG" --train "$TRAIN" --val "$VAL" \
      --out "$RUN" --epochs "${EPOCHS:-20}" --wandb jepa-flashmd "$@" >> "$RUN/train.log" 2>&1
  status=$?
  if [ ! -f "$RUN/manifest.json" ]; then
    fails=$((fails + 1))
    echo "$(date '+%F %T') $NAME stopped with status $status (failure $fails of $MAX_FAILS); last log lines:" | tee -a "$RUN/supervisor.log"
    tail -5 "$RUN/train.log" | tee -a "$RUN/supervisor.log"
    if [ "$fails" -ge "$MAX_FAILS" ]; then
      echo "giving up on $NAME" | tee -a "$RUN/supervisor.log"
      # an abandoned pretraining's Weights and Biases run is removed (experiment project only)
      # wandb refuses to reuse a deleted run's id, so the id file goes too: a later restart gets a new id
      if [ -d "$RUN/wandb" ]; then
        timeout 300 python "$REPO/scripts/wandb_delete_runs.py" "$RUN/wandb" >> "$RUN/supervisor.log" 2>&1 \
          || echo "Weights and Biases deletion did not finish: the run may need deleting by hand" >> "$RUN/supervisor.log"
        mv "$RUN/wandb" "$RUN/wandb_abandoned_$(date '+%Y%m%d_%H%M%S')"; rm -f "$RUN/wandb_run_id"
      fi
      exit 1
    fi
    sleep 300
  fi
done
echo "$(date '+%F %T') $NAME finished: $RUN/pretrained.ckpt" | tee -a "$RUN/supervisor.log"
