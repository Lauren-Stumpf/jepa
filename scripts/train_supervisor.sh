#!/usr/bin/env bash
# Run one metatrain training to completion, resuming from its own latest checkpoint
# whenever it is interrupted (crash, killed process, node reboot followed by launch_all.sh).
#   bash scripts/train_supervisor.sh <name> <config.yaml> <gpu index>
# Each run lives in results/runs/<name>/ so its checkpoints (outputs/<date>/<time>/*.ckpt)
# cannot be confused with another run's. Finished when results/runs/<name>/model.pt exists.
set -uo pipefail
NAME=${1:?name}; CFG=${2:?config}; GPU=${3:?gpu index}
REPO=/storage3/lauren/code/jepa
RUN=$REPO/results/runs/$NAME
MAX_FAILS=${MAX_FAILS:-5}
mkdir -p "$RUN" && cd "$RUN"
ln -sfn "$REPO/data" data            # configs read data/... relative to the working directory
cp "$REPO/$CFG" options.yaml
source "$REPO/env_train.sh"
export CUDA_VISIBLE_DEVICES=$GPU
echo "$GPU" > gpu
fails=0
while [ ! -f model.pt ]; do
  ckpt=$(ls -t outputs/*/*/*.ckpt 2>/dev/null | head -1)
  if [ -n "$ckpt" ]; then
    echo "$(date '+%F %T') resuming $NAME from $ckpt on GPU $GPU" | tee -a supervisor.log
    mtt train options.yaml -o model.pt --restart "$ckpt" >> train.log 2>&1
  else
    echo "$(date '+%F %T') starting $NAME on GPU $GPU" | tee -a supervisor.log
    mtt train options.yaml -o model.pt >> train.log 2>&1
  fi
  status=$?
  if [ ! -f model.pt ]; then
    fails=$((fails + 1))
    echo "$(date '+%F %T') $NAME stopped with status $status (failure $fails of $MAX_FAILS); last log lines:" | tee -a supervisor.log
    tail -5 train.log | tee -a supervisor.log
    [ "$fails" -ge "$MAX_FAILS" ] && { echo "giving up on $NAME" | tee -a supervisor.log; exit 1; }
    sleep 300
  fi
done
echo "$(date '+%F %T') $NAME finished: $RUN/model.pt" | tee -a supervisor.log
