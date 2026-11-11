#!/usr/bin/env bash
# Run one metatrain training to completion. If it is interrupted (crash, killed process,
# node reboot followed by launch_all.sh) it is started AGAIN FROM SCRATCH: metatrain's
# FlashMD trainer, when restarted from a checkpoint, trains num_epochs MORE epochs on a
# learning-rate curve that no longer matches, and a restarted fine-tune also loses its
# optimiser state, so a resumed run would not be comparable with an uninterrupted one.
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
hostname -s > host   # which node runs this, so the launcher on the other node leaves it alone
fails=0
while [ ! -f model.pt ]; do
  if [ -d outputs ]; then
    echo "$(date '+%F %T') $NAME was interrupted; starting again from scratch (previous outputs moved aside)" | tee -a supervisor.log
    mv outputs "outputs_interrupted_$(date '+%Y%m%d_%H%M%S')"
    mv train.log "train_interrupted_$(date '+%Y%m%d_%H%M%S').log" 2>/dev/null
    for c in model.ckpt model_last.ckpt; do [ -f "$c" ] && mv "$c" "${c%.ckpt}_interrupted_$(date '+%Y%m%d_%H%M%S').ckpt"; done
  fi
  echo "$(date '+%F %T') starting $NAME on GPU $GPU" | tee -a supervisor.log
  mtt train options.yaml -o model.pt >> train.log 2>&1
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
