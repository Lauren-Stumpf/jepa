#!/usr/bin/env bash
# Delete most per-epoch checkpoints of FINISHED fine-tuning runs, keeping every 10th epoch
# and the last one, so the FlashMD paper's checkpoint selection (decisions.md, 2026-11-05)
# stays possible on a 10-epoch grid. Agreed with Lauren on 12 Nov 2026.
#
# Only files named model_<N>.ckpt inside results/runs/<run>/outputs*/<date>/<time>/ are ever
# deleted, and only when N is not a multiple of 10 and N is not the highest N in that folder.
# Never touched: model.ckpt, model_last.ckpt, model.pt, pretrained.ckpt, scaler.ckpt, logs.
#
# A run counts as finished when model.pt exists, it has no tmux session on this node, and its
# train.log has not changed for 30 min (a run alive on another node keeps writing train.log).
#
#   bash scripts/thin_checkpoints.sh            # preview: lists what would go, deletes nothing
#   bash scripts/thin_checkpoints.sh --delete   # deletes
set -u
cd /storage3/lauren/code/jepa/results/runs || exit 1
DELETE=0
[ "${1:-}" = "--delete" ] && DELETE=1
EVERY=10

total_files=0; total_bytes=0; skipped=""
for run in */; do
  run=${run%/}
  ls -d "$run"/outputs*/*/*/ >/dev/null 2>&1 || continue
  if [ ! -f "$run/model.pt" ] || tmux has-session -t "=train_$run" 2>/dev/null \
     || [ -n "$(find "$run" -maxdepth 1 -name train.log -mmin -30 2>/dev/null)" ]; then
    skipped="$skipped $run"; continue
  fi
  for d in "$run"/outputs*/*/*/; do
    last=$(ls "$d" | sed -nE 's/^model_([0-9]+)\.ckpt$/\1/p' | sort -n | tail -1)
    [ -n "$last" ] || continue
    for f in "$d"model_*.ckpt; do
      n=$(basename "$f" | sed -nE 's/^model_([0-9]+)\.ckpt$/\1/p')
      [ -n "$n" ] || continue
      [ $((10#$n % EVERY)) -eq 0 ] && continue
      [ "$n" = "$last" ] && continue
      b=$(stat -c %s "$f")
      total_files=$((total_files + 1)); total_bytes=$((total_bytes + b))
      if [ "$DELETE" -eq 1 ]; then rm -f -- "$f"; else echo "would delete $f"; fi
    done
  done
done
echo "skipped (not finished or still running):${skipped:- none}"
action=$([ "$DELETE" -eq 1 ] && echo deleted || echo "would delete")
echo "$action $total_files files, $((total_bytes / 1024 / 1024 / 1024)) GB"
