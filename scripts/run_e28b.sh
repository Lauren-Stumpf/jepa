#!/bin/bash
# E28, second round (approved by Lauren, 26 Nov: "run whatever you need to"). Run on torrnode7 from the repository root:
#   tmux new-session -d -s e28b "bash scripts/run_e28b.sh > results/eval/e28/run_e28b.log 2>&1"
#
# 1. Checks of the new rollout options on the primary model (stop if either fails):
#    the batched step still matches the one-step evaluation; the rotation checks.
# 2. B: the primary model rolled out with all of the FlashMD paper's inference steps that apply
#    without a thermostat: energy correction, total-momentum constraint, random rotation.
# 3. C: every other trained water model at 16 fs, rolled out the same way, to measure how much
#    each score varies between training seeds. Two rollouts per graphics card.
# 4. Exact dynamics with the same Langevin thermostat as the thermostat rollouts (processors only),
#    the reference for those rollouts.
# 5. Scores of every rollout against the matching exact reference.
set -euo pipefail
source env_train.sh >/dev/null 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=results/eval/e28
mkdir -p $OUT/c
PRIMARY=flashmd_water16_scratch
FLAGS="--conserve_energy --fix_momentum --random_rotation"

echo "$(date) step 1: checks"
CUDA_VISIBLE_DEVICES=0 python eval/rollout_water.py results/runs/$PRIMARY/model.pt configs/$PRIMARY.yaml --check_onestep data/water/pairs/16fs/test.xyz
CUDA_VISIBLE_DEVICES=0 python eval/rollout_water.py results/runs/$PRIMARY/model.pt configs/$PRIMARY.yaml --check_rotation 10

echo "$(date) step 4 started in the background: exact dynamics with the Langevin thermostat"
python eval/exact_reference_water.py --like $OUT/rollout_langevin.npz --langevin_fs 100 --target_T 300 \
    --out $OUT/exact_reference_langevin.npz --workers 24 > $OUT/exact_reference_langevin.log 2>&1 &
EXACT_PID=$!

echo "$(date) steps 2 and 3: rollouts"
MODELS="$PRIMARY flashmd_water16_scratch_10pct_s0
e16_baseline_water16_10pct_1400ep_s0 e16_baseline_water16_10pct_1400ep_s1 e16_baseline_water16_10pct_1400ep_s2
e05_plain_water16_10pct_1000ep_s0 e05_plain_water16_10pct_1000ep_s1 e05_plain_water16_10pct_1000ep_s2
e27_baseline_2400ep_water16_10pct_s0 e27_baseline_2400ep_water16_10pct_s1 e27_baseline_2400ep_water16_10pct_s2
e27_plain_2000ep_water16_10pct_s0 e27_plain_2000ep_water16_10pct_s1 e27_plain_2000ep_water16_10pct_s2"
i=0
PIDS=()
for m in $MODELS; do
  gpu=$(( i % 8 ))
  CUDA_VISIBLE_DEVICES=$gpu python eval/rollout_water.py results/runs/$m/model.pt configs/$m.yaml $FLAGS \
      --out $OUT/c/rollout_$m.npz > $OUT/c/rollout_$m.log 2>&1 &
  PIDS+=($!)
  i=$(( i + 1 ))
done
FAILED=0
for pid in "${PIDS[@]}"; do wait $pid || FAILED=$(( FAILED + 1 )); done
echo "$(date) rollouts finished; $FAILED failed"
wait $EXACT_PID || echo "$(date) exact dynamics with the thermostat FAILED its check (see $OUT/exact_reference_langevin.log)"
tail -3 $OUT/exact_reference_langevin.log

echo "$(date) step 5: scores"
ARGS=()
for m in $MODELS; do
  [ -f $OUT/c/rollout_$m.npz ] && ARGS+=("$m=$OUT/c/rollout_$m.npz")
done
python eval/rollout_scores.py --exact $OUT/exact_reference.npz \
    --rollouts "energy correction (first round)=$OUT/rollout_energy.npz" "${ARGS[@]}" --out $OUT/scores_c.json
if [ -f $OUT/exact_reference_langevin.npz ]; then
  python eval/rollout_scores.py --exact $OUT/exact_reference_langevin.npz \
      --rollouts "thermostat only=$OUT/rollout_langevin.npz" "energy correction and thermostat=$OUT/rollout_energy_langevin.npz" \
      --out $OUT/scores_langevin.json
fi
echo "$(date) done"
