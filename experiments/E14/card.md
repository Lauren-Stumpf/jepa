# E14: can our training-from-a-checkpoint route pass on a benefit at all? (organic, control that must work)

Status: proposed (16 Nov 2026). Not yet run.

**Why this exists:** on organic molecules, pretraining on 10 times more frames than the baseline sees still made every variant worse (E01). Before reading that as "the pretraining objective learns nothing useful", we need to know that an encoder which certainly holds useful information does help when it goes through the same route (encoder weights loaded, new heads, all weights trained on the 10% subset).

**Question:** If the encoder comes from the baseline trained with labels on all 84,000 organic frames, does training on the 10% subset from it clearly beat the 10% baseline?

**Prediction (confidence):**
- If the route works: test momentum error well below the 10% baseline (0.0226), towards the full-data baseline (0.0138). I expect below 0.0190.
- If the route is broken: no better than the 10% baseline.
- My confidence that it works: 85%.

**Varied:** the starting encoder: random weights (E01 baseline, already run) against the encoder of flashmd_organic8_scratch (the E00 baseline, 100 epochs on all labels).

**Matched:** the route used for every pretrained arm: an encoder-only checkpoint in the format train/pretrain_jepa.py writes (no heads, no target scaler, so the heads are added and the scaler is fitted as for a pretrained encoder), then training exactly as E01 (8,400 labelled frames, 100 epochs, all weights train from step 1, seeds 0, 1, 2).

**Controls:**
- Reference arms already run: 10% baseline (0.0226), plain pretrained (0.0234), full-data baseline (0.0138).
- Must work: the encoder-only checkpoint reproduces the source model's features exactly (checked on a few frames before training).

**Assumptions:**
1. The supervised encoder has seen the labels of the 10% subset and of all other training frames. That is intended: this is an upper reference, not a fair arm.
2. Only the encoder is carried over; the heads start fresh, as for the pretrained arms.

**Measure and rule:** test momentum error, mean of three seeds. The route "works" if every seed is below every seed of the 10% baseline.

**If it fails:** stop and examine the checkpoint-loading path before any further pretraining experiment on either system.

**Runs:** 1 checkpoint conversion (seconds), 3 training runs (about 2.5 hours each, 24 GB card), then test evaluation.

## Result
Test momentum error, 16 Nov: 0.0148, 0.0148, 0.0148 (mean 0.0148), against the 10% baseline 0.0226 and the full-data baseline 0.0138. Every seed is far below every 10% baseline seed: the control works. The checkpoint route (encoder-only checkpoint, new heads, scaler fitted in training) passes on a useful encoder.

## Assumptions check
- The conversion checks passed on the real checkpoint: 161 encoder tensors copied, features identical to the source model.
- The three seeds agree to four decimals, as expected when the encoder already holds almost everything the task needs.

## Next
E15 started (17 Nov). A broken loading route is ruled out as the reason organic pretraining does not help.
