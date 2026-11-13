# E08: where do encoder features get their spread, and how much do they change over the gap? (hypothesis H2)

Status: agreed (13 Nov 2026). Measurement only, no training.

**Question:** Do organic and water encoders satisfy the regulariser in different ways? The regulariser demands that each of the 640 features varies across the atoms of a batch and that no two features duplicate each other. It does not say what the variation should describe.

**Hypothesis H2:** on organic molecules the encoder can meet the regulariser by describing what each atom is (element and chemical surroundings), which does not change over 8 fs, so the future features equal the present ones and there is nothing to predict. On water, with two elements in one chemistry, it has to describe the current geometry, which does change.

**Prediction (confidence):**
- If H2 is true: organic pretrained encoders have a "moment" share of the spread below 5% and change by less than 1% over 8 fs; water pretrained encoders have a "moment" share above 20% and change by more than 10% over 16 fs.
- If H2 is false: the two systems give similar shares for their pretrained encoders.
- My confidence that H2 is true: 45%.

**Measured:**
1. Spread split: each feature vector = element average + atom's own average over time (minus element average) + the rest (this moment). Shares of the total sum of squares; they add to exactly 1.
2. Change over the gap: mean squared change of an atom's features between a frame and 8 fs (and, for water, 16 fs) later, divided by the spread across atoms within one frame; also within each element.

**Encoders:**
- Organic: untrained; trained baseline (flashmd_organic8_scratch_10pct_s0); the five E01 pretrainings.
- Water: untrained; trained baseline (flashmd_water16_scratch_10pct_s0); the five E04 pretrainings; the regulariser-only pretraining (E06).

**Fixed:** frames no encoder was trained on. Organic: data/organic8/val.xyz, every 10th frame, which are forward frames only (1,050 frames, 50 per molecule, 21 molecules). Water: the raw runs of the validation split, a start every 100 fs (14 runs).

**Controls:**
- Must work: the three shares add to 1.000 for every encoder.
- Must work: the untrained water encoder's change at 16 fs is close to the 19% measured on 11 Nov (eval/feature_change_vs_jump.py, other runs and spacing, so within a few points).

**Assumptions:**
1. "Which atom" is defined within one molecule (organic) or one run (water). In water, molecules move around during a 2 ps run, so an atom's own average drifts towards its element's average; this is part of what H2 says about water, not a flaw, but the water atom share depends on run length.
2. Both systems are measured on forward frames only: water from raw runs, organic skipping the stored time-reversed copies (which have flipped momenta and would add spread that is not motion).

**Measure and rule:** the shares and changes per encoder, compared between organic and water pretrained encoders, with the untrained and trained baselines as references. No seeds: one encoder each.

**Runs:** 2 runs of eval/feature_decomposition.py (organic 7 encoders, water 8 encoders), minutes each on one GPU.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
