# E11: does the water gain come from pulling features now and later together, without a predictor? (hypothesis H5)

Status: agreed (13 Nov 2026).

**Question:** On water, four pretraining variants help although their predictors score no better than copying the current features, and the regulariser alone (E06) gives no gain. Is the gain produced by the prediction term pulling an atom's features now and after the gap towards each other, rather than by the predictor learning the motion?

**Hypothesis H5:** most of the water gain comes from features that change slowly in time (pulled together by the prediction term) while differing between atoms (pushed by the regulariser). A learned predictor adds little.

**Prediction (confidence):**
- If H5 is true: fine-tunes from the no-predictor pretraining land in the plain range (E04 plain test momentum 0.0268, seeds 0.0260 to 0.0274), with every seed below every baseline seed (baseline seeds 0.0286 to 0.0313).
- If H5 is false: they land near the baseline (0.0303) like the regulariser-only arm (0.0299).
- My confidence that H5 is true: 60%.

**Varied:** the predictor during pretraining: a learned network (E04 plain) or none (the "guess" is the current features themselves, so the prediction term is the distance between features now and 16 fs later).

**Matched:** everything else as E04 plain: prediction weight 25, variance weight 25, covariance weight 1, 20 epochs, batch 8 water frames, pretraining seed 0, the same data; fine-tuning 100 epochs on the 10% subset, seeds 0, 1, 2.

**Controls:**
- Reference arms already run: baseline (E04, test momentum 0.0303), plain (E04, 0.0268), regulariser only (E06, 0.0299).
- Must work: the pretraining does not collapse (effective rank and dead-feature fraction logged), and its prediction-over-copying ratio is exactly 1.00 (no predictor).

**Assumptions:**
1. Removing the predictor also removes its 2.4 million parameters and their share of the gradient clipping norm (clipped to 1 over all parameters); this comes with the change and is not a separate one.
2. One pretraining seed, as in E04 and E06.
3. A plain predictor that "scores no better than copying" is not the same as an identity map; this experiment tests whether the difference matters.

**Measure and rule:** test momentum error, mean of three fine-tuning seeds, against the three reference arms. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** 1 pretraining (about 3.5 hours on a 24 GB card) and 3 fine-tunes (about 1 hour each on a 46 GB card). See `runs.txt`. Test evaluation as for E04 to E06.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
