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
Validation only so far (16 Nov; the test evaluation is still to run). Validation momentum error, three seeds:

| Arm | Seeds | Mean | Against baseline |
|---|---|---|---|
| Baseline (E04) | 0.0309, 0.0286, 0.0312 | 0.0303 | |
| Regulariser only (E06) | 0.0317, 0.0291, 0.0289 | 0.0299 | 1% better |
| Plain (E04) | 0.0274, 0.0260, 0.0271 | 0.0268 | 12% better |
| No predictor (this experiment) | 0.0218, 0.0218, 0.0223 | 0.0220 | 27% better |

Every no-predictor seed is below every seed of plain and of the baseline. The prediction "lands in the plain range" was too cautious: it is better than plain.

## Assumptions check
- The pretraining did not collapse (effective rank 507, no dead features) and its prediction-over-copying ratio was 1.000, as required.
- Found on 16 Nov, applies to this experiment and to E04, E05, E06: the pretraining used all 8,960 training frames, the baseline 896. Since the label is the later frame and pretraining uses the later frame too, the pretrained arm saw 10 times more data. The gain here is confounded with extra data. E13 repeats this with pretraining on the 896 frames only.

## Next
Test evaluation; E13.
