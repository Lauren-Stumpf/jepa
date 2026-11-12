# E06: does the water gain come from the regulariser rather than from prediction? (hypothesis B)

Status: agreed (12 Nov 2026)

**Question:** In E04 the two variants that helped on water (plain and normalised target) have predictors that never beat simply copying the current features: error ÷ copying error of 1.04 and 1.07 at the end of pretraining. If the prediction part learned nothing, is the gain coming from the regulariser alone?

**Hypothesis B:** the water gain comes from the regulariser, the part of the loss that keeps each feature varying across atoms and stops features duplicating each other. Predicting the future contributes nothing.

**Prediction (confidence):**
- If B is true: fine-tuning from a regulariser-only pretraining gives about the same error as from the plain pretraining (E04 plain: validation momentum 0.0268), well below the baseline (0.0311).
- If B is false: it lands near the baseline.
- My confidence that B is true: 65%.

**Varied:** the weight on the prediction term during pretraining: 25 (E04 plain) or 0 (this experiment). Nothing else in the pretraining changes.

**Matched:** the regulariser weights (25 on the variance term, 1 on the covariance term), the pretraining length (20 epochs, batch 8) and the fine-tuning (100 epochs, 10% subset).

**Fixed:**
- Water 16 fs data and model.
- Pretraining seed 0, as in E04.
- Fine-tuning seeds 0, 1, 2.
- Plain predictor (it receives no training signal when its weight is 0).
- Validation pairs shuffled with the fixed seed. In E04 they were not, which affects only the pretraining diagnostics, not the fine-tuning results.

**Controls:**
- Reference arms already run in E04: baseline (0.0311) and plain-pretrained (0.0268).
- Must work: the regulariser-only pretraining keeps its features spread out (no collapse). Checked from its effective rank.

**Assumptions:**
1. One pretraining seed is enough to tell 0.0268 from 0.0311, a 14% gap against a seed spread of about 2%. If the result lands in between, a second pretraining seed is needed.
2. With weight 0, the prediction term has no effect on the encoder. Checked: its share of the gradient push is logged as 0.

**Measure and rule:**
- Primary measure: validation and test momentum error, mean of three fine-tuning seeds, compared with E04 plain and E04 baseline.
- B is supported if regulariser-only is within three seed spreads of plain, and clearly below the baseline.

**Runs:** 1 pretraining (about 3.5 hours on a 24 GB card) and 3 fine-tunes (about 1 hour each on a 46 GB card). See `runs.txt`.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
