# E09: does organic pretraining lose everywhere, or on unusual frames? (hypothesis H3, extreme frames)

Status: agreed (13 Nov 2026). Analysis of existing results only.

**Question:** The organic data include frames where the molecule has come apart (5,582 of 105,000) and frames where atoms see their own periodic image (1,115). Pretraining used all of them. Are the pretrained arms' losses against the baseline concentrated on these frames, or on hot frames?

**Hypothesis H3 (extreme frames):** rare unusual frames let the encoder meet the regulariser cheaply (features that flag "unusual frame") instead of describing ordinary motion, or they make pretraining learn things that do not carry over.

**Prediction (confidence):**
- If true: on normal frames the pretrained arms are within half of their overall deficit (for example plain within 2% of the baseline instead of 4% worse), and the deficit is much larger on unusual frames.
- If false: the percentage against the baseline is about the same on every kind of frame.
- My confidence: 20%.

**Varied:** the kind of frame (normal, apart, own image) and the temperature band (below 500 K, 500 to 1,000 K, above 1,000 K).

**Fixed:** the 18 E01 models and their existing per-frame results: test set (results/eval/test) and the E07 unused frames (results/eval/seen).

**Controls:**
- Must work: each evaluated frame is matched to its frames.csv row, checked by formula on every frame.
- Reference: the overall error per arm reproduces the published E01 test numbers (baseline 0.0226).

**Assumptions:**
1. Few unusual frames are in each evaluated set, so their errors are noisy; frame counts are reported with the result and a category with fewer than 50 frames is not read.
2. Errors are pooled over atoms as in the evaluation.

**Measure and rule:** pooled momentum error per arm and category, mean of three seeds, percentage against the baseline in the same category. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** eval/errors_by_frame_type.py on the test set and on the unused frames (after E07 finishes).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
