# E13: does water pretraining still help when it sees only the frames the baseline sees?

Status: proposed (16 Nov 2026). Not yet run.

**Why this exists (a confound in E04, E05, E06 and E11):** in those experiments the pretrained arm was pretrained on all 8,960 water training frames and then trained with labels on 896 of them, while the baseline saw only the 896. In this task the label is the frame 16 fs later, and pretraining also uses that later frame. So the pretrained arm saw the same kind of information as the labels for 10 times more frames. The water gain (12 to 28%) is confounded with extra data.

**Question:** If pretraining uses only the 896 frames the baseline uses, is the pretrained arm still better than the baseline?

**Prediction (confidence):**
- If the gain came from the pretraining objective: the pretrained arms still beat the baseline (0.0303 test momentum error) by about as much as before (plain 0.0268, no predictor about 0.0220 on validation).
- If the gain came from the extra frames: the pretrained arms land near the baseline.
- My confidence that less than half of the earlier gain remains: 55%.

**Varied:** the frames used for pretraining: all 8,960 (E04 plain, E11 no predictor: already run) against the 896 labelled frames only (this experiment).

**Arms (4 pretrainings, then 3 training seeds each):**

| Arm | Predictor | Pretraining frames | Pretraining length | Optimiser steps |
|---|---|---|---|---|
| plain-same-epochs | plain | 896 | 20 epochs | 2,240 |
| plain-same-steps | plain | 896 | 200 epochs | 22,400 |
| nopred-same-epochs | none | 896 | 20 epochs | 2,240 |
| nopred-same-steps | none | 896 | 200 epochs | 22,400 |

Two lengths because both are defensible ways to match E04 (20 epochs, 22,400 steps on 8,960 frames): the same number of passes over the data, or the same number of optimiser steps.

**Matched:** everything else as E04 plain and E11: batch 8 frames, prediction weight 25, variance weight 25, covariance weight 1, pretraining seed 0, validation on data/water/pairs/16fs/val.xyz; training with labels exactly as E04 (896 frames, 100 epochs, seeds 0, 1, 2).

**Fixed:** pretraining file data/water/pairs/16fs/train_10pct.xyz (the 896 frames the labelled training uses; it holds each forward pair with its time-reversed copy).

**Controls:**
- Reference arms already run: baseline (E04, test 0.0303), plain on all frames (E04, 0.0268), no predictor on all frames (E11, validation 0.0220; its test evaluation is part of this work), regulariser only (E06, 0.0299).
- Must work: no pretraining collapses (effective rank, dead-feature fraction in the log).
- Watch: with 200 epochs on 896 frames the pretraining can overfit; its loss on the validation frames is logged every epoch and is reported with the result.

**Assumptions:**
1. One pretraining seed per arm, as in E04, E06 and E11.
2. The learning-rate schedule (1% warm-up, cosine decay) keeps its shape; its length follows the number of steps.

**Measure and rule:** test momentum error, mean of three seeds, against the baseline and against the same variant pretrained on all frames. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** 4 pretrainings (about 30 minutes each at 20 epochs, about 5 hours each at 200 epochs, 24 GB card; the full validation set is evaluated every epoch) and 12 training runs (about 1 hour each, 46 GB card), then test evaluation.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
