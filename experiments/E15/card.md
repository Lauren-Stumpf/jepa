# E15: organic pretraining on only the frames the baseline sees (3 seeds)

Status: proposed (16 Nov 2026). Not yet run.

**Why this exists:** the only same-frames comparison we have on organic molecules is E02 (all labels, one variant, one seed: 4% worse). E01 gave pretraining 10 times more frames than the baseline and it was still worse. This experiment fills the "same frames" cell for organic properly, and it is the organic twin of E13. It also tries the no-predictor objective on organic for the first time.

**Question:** With pretraining restricted to the 8,400 frames the baseline trains on, how do the pretrained arms compare with the baseline?

**Prediction (confidence):**
- Pretrained arms no better than the baseline (0.0226 test momentum error): 80% confident.
- No predictor at least as good as plain: 60% confident.

**Varied:** the frames used for pretraining: all 84,000 (E01 plain: already run) against the 8,400 labelled frames only (this experiment); and the predictor (plain against none).

**Arms (4 pretrainings, then 3 training seeds each):**

| Arm | Predictor | Pretraining frames | Pretraining length | Optimiser steps |
|---|---|---|---|---|
| plain-same-epochs | plain | 8,400 | 20 epochs | 2,640 |
| plain-same-steps | plain | 8,400 | 200 epochs | 26,400 |
| nopred-same-epochs | none | 8,400 | 20 epochs | 2,640 |
| nopred-same-steps | none | 8,400 | 200 epochs | 26,400 |

**Matched:** everything else as E01 plain: batch 64 frames, weights 25/25/1, pretraining seed 0, validation on data/organic8/val.xyz; training with labels exactly as E01 (8,400 frames, 100 epochs, seeds 0, 1, 2).

**Fixed:** pretraining file data/organic8/train_10pct.xyz.

**Controls:**
- Reference arms already run: baseline (E01, 0.0226), plain on all frames (E01, 0.0234).
- E14 must have passed first (the route can pass on a benefit).
- Must work: no pretraining collapses.

**Assumptions:**
1. The 10% organic subset holds forward frames only (no time-reversed copies), unlike the full file E01 pretrained on. So these pretrainings differ from E01's in two ways: fewer frames and no reversed copies. Recorded, not corrected, because the baseline's labelled frames are what "same frames" means.
2. One pretraining seed per arm.

**Measure and rule:** test momentum error, mean of three seeds, against the baseline. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** 4 pretrainings (about 40 minutes each at 20 epochs, about 7 hours each at 200 epochs, 24 GB card) and 12 training runs (about 2.5 hours each, 24 GB card), then test evaluation.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
