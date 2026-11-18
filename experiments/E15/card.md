# E15: organic pretraining on only the frames the baseline sees (3 seeds)

Status: done (results 18 Nov 2026).

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

**Runs:** 4 pretrainings (about 1 hour each at 20 epochs, about 9 hours each at 200 epochs, 24 GB card; the full validation set is evaluated every epoch) and 12 training runs (about 2.5 hours each, 24 GB card), then test evaluation.

## Result
Test momentum error, tested on organic test frames (molecules never seen in training), 3 training seeds each:

| Arm | Seeds | Mean | Against baseline | Seed rule against baseline |
|---|---|---|---|---|
| Baseline (E01) | 0.0223, 0.0224, 0.0231 | 0.0226 | | |
| Plain, 20 epochs (2,640 steps) | 0.0222, 0.0223, 0.0225 | 0.0223 | 1% better | overlap |
| No predictor, 20 epochs | 0.0226, 0.0226, 0.0227 | 0.0226 | equal | overlap |
| Plain, 200 epochs (26,400 steps) | 0.0248, 0.0243, 0.0243 | 0.0245 | 8% worse | worse |
| No predictor, 200 epochs | 0.0243, 0.0253, 0.0241 | 0.0246 | 9% worse | worse |

Prediction 1 ("pretrained arms no better than the baseline", 80%) held. Prediction 2 ("no predictor at least as good as plain", 60%) failed at 20 epochs: every plain seed (0.0222 to 0.0225) beats every no-predictor seed (0.0226 to 0.0227), a 1% difference. At 200 epochs it held: the seeds overlap.

Short pretraining on organic changes nothing against the baseline; long pretraining makes the model worse. The predictor makes at most a 1% difference. This is the opposite of water, where long pretraining was needed for any gain (E13).

## Assumptions check
- Collapse of the pretrained encoders was not checked directly.
- Assumption 1 (the 10% organic subset has no time-reversed copies) still stands as a difference from E01; E15 and E01 agree that pretraining does not help, so it does not change the conclusion.
- One pretraining seed per arm.

## Next
E16 showed the water gain was extra optimiser steps. The organic 10% baseline already had 52,500 steps, nine times the water baseline, so one explanation of this result is that organic training was already close to finished. E20 tests this with organic baselines of 150 epochs (the same total steps as the 200-epoch pretrained arms) and 500 epochs.
