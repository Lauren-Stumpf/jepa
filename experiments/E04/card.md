# E04: water 16 fs, 10% of the labels, five pretraining variants against the baseline

Status: running (the residual, projector and relative-loss fine-tunes, and one baseline seed, were still running on 12 November). Card written afterwards.

**Question:** Same as E01, on water at the 16 fs jump.

**Varied:** starting encoder (baseline, or one of the five pretraining variants).

**Fixed:**
- 896 labelled pairs (every tenth forward pair with its time-reversed copy)
- 100 epochs (5,600 optimiser steps)
- Pretraining: 20 epochs on all 8,960 training pairs, batch 8 frames, chosen to match organic's atoms per batch
- Seeds 0, 1, 2

## Result so far
Validation momentum error, mean of seeds:

| Arm | Validation momentum | Against baseline |
|---|---|---|
| Baseline (2 seeds) | 0.0311 | |
| Plain | 0.0268 | 14% better |
| Normalised target | 0.0251 | 19% better |

Test evaluation not yet run.

## Assumptions check
- Only 5,600 optimiser steps, against about 52,000 for organic 10%. The organic lead faded after about 5,000 steps, so E05 tests whether this one lasts.
- In pretraining, the plain and normalised-target predictors never beat copying (error ÷ copying 1.04 and 1.07). E06 tests whether the gain comes from the regulariser alone.
