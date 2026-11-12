# E04: water 16 fs, 10% of the labels, five pretraining variants against the baseline

Status: running (the residual, projector and relative-loss fine-tunes, and one baseline seed, were still running on 12 November). Card written afterwards.

**Question:** Same as E01, on water at the 16 fs jump.

**Varied:** starting encoder (baseline, or one of the five pretraining variants).

**Fixed:**
- 896 labelled pairs (every tenth forward pair with its time-reversed copy)
- 100 epochs (5,600 optimiser steps)
- Pretraining: 20 epochs on all 8,960 training pairs, batch 8 frames, chosen to match organic's atoms per batch
- Seeds 0, 1, 2

## Result
Validation momentum error, mean of seeds:

| Arm | Validation momentum | Against baseline |
|---|---|---|
| Baseline (2 seeds) | 0.0311 | |
| Plain | 0.0268 | 14% better |
| Normalised target | 0.0251 | 19% better |

Test set (all 1,120 water test frames, scripts/eval_test_water16_queue.sh), 12 Nov:

| Arm | Test momentum | Test position | Against baseline (momentum) |
|---|---|---|---|
| Baseline (seeds 0, 2) | 0.0312 (0.0310, 0.0313) | 0.0217 | |
| Plain (3 seeds) | 0.0268 (0.0274, 0.0260, 0.0271) | 0.0183 | 14% better |
| Normalised target (3 seeds) | 0.0252 (0.0246, 0.0253, 0.0257) | 0.0177 | 19% better |
| Residual (seed 0 only) | 0.0249 | 0.0164 | 20% better |
| Projector (seed 0 only) | 0.0291 | 0.0202 | 7% better |
| Residual, relative loss (seed 0 only) | 0.0213 | 0.0145 | 32% better |

Test agrees with validation. Unlike organic molecules (E01, where every variant was worse
than the baseline on test), every water variant is better. The gap to the baseline is about
3 times the seed spread for plain (0.0044 against 0.0014) and 5 times for normalised target
(0.0060 against 0.0011), so both pass the 3-times rule.

The residual-relative-loss variant is the only one whose water predictor clearly beat
copying in pretraining (error ÷ copying 0.085), and on one seed it gives the largest gain.
That is evidence against hypothesis B (E06): prediction may add to whatever the regulariser
gives. Seeds 1 and 2 are running; do not read it before they finish.

## Assumptions check
- Only 5,600 optimiser steps, against about 52,000 for organic 10%. The organic lead faded after about 5,000 steps, so E05 tests whether this one lasts.
- In pretraining, the plain and normalised-target predictors never beat copying (error ÷ copying 1.04 and 1.07). E06 tests whether the gain comes from the regulariser alone.
