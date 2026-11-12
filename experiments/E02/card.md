# E02: organic molecules, all the labels, pretraining against the baseline

Status: done. Card written afterwards. One seed.

**Varied:** starting encoder (random, or normalised-target pretrained).

**Fixed:** all 84,000 labelled pairs, 100 epochs.

**Not matched:** total compute. Pretraining cost about 50 baseline epochs.

## Result
| Run | Validation momentum | Test momentum |
|---|---|---|
| Baseline, 100 epochs (E00) | 0.0154 | 0.0138 |
| Pretrained, then 100 epochs | 0.0149 | 0.0144 |

The pretrained run is 3.7% better on validation and 4.3% worse on test. With one seed and a validation/test disagreement of this size, there is no effect to report. Not pursued further (decision of 11 November).
