# E01: organic molecules, 10% of the labels, five pretraining variants against the baseline

Status: done. Card written afterwards. The prediction below is what we expected at the time, not a forecast written down before running.

**Question:** Does JEPA pretraining of the encoder lower FlashMD's error when labels are scarce?

**Varied:** starting encoder. Arms:
- baseline (random weights)
- plain, normalised target, residual, projector, relative loss: one pretraining each, 20 epochs on all 84,000 training pairs without labels

**Matched / fixed:**
- 8,400 labelled pairs (every tenth pair, all 168 training molecules)
- 100 epochs, identical fine-tuning settings
- All weights train from step 1
- Seeds 0, 1, 2 (the three seeds of an arm share one pretraining)

**Expected:** pretrained arms better than the baseline.

**Measure:** validation and test momentum error, mean of three seeds.

## Result
Every variant is worse than the baseline, on validation and on test.

| Arm | Test momentum | Against baseline |
|---|---|---|
| Baseline | 0.0226 | |
| Plain | 0.0234 | 4% worse |
| Relative loss | 0.0236 | 4% worse |
| Normalised target | 0.0241 | 7% worse |
| Residual | 0.0241 | 7% worse |
| Projector | 0.0269 | 19% worse |

Pretrained arms lead early, with about half the baseline's validation error after epoch 1. The baseline catches up by about epoch 10 and stays ahead.

## Assumptions check
- The 10% subset covers the same 168 molecules that pretraining saw, so pretraining added no new molecules.
- Validation diagnostics during pretraining were distorted by batches that held one molecule each. Fixed on 11 November.
