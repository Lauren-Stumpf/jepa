# E05: does the water gain last with ten times more training? (hypothesis A)

Status: agreed (12 Nov 2026)

**Question:** In E04, pretraining lowered water validation momentum error by 14% (plain) and 19% (normalised target) after 100 epochs. Does that gain last when training is ten times longer?

**Hypothesis A (the one being tested):** the gain is an early lead that fades. On organic molecules (E01), pretrained models had about half the baseline's error after one epoch, and the baseline caught up within about 5,000 optimiser steps. The E04 water runs stop at 5,600 steps.

**Prediction (confidence):**
- If A is true: at 1,000 epochs (56,000 steps) the gap between pretrained and baseline is within the seed spread.
- If A is false: the pretrained arms are still more than 5% better.
- My confidence that A is true: 60%.

**Varied:** the starting encoder. Arms:
- baseline (random weights)
- plain-pretrained
- normalised-target-pretrained

These use the existing E04 pretrainings.

**Matched:**
- The same number of epochs (1,000) and optimiser steps for every arm.
- The same learning-rate schedule shape: 1% warm-up, then cosine decay to zero over 1,000 epochs.

**Fixed:**
- Data: 896 labelled pairs (10% subset) and the same validation and test sets.
- Model: PET backbone, cutoff 4.5 Å.
- Batch 16, learning rate 0.0003, random rotations, mass-scaled loss.
- Seeds 0, 1, 2.
- Pretrained encoders taken unchanged from E04.

**Controls:**
- Must work: the 300-epoch full-data water baseline (test momentum error 0.0038). The 10% runs should land between this and the straight-line guess.
- Must fail: the straight-line guess (test momentum error 0.204).

**Assumptions:**
1. 1,000 epochs on 896 pairs does not simply overfit. Checked by watching whether validation error starts rising while training error keeps falling.
2. The E04 pretrainings are representative. Only one pretraining seed per variant, as in E04.
3. The 100-epoch and 1,000-epoch schedules differ only in length. The cosine shape is the same.
4. Checkpoints are kept every 50 epochs instead of every epoch, to save disk (a checkpoint is about 430 MB). This does not change training.

**Measure and rule:**
- Primary measure: test momentum error, mean of three seeds.
- A difference counts if it is larger than three times the larger seed spread.
- Also reported: validation curves, to see when (if ever) the gap closes.

**Runs:** 9 (3 arms × 3 seeds), about 10.5 hours each on a 46 GB card. See `runs.txt`.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
