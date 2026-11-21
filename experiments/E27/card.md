# E27: water, longer runs: does the pretraining lead grow until it pays back its cost?

Status: agreed 21 Nov 2026 (Lauren: "finish e21 e22 and then 1,2,3,5", "write review and run").

**Question:** at 134,400 total optimiser steps, does the pretrained model beat a baseline given the same total steps?

**Why:** at equal labelled steps the plain pretrained model is ahead by an amount equal to 1.2 to 1.3 times the labelled steps. 22,400 pretraining steps bought back about 1,200 labelled steps in the 100-epoch run and about 18,100 in the 1,000-epoch run (E16), where the two arms came out level. If the lead keeps growing in proportion, pretraining wins in a longer run. If it is a fixed lead, it never does.

**Tested on:** water test frames (the same liquid as training, runs never seen), 10% of the labels (896 frames).

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|
| Baseline, 2,400 epochs | 896 | 896 | 134,400 | plain | water test frames |
| Plain pretrained (the E04 encoder), then 2,000 epochs | 8,960 without labels, then 896 | 896 | 22,400 + 112,000 = 134,400 | baseline | water test frames |

What does each arm get that the other does not? The pretrained arm spends its first 22,400 steps on pretraining over 8,960 frames, and has two warm-up and decay cycles of the learning rate; the baseline spends those steps on labels and has one cycle. E13 showed pretraining on the 896 labelled frames alone gives the same result, so the extra frames are not expected to matter.

**Prediction (confidence, Claude's):** the two arms overlap (no winner by the seed rule): 55%. Pretrained wins: 25%. Baseline wins: 20%. A proportional lead of 1.3 predicts the pretrained arm about 5% better; a fixed lead predicts it about 2% worse.

**Controls:** the 2,400-epoch baseline must be better than the 1,400-epoch baseline (0.0061, E16). If not, the baseline has stopped improving and the comparison is between finished models.

**Assumptions:** one pretraining seed. A learning rate that decays over the run means a longer run is not the first part of a shorter one.

**Measure and rule:** test momentum error, 3 seeds per arm; one arm wins when every seed of it beats every seed of the other.

**Runs:** experiments/E27/runs.txt, 6 runs of about 22 to 26 hours on 46 GB cards (torrnode15).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
