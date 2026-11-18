# E21: water with all the labels: does the pretraining gain survive equal steps?

Status: proposed (18 Nov 2026). Lauren approved the close-out runs in principle; nothing is launched until she gives the go-ahead in writing.

**Why this exists:** in E17 (water, all 8,960 frames labelled) both pretrained arms beat the baseline by 12% (0.0061 against 0.0069), but they had 78,400 optimiser steps in total (22,400 pretraining + 56,000 labelled) against the baseline's 56,000. E16 showed that on water with 10% of the labels the whole gain was this difference in steps.

**Question:** With all the water labels, is a baseline with the same total steps as the E17 pretrained arms at least as good as them?

**Prediction (confidence, Claude's):** the 140-epoch baseline is not worse than either E17 pretrained arm (0.0061) by the seed rule: 75%. The water baseline on all labels at 300 epochs already reaches 0.0038 (E00, one seed), which points this way.

**What each arm gets:**

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|
| Baseline, 140 epochs (new) | 8,960 | 8,960 | 78,400 | E17 plain and no predictor: 8,960 / 8,960 / 22,400 + 56,000 = 78,400 | Water test frames: the same liquid, runs never seen in training |

What differs between compared arms: how the 78,400 steps are spent (all on labels, against pretraining then labels). Side differences that cannot be removed: the pretrained arms have two learning-rate schedules and the baseline one; a pretraining step uses 8 frames each encoded twice, a labelled step 16 frames.

**Varied:** how the steps are spent.

**Matched:** the same 8,960 frames, batch 16 for labelled training, learning rate, schedule shape, seeds 0, 1, 2, test command.

**Fixed:** everything else. The configs (scripts/make_e20.py) differ from E17's baseline configs only in the number of epochs (140 instead of 100) and the run names.

**Controls:** the 140-epoch run should land between the E17 100-epoch baseline (0.0069) and the E00 300-epoch baseline (0.0038). If it lands outside, something other than training length has changed.

**Assumptions:**
1. A pretraining step and a labelled step cost about the same compute (as in E16); the comparison is stated in steps.
2. One pretraining seed per E17 arm.
3. model.pt is the epoch with the lowest product of validation position and momentum errors, as for every other run.

**Measure and rule:** test momentum error, mean of three seeds. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** experiments/E21/runs.txt. 3 runs of about 10 hours, 46 GB cards (as E17). Test evaluation: scripts/eval_test_water16_queue.sh (must be started again; it has finished).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
