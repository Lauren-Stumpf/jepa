# E20: organic, was the baseline already finished, and does pretraining lose with equal steps?

Status: proposed (18 Nov 2026). Lauren approved the close-out runs in principle; nothing is launched until she gives the go-ahead in writing.

**Why this exists:** E16 showed that on water with 10% of the labels, the pretraining gain was the baseline having fewer optimiser steps. On organic molecules pretraining never helped: E15's arms with 200 pretraining epochs were 8 to 9% worse than the 100-epoch baseline. Two things are open:

1. E15's long-pretraining arms had 78,900 optimiser steps (26,400 pretraining + 52,500 labelled) against the baseline's 52,500. The organic version of E16 (baseline with the same total steps) has not been run.
2. The organic 10% baseline already had 52,500 steps, nine times the water baseline's 5,600. If organic training was close to finished at 100 epochs, the extra steps from pretraining had nothing to add. Related evidence: with all organic labels, 300 epochs gave no gain over 100 (E00: 0.0141 against 0.0138, one seed each).

**Question:**
1. With the same total steps as E15's long-pretraining arms, is the organic baseline at least as good as them?
2. Does the organic 10% baseline still improve with five times as many steps?

**Prediction (confidence, Claude's):**
- 150-epoch baseline not worse than E15's 200-epoch pretrained arms (0.0245, 0.0246) by the seed rule: 85%.
- 500-epoch baseline improves on the 100-epoch baseline (0.0226) by less than 10%: 55%. If it improves by 10% or more, "organic training was already close to finished" is wrong.

**What each arm gets** (rule of 18 Nov: if two compared arms differ in more than the one varied thing, it is not launched):

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|
| Baseline, 100 epochs (E01, already run) | 8,400 | 8,400 | 52,500 | | Organic test frames: molecules never seen in training |
| Baseline, 150 epochs (new) | 8,400 | 8,400 | 78,750 | E15 plain and no predictor, 200 pretraining epochs: 8,400 / 8,400 / 26,400 + 52,500 = 78,900 | Same |
| Baseline, 500 epochs (new) | 8,400 | 8,400 | 262,500 | Baseline, 100 epochs | Same |

What differs between compared arms:
- 150-epoch baseline against E15: how the steps are spent (all on labels, against pretraining then labels). Two side differences come with this and cannot be removed: the pretrained arms have two learning-rate schedules (one for pretraining, one for labelled training), the baseline one; and a pretraining step uses 64 frames each encoded twice, a labelled step 16 frames.
- 500-epoch against 100-epoch baseline: only the number of steps, and so the length of the learning-rate schedule.

**Varied:** number of training epochs (100, 150, 500).

**Matched:** the same 8,400 frames, batch 16, learning rate, schedule shape (the learning rate rises over the first 1% of training, then falls smoothly to zero by the end of the run), seeds 0, 1, 2, test command.

**Fixed:** model and every other setting, as in the source configs. The configs are copies made by scripts/make_e20.py; the only changes are the number of epochs and, for 500 epochs, a checkpoint every 50 epochs instead of 10 (disk).

**Controls:** the 100-epoch baseline is already run (0.0226). No must-fail control: this experiment changes only the training length, and the training route itself is not in question.

**Assumptions:**
1. metatrain keeps the epoch with the lowest product of validation position and momentum errors (checked every epoch) as model.pt, and model.pt is what is tested. So overfitting late in a 500-epoch run would show as an early saved epoch, not a worse model. Checked by looking at which epoch was saved.
2. The coarser checkpoint grid (every 50 epochs) does not affect model.pt. It would matter only if checkpoints were selected afterwards by another criterion, which no current result does.
3. One pretraining seed per E15 arm; recorded, not corrected.

**Measure and rule:** test momentum error (mass-scaled root mean square error), mean of three seeds. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** experiments/E20/runs.txt. 3 runs of about 8 hours (150 epochs) and 3 of about 27 hours (500 epochs), 24 GB cards; the organic 10% baseline took about 5.3 hours per 100 epochs. Test evaluation: scripts/eval_test_organic_queue.sh (must be started again; it has finished).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
