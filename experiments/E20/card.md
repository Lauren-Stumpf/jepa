# E20: organic, 10% of the labels: is the baseline still improving?

Status: proposed (18 Nov 2026). Lauren approved the close-out runs in principle; nothing is launched until she gives the go-ahead in writing.

**Why this exists:** E16 showed that on water with 10% of the labels, the pretraining gain was the baseline having fewer optimiser steps. On organic molecules pretraining never helped, and this needs no new run to settle: the 100-epoch baseline (52,500 steps) already beats E15's long-pretraining arms (78,900 steps) by the seed rule, with fewer steps (every baseline seed 0.0223 to 0.0231, every pretrained seed 0.0241 to 0.0253).

What is still open is why organic and water differ. The organic 10% baseline had 52,500 steps, nine times the water baseline's 5,600. If organic training was already close to finished at 100 epochs, extra steps from pretraining had nothing to add there, while on water they did. Related evidence: with all organic labels, 300 epochs gave no gain over 100 (E00: 0.0141 against 0.0138, one seed each).

**Question:** Does the organic 10% baseline still improve with five times as many steps?

**Prediction (confidence, Claude's):** the 500-epoch baseline's test error changes by less than 10% from the 100-epoch baseline's 0.0226 (no change or worse also counts): 55%. If it improves by 10% or more, "organic training was already close to finished" is wrong.

**What each arm gets** (rule of 18 Nov: if two compared arms differ in more than the one varied thing, it is not launched):

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|
| Baseline, 100 epochs (E01, already run) | 8,400 | 8,400 | 52,500 | | Organic test frames: molecules never seen in training |
| Baseline, 500 epochs (new) | 8,400 | 8,400 | 262,500 | Baseline, 100 epochs | Same |

What differs between compared arms: only the number of steps, and so the length of the learning-rate schedule.

**Varied:** number of training epochs (100 against 500).

**Matched:** the same 8,400 frames, batch 16, learning rate, schedule shape (the learning rate rises over the first 1% of training, then falls smoothly to zero by the end of the run), seeds 0, 1, 2, test command.

**Fixed:** model and every other setting, as in the source configs. The configs are copies made by scripts/make_e20.py; the only changes are the number of epochs and a checkpoint every 50 epochs instead of 10 (disk).

**Controls:** the 100-epoch baseline is already run (0.0226). No must-fail control: this experiment changes only the training length.

**Assumptions:**
1. metatrain keeps the epoch with the lowest product of validation position and momentum errors (checked every epoch) as model.pt, and model.pt is what is tested. So overfitting late in the run would show as an early saved epoch, not a worse model. Checked by looking at which epoch was saved.
2. The coarser checkpoint grid does not affect model.pt. The source configs mention selecting checkpoints afterwards with an energy criterion on a 10-epoch grid; no current result uses that, and if one ever does, this run's grid is coarser than E01's.

**Measure and rule:** test momentum error (mass-scaled root mean square error), mean of three seeds. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** experiments/E20/runs.txt. 3 runs of about 27 hours, 24 GB cards (the organic 10% baseline took about 5.3 hours per 100 epochs). Test evaluation: scripts/eval_test_organic_queue.sh (must be started again; it has finished).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
