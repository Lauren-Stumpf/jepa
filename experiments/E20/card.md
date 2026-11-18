# E20: close-out: was the pretraining gain only extra training steps?

Status: proposed (18 Nov 2026). Lauren approved running it in principle; not launched until she gives the go-ahead in writing.

**Why this exists:** E16 showed that on water with 10% of the labels, a baseline given the same number of optimiser steps as the pretrained arms is better (28,000 steps: 0.0107 against 0.0234 at best) or equal (78,400 steps: 0.0061 against 0.0063). Two cells are still open before the one-step pretraining question can be closed:

1. **Organic.** The organic 10% baseline had 52,500 steps, nine times the water baseline's 5,600. If organic training was already close to finished at 100 epochs, the extra steps pretraining adds had nothing to give, which would explain why pretraining never helped there.
2. **Water, all labels (E17).** The pretrained arms beat the baseline by about 12% (0.0061 against 0.0069), but they had 78,400 steps in total against the baseline's 56,000.

**Question:**
1. Does the organic 10% baseline improve when trained for five times as many steps?
2. With all the water labels, is a baseline with the same total steps as the pretrained arms as good as them?

**Prediction (confidence):**
- Organic 10% baseline at 500 epochs improves on 0.0226 by less than 10%: 55% confident. If it improves by 10% or more, the explanation "organic training was already close to finished" is wrong.
- Water, all labels, 140 epochs: at least as good as the E17 pretrained arms (0.0061): 75% confident. The water baseline on all labels at 300 epochs already reaches 0.0038 (one seed), which points this way.

**What each arm gets** (the check agreed on 18 Nov: if two compared arms differ in more than the one varied thing, it is not launched):

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|
| Organic baseline, 500 epochs (new) | 8,400 | 8,400 | 262,500 | Organic baseline, 100 epochs (E01): 8,400 / 8,400 / 52,500 | Organic test frames: molecules never seen in training |
| Water all labels, baseline 140 epochs (new) | 8,960 | 8,960 | 78,400 | E17 plain and no-predictor pretrained: 8,960 / 8,960 / 22,400 + 56,000 = 78,400 | Water test frames: the same liquid, runs never seen in training |

What differs between compared arms:
- Organic: only the number of steps (and so the length of the learning-rate schedule).
- Water: only how the 78,400 steps are spent (all on labels, against pretraining then labels).

**Varied:** organic: number of epochs (100 against 500). Water: how the steps are spent.

**Matched:** same frames, batch 16, learning rate, schedule shape (warm-up, cosine decay over the run's own length), seeds 0, 1, 2, test command.

**Fixed:** model architecture and every other setting, as in the source configs. The configs are copies made by scripts/make_e20.py; the only changes are the number of epochs, how often a checkpoint is saved (organic only, for disk), and for water the training file (all frames, exactly as E17's baseline).

**Controls:** reference arms already run: organic baseline 100 epochs (E01, 0.0226); water all-labels baseline at 100 epochs (E17, 0.0069) and at 300 epochs (E00, 0.0038, one seed). The 140-epoch water run should land between those two.

**Assumptions:**
1. A pretraining step and a labelled step cost about the same compute (as in E16).
2. Organic: overfitting is possible with 8,400 frames over 500 epochs. metatrain keeps the model with the best validation loss, so overfitting late in training would show as an early best epoch, not as a worse model. Checked by looking at the epoch of the saved model.
3. One pretraining seed per arm in E17; recorded, not corrected.

**Measure and rule:** test momentum error (mass-scaled root mean square error), mean of three seeds. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** experiments/E20/runs.txt. 3 organic runs of about 27 hours (24 GB card is enough; the organic 10% runs took about 5.3 hours per 100 epochs) and 3 water runs of about 10 hours (46 GB card, as E17). Test evaluation is in the existing queues (scripts/eval_test_organic_queue.sh and scripts/eval_test_water16_queue.sh), which must be started again because both have finished.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
