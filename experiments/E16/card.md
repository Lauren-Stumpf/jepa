# E16: does the water gain survive when the baseline gets the same number of training steps?

Status: done (results 18 Nov 2026).

**Why this exists:** E13 showed the water gain is not from extra frames, but it needs long pretraining (22,400 optimiser steps; with 2,240 there is no gain). The pretrained arms therefore train for 22,400 + 5,600 = 28,000 steps while the baseline trains for 5,600. We know the baseline improves a lot with more steps (0.0303 at 100 epochs, 0.0074 at 1,000). So the gain may be extra training, not pretraining.

**Question:** If the baseline is given the same total number of optimiser steps, all of them on labels, is pretraining still better?

**Prediction (confidence):**
- Baseline with 500 epochs (28,000 steps) beats the pretrained arms that total 28,000 steps (E13: plain 0.0260, no predictor 0.0234): 85% confident. I expect about 0.010.
- Baseline with 1,400 epochs (78,400 steps) is at least as good as plain pretraining plus 1,000 epochs (E05: 0.0063, also 78,400 steps): 60% confident.

**Varied:** how the optimiser steps are spent: all on labelled training (this experiment) against pretraining then labelled training (E13, E05).

**Arms:**

| Arm | Labelled epochs | Total steps | Compared with |
|---|---|---|---|
| baseline-500ep | 500 | 28,000 | E13 plain and no predictor, 200 pretraining epochs + 100 labelled epochs (28,000 steps) |
| baseline-1400ep | 1,400 | 78,400 | E05 plain: pretraining on all frames (22,400) + 1,000 labelled epochs (56,000) |

**Matched:** the 896 labelled frames, batch 16, learning rate, schedule shape (1% warm-up, cosine decay over the run's own length), seeds 0, 1, 2. Checkpoints every 50 epochs (500) and every 100 epochs (1,400): saving does not affect training, but the grid is coarser than the 10-epoch grid of the E13 arms, which matters only if a checkpoint is ever selected afterwards.

**Controls:** reference arms already run: baseline 100 epochs (0.0303), baseline 1,000 epochs (0.0074).

**Assumptions:**
1. A pretraining step (8 frames, each encoded now and later) and a labelled step (16 frames) cost about the same compute, so equal steps is close to equal compute.
2. The E05 comparison uses pretraining on all 8,960 frames; E13 showed frames are not what matters, so it is used as it is.
3. Each arm's learning-rate schedule is stretched over its own length; a pretrained arm has two schedules (pretraining, then labelled). This cannot be matched exactly.

**Measure and rule:** test momentum error, mean of three seeds. A difference counts if every seed of one arm beats every seed of the other.

**Runs:** 3 runs of about 5 hours and 3 of about 15 hours, 46 GB cards.

## Result
Test momentum error, tested on water test frames (the same liquid as training, in runs never seen), 3 seeds each:

| Arm | Total steps | Seeds | Mean | Compared with | Seed rule |
|---|---|---|---|---|---|
| Baseline, 100 epochs (E04) | 5,600 | 0.0310, 0.0286, 0.0313 | 0.0303 | | |
| **Baseline, 500 epochs** | 28,000 | 0.0105, 0.0106, 0.0109 | **0.0107** | E11 no predictor on all frames, 22,400 + 5,600 steps: 0.0220 (the best pretrained arm with three seeds at 28,000 steps; E13 no predictor on the same frames: 0.0234) | baseline better |
| Baseline, 1,000 epochs (E05) | 56,000 | 0.0079, 0.0073, 0.0070 | 0.0074 | | |
| **Baseline, 1,400 epochs** | 78,400 | 0.0063, 0.0061, 0.0060 | **0.0061** | E05 plain, pretraining + 1,000 epochs: 0.0063 (0.0062 to 0.0064) | overlap: equal |

Both predictions held: the 500-epoch baseline beats every pretrained arm with the same total steps (predicted about 0.010, got 0.0107), and the 1,400-epoch baseline is at least as good as E05 plain.

With the same number of optimiser steps, spending them on labelled training is better (at 28,000) or the same (at 78,400) as spending some on pretraining. The water gain in E04, E11 and E13 was the baseline having fewer steps, not a benefit of pretraining. The 100-epoch water baseline was far from finished.

## Assumptions check
- Assumption 1 (a pretraining step and a labelled step cost about the same compute) was not measured; it does not affect the conclusion, which is stated in steps.
- Assumption 3 (each arm's learning-rate schedule stretched over its own length) cannot be matched exactly; at 78,400 steps the two arms are equal, so it does not hide a difference there.

## Next
Close out the same question on organic (E20) and on water with all labels (E21). Then move to the end goal, latent rollout, starting with a rollout of the water baseline (E22). Decision of 18 Nov: one-step pretraining is kept only as a possible part of the rollout design, not developed on its own for now.
