# E26: do the water lead and the organic loss come from what the encoder learned, or from the learning rate and the sizes of the starting weights?

Status: organic part agreed 21 Nov 2026 (see the update below); the rest proposed.

**Why this exists:** every run so far, baseline or pretrained, used one learning rate (0.0003, the value in the FlashMD paper) and one schedule. Two readings in the deep dive of 20 Nov (https://claude.ai/artifact/9SqGJY3qrzteSPSwm5ySJ5) depend on that never-tested choice.
- On water, pretrained models are 12 to 28% ahead of the baseline at equal labelled steps. The deep dive reads this as faster early training caused by the pretrained encoder. It could instead be a baseline trained at a poor learning rate, or an effect of the sizes of the pretrained weights, layer by layer, and not of what they compute (hypothesis 7 in notes/decisions.md, 20 Nov).
- On organic molecules, pretrained models end 4 to 19% worse. Labelled training that starts from a pretrained encoder is normally run at a lower learning rate than training from random weights; ours was not (explanation 4a).

**Update, 21 Nov:** after the independent review of the designs, only the organic part is launched, with three learning rates (0.0001 and 0.001 new, 0.0003 existing) for the baseline and for plain: experiments/E26/runs.txt, 12 runs of about 5.3 hours on 24 GB cards. The water sweep is dropped (no outcome would change a decision). The weight-size arm waits for an untrained start passed through the same route. Launched 21 Nov on Lauren's instruction ("finish e21 e22 and then 1,2,3,5", "write review and run").

**Question:** when the baseline and the pretrained arm each get their own best learning rate, does the water lead remain and does the organic loss remain? And does a random start that keeps the pretrained weights' sizes reproduce the water lead?

**Hypothesis:** the water lead belongs to what the encoder learned, so it survives best-against-best and is not reproduced by weight size alone. The organic loss is not a recipe effect, so it survives the lower learning rate.

## Water arms

10% of the labels (896 frames), 100 labelled epochs (5,600 steps), 3 seeds each. Comparisons here are at equal labelled steps, because the question is where the lead at equal labelled steps comes from. The result at equal total steps (E16) is not reopened.

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|
| Baseline at learning rate 0.0001, 0.0003 (exists), 0.001 | 896 | 896 | 5,600 | the pretrained arms, each at its best rate | water test frames: the same liquid, runs never seen |
| No predictor (the E11 encoder) at the same three rates | 8,960 without labels, then 896 | 896 | 22,400 + 5,600 | baseline at its best rate | same |
| Plain (the E04 encoder) at the same three rates | 8,960 without labels, then 896 | 896 | 22,400 + 5,600 | baseline at its best rate | same |
| Random start with the pretrained sizes, at 0.0003. Built from the no-predictor encoder: every weight matrix keeps its singular values (its overall size, and how unevenly it stretches its input) but gets random directions; every other tensor keeps its values in a random order | 896 | 896 | 5,600 | baseline at 0.0003, no predictor at 0.0003 | same |

What does each compared arm get that the other does not?
- Within one row, only the learning rate differs.
- Random start with the pretrained sizes against the baseline: only the sizes of the starting weights, layer by layer.
- Random start with the pretrained sizes against no predictor: the same sizes, without what pretraining learned.
- A pretrained arm against the baseline: 22,400 pretraining steps on 8,960 frames. This is the known difference whose effect is being explained; it is not a fair comparison at equal total steps.

## Organic arms

10% of the labels (8,400 frames), 100 labelled epochs (52,500 steps), 3 seeds each.

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|
| Baseline at 0.0001 and 0.0003 (exists) | 8,400 | 8,400 | 52,500 | plain at its best rate | 21 test molecules, never seen |
| Plain (the E01 encoder) at 0.0001 and 0.0003 (exists) | 84,000 without labels, then 8,400 | 8,400 | about 26,250 + 52,500 | baseline at its best rate | same |

What does each compared arm get that the other does not? Within a row, only the learning rate. Plain against the baseline: the pretraining, as in E01.

## Predictions (confidence, Claude's; Lauren's to add)

1. Water: with each arm at its best learning rate, no predictor still beats the baseline (seed rule): 70%.
2. Water: the baseline at its best learning rate is at least 10% better than at 0.0003: 45%. If so, part of the lead was the baseline's learning rate.
3. Water: the random start with the pretrained sizes does not beat the baseline at 0.0003 (seed rule): 80%.
4. Organic: at 0.0001, plain still does not beat the baseline at the baseline's best rate: 85%.

## Controls

- Must reproduce: one rerun of the water baseline at 0.0003, seed 0, lands inside the earlier range of seeds (0.0286 to 0.0313). It checks that nothing in the set-up has changed since 12 Nov.
- The random start with the pretrained sizes is what it claims to be, checked before its runs are read: every weight matrix has the pretrained matrix's singular values to six significant figures, every other tensor has the same values when sorted, and its features do not behave like the pretrained encoder's (which change about 2% over 16 fs and give a momentum read-out of 0.00). If its features change by less than 5% over 16 fs or its momentum read-out is below 0.5, the construction has kept what was learned and the arm is void.
- Range: if an arm's best learning rate is the lowest or the highest one tried, one more rate is added on that side before the arm is compared.

## Assumptions

1. Three learning rates a factor of about three apart are enough to find each arm's best.
2. Warm-up (1% of the steps) and the decay of the learning rate stay as they are.
3. The water runs need the 46 GB cards of node 15, which other people also use.
4. One pretrained encoder per variant.

**Measure and rule:** the best learning rate of an arm is the one with the lowest mean validation error over its 3 seeds. Arms are then compared on the test set at their best rates: one arm beats another when every seed of it beats every seed of the other.

## What we do with the result

| Result | Reading | Next |
|---|---|---|
| Water lead survives best-against-best, and the random start with the pretrained sizes gives no lead | The lead comes from what the encoder learned. "Faster early training" stands. | Nothing changes. |
| Water lead disappears best-against-best | The lead was the baseline's learning rate. Pretraining gives nothing at equal labelled steps either. | Correct the deep dive and the results report. |
| The random start with the pretrained sizes reproduces the lead | The lead comes from the sizes of the starting weights, not from learned features. | Correct the documents; pretrained features on water are not shown to help at all. |
| Organic: plain at the lower rate beats the baseline | The organic loss was the training recipe (4a). | Rerun E24's pretrained arms at that rate. |

**Runs and cost:** water: 6 baseline, 6 no predictor, 6 plain, 3 with the pretrained sizes, 1 rerun: 22 runs of about 1 hour on 46 GB cards. Organic: 6 runs of about 5.3 hours on 24 GB cards (32 GPU hours).

**Code needed after the card is agreed** (reviewed by an independent agent before anything runs): configs and run lists with the learning rate changed; a script that writes the random start with the pretrained sizes (from train/make_random_checkpoint.py) and prints the checks listed under Controls.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
