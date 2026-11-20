# E24: does pretraining help when the unlabelled frames hold more molecules than the labels? (organic, labels on 17 molecules)

Status: proposed (20 Nov 2026). Card first; no code is written until it is agreed, and nothing runs until Lauren gives the go-ahead in writing.

**Why this exists:** in every comparison so far the unlabelled frames held nothing the labelled frames lacked. The labelled 10% is every tenth frame, so it covers all 168 training molecules, and the label is the later frame, which pretraining also uses. The deep dive of 20 Nov (https://claude.ai/artifact/9SqGJY3qrzteSPSwm5ySJ5) gives this as the most likely main reason for "no gain" (its group 1). The rival reason is that the pretraining objective learns nothing that carries over to other molecules (its group 2). The two reasons only predict different results when the unlabelled frames do hold something more. Here they hold ten times more molecules than the labels.

**Question:** with labels on 17 molecules and pretraining on all 168, does the pretrained model predict new molecules better than a baseline given the same total number of optimiser steps?

**Hypotheses (rivals):**
- Room: the objective can pass on what it learns from molecules it saw without labels. Then P168 beats the baseline on new molecules, and beats P17.
- Objective: what the objective learns does not carry over to other molecules. Then P168 is no better than the baseline on new molecules, although the control U168 is.
- Tied to seen molecules (explanation 3b of the deep dive): P168 is better only on the 151 molecules it saw in pretraining.

**Tested on:** organic molecules, 8 fs. Three sets of frames, all reported for every arm:
1. The 21 test molecules, never seen in any stage (primary).
2. Frames of the 151 training molecules that carry no labels in this experiment. The baselines and P17 never see them. P168 sees them without labels. U168's encoder saw them with labels.
3. Frames of the 17 labelled molecules (how well each arm fits what it trained on).

## Arms

Main subset A (17 of the 168 training molecules, chosen at random with a fixed seed; every frame of those molecules is labelled: 8,500 frames, about the same number as the usual 10% subset). Three seeds per arm. L is the number of labelled epochs: 100 unless the pilot below says 50. One labelled epoch is about 532 optimiser steps (batch 16).

| Arm | Frames seen | Labels seen | Total optimiser steps (L = 100) | Compared with | Tested on |
|---|---|---|---|---|---|
| B: baseline, L epochs | 8,500 (17 molecules) | 8,500 | about 53,200 | P168, P17, U168 at equal labelled steps | sets 1, 2, 3 |
| B+: baseline, L + 49 epochs | 8,500 (17 molecules) | 8,500 | about 79,300 | P168 and P17 at equal total steps (primary) | sets 1, 2, 3 |
| P168: plain pretraining on all 168 molecules (the existing E01 encoder, 20 epochs), then L epochs | 84,000 (168 molecules) | 8,500 | about 26,250 + 53,200 = 79,450 | B+ (primary), B, P17 | sets 1, 2, 3 |
| P17: plain pretraining on the 17 labelled molecules only (197 epochs, the same number of steps), then L epochs | 8,500 (17 molecules) | 8,500 | about 26,200 + 53,200 = 79,400 | P168, B+ | sets 1, 2, 3 |
| U168, a control that must work: the encoder of the baseline trained with labels on all 168 molecules (the E14 route), then L epochs | 84,000, with labels | 84,000 for the encoder, 8,500 in this training | not matched | B+ | sets 1, 2, 3 |

What does each compared arm get that the other does not?
- P17 against B+: its first 26,200 or so steps are pretraining steps on the same frames, in place of about 26,100 more labelled steps. One thing differs: the kind of step.
- P168 against P17: the same number and kind of pretraining steps, on frames of 168 molecules in place of 17. One thing differs: the molecules pretraining saw.
- P168 against B+: both of the above together. This is the headline comparison. The two comparisons before it say which part a difference comes from.
- U168 against B+: an encoder trained with labels on all 168 molecules. It is a control, not a fair arm.

Second subset B (another 17 molecules, none shared with A), three seeds: B+ and P168 only. It checks that the headline comparison does not depend on which molecules were labelled.

Optional arm, subset A only (3 runs, no new pretraining): P168 with the relative-loss encoder of E01, the one variant whose features change quickly. It asks whether the answer depends on the variant.

## Pilot (one run, before anything else)

B+ on subset A, seed 0. It is reused as one seed of the main experiment. Two things are read from it:
1. Is there room? Its test error on new molecules is compared with 0.0226, the baseline trained on the same number of frames spread over all 168 molecules (E01). If it is less than 15% worse, 17 molecules are enough for the baseline, this design cannot separate the hypotheses, and the card is redone with fewer molecules.
2. Has the baseline finished? If its best validation epoch is 40 or earlier, L = 50 (and B+ is 99 epochs). Otherwise L = 100. If the best epoch is the last one, B+ is lengthened and the card says so.

## Predictions (confidence, Claude's; Lauren's to add)

1. The pilot's test error is at least 40% above 0.0226 (0.0316 or more): 75%.
2. U168 beats B+ by at least 20% on new molecules, on every seed: 85%.
3. P168 does not beat B+ on new molecules (seed rule), in both subsets: 65%.
4. P168 is not better than P17 on new molecules: 55%.
5. P168's difference from B+ is at least 2 points better on set 2 (molecules it saw without labels) than on set 1 (new molecules): 50%.

## Controls

- Must work: U168 beats B+ on new molecules. If it does not, this design cannot show the benefit of a good encoder and nothing is concluded.
- Must not win: P17 gets no extra molecules, and long pretraining on the labelled frames made the model 8 to 9% worse in E15. If P17 beats B+ here, a gain of P168 cannot be put down to the extra molecules, and P168 is read against P17 only.
- Noise first: the three seeds of B give the seed spread for training on 17 molecules before any difference is read.
- Bookkeeping: the script that writes the subsets saves the molecule ids, and the evaluation checks that no frame of the 17 labelled molecules is in set 2.

## Assumptions

1. One subset of 17 molecules may be unusual. The second subset checks the headline comparison only.
2. One pretraining seed per encoder, as in every experiment so far.
3. At equal pretraining steps, P17 passes over each of its frames 197 times and P168 over each of its frames 20 times. Equal steps makes this unavoidable. If the comparison of P168 with P17 decides the reading, a P17 pretrained for 20 epochs is added.
4. A pretraining step on organic molecules processes 64 pairs (128 frames through the encoder) and a labelled step 16 frames. Equal optimiser steps is therefore generous to pretraining in compute. If P168 beats B+, a baseline matched on compute is added before the result is claimed.
5. Labelled training uses the baseline's learning rate (0.0003) for every arm. E26 tests whether that matters. If it does, the pretrained arms are rerun at the better rate.
6. The saved model is the epoch with the best error on the 21 validation molecules (never seen in training or pretraining), the same for every arm.

**Measure and rule:** test momentum error on the 21 test molecules, 3 seeds per arm. One arm beats another when every seed of it beats every seed of the other. "P168 beats B+" counts only if that holds in both subsets.

## What we do with the result

| Result | Reading | Next |
|---|---|---|
| P168 beats B+ on new molecules, and beats P17 | The objective passes on information from unlabelled molecules. Earlier experiments had no room to show it. | One-step pretraining is worth pursuing where unlabelled molecules outnumber labelled ones. Add the compute-matched baseline. |
| P168 is no better than B+ and P17, while U168 is better | The objective does not carry over to new molecules even when there is room. | Change the objective so that it keeps fast information, or close one-step pretraining. |
| P168 is better than B+ on set 2 only | The features are tied to the molecules seen in pretraining (explanation 3b). | As the row above, with 3b supported. |
| U168 is no better than B+ | The design cannot detect a benefit. | No conclusion. Redo with a gentler labelled training. |

**Runs and cost:** 1 pretraining (P17, about 9 hours, as the 200-epoch pretraining of E15). With L = 100: 21 training runs (15 on subset A, 6 on subset B) of about 5.3 hours, and about 8 hours for each B+ run. About 135 GPU hours on 24 GB cards in all, about a day on six cards. With L = 50, about 70 GPU hours. The optional arm adds 16 GPU hours. Test and frame-set evaluation use the existing evaluation scripts.

**Code needed after the card is agreed** (reviewed by an independent agent before anything runs): a script that writes the labelled subsets by molecule (A and B, fixed seed, molecule ids saved); configs and run lists (as scripts/make_e13_e15.py); the three evaluation sets added to the evaluation queue.

**Not included, and why:** a water version with far fewer labelled frames. It asks the same question for a liquid and needs the 46 GB cards. It is worth designing only if this experiment shows a gain.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
