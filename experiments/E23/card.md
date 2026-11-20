# E23: does pretraining learn slow features, what are they, and is that why it does not help?

Status: proposed (20 Nov 2026). Card first; no code is written until it is agreed, and nothing runs until Lauren gives the go-ahead in writing.

**Why this exists:** with equal data and equal optimiser steps, pretraining gives no gain (E16, E15, E20). We already know two things about the pretrained encoders: their features change very little over one time step (0.4 to 2%, against 35 to 55% for an encoder trained with labels; E08), and a linear read-out cannot recover momentum from them (0.00, against about 0.8 for an untrained encoder; E18). A 2026 paper on the same kind of training on video (MotionJEPA, pointed out by Bruno) reports the same pattern and calls it a bias towards slow features. What we have not shown: (1) that pretraining removes the fast information, as opposed to never learning it; (2) what the slow features describe; (3) that the task needs the fast information, which would make slowness a reason for "no gain".

**Hypothesis:** the pretraining loss (features now and later should match, while every feature dimension stays varied and uncorrelated across atoms) selects the slowest-changing quantities available. In liquid water the cheapest slow quantity is the orientation of each molecule in the box, which barely changes in 16 fs and differs between atoms. Momentum, which for a hydrogen reverses within about 10 fs, is discarded. The one-step task needs momentum and other fast quantities, so labelled training has to rebuild them.

**Tested on:** liquid water, 16 fs. All measurements use the 28 test runs (the same liquid, simulation runs never seen in training or pretraining), read from the raw trajectories stored every 0.25 fs.

## Part 1: is the fast information removed during pretraining?

**Method:** rerun two existing pretrainings unchanged except that the encoder is saved at optimiser steps 0, 100, 300, 1,000, 3,000, 10,000 and 22,400 (the end). At each saved point measure (a) feature change over 16 fs divided by the spread across atoms (as E08), (b) the momentum read-out (as E18 and E19: size and vector), (c) the pretraining loss terms.

| Arm | Frames seen | Labels seen | Optimiser steps | What differs |
|---|---|---|---|---|
| Plain pretraining (as E04) | 8,960 | 0 | 22,400 | Reference |
| No-predictor pretraining (as E11) | 8,960 | 0 | 22,400 | No predictor |

**Prediction (confidence, Claude's):** the momentum read-out falls from about 0.8 at step 0 to below 0.1 by the end in both arms, while the total loss falls throughout: 85%. It falls earlier without a predictor than with one: 70% (E19 follow-up: gone by 2,240 steps without a predictor, still 1.00 with one).

**Controls:** step 0 must reproduce the untrained encoder's numbers (change 20.5%, momentum size 0.82). The final step must reproduce the E08 and E18 numbers of the original pretrainings to within 0.02 (same seed; small differences from hardware are possible).

## Part 2: what are the slow features?

**Method, for each encoder in the table below:**
1. Timescales. Encode frames at time lags of 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 and 1,024 fs from many start times in each test run. Split the features into independent directions ordered from slowest to fastest (the directions that change least over 16 fs relative to their spread, found by a standard eigenvalue calculation on the features; done per element). For each direction, the timescale is the lag at which its correlation with itself falls below 1/e. Directions still above 1/e at 1,024 fs are counted as "longer than 1 ps", because the runs are only 2 ps long.
2. Candidates. Read-outs (as E19, on rotated and unrotated frames) of quantities with known timescales: momentum (fast); O–H bond length (fast, about 10 fs vibration); the direction of each molecule's axis and of its H–H line in the box (slow, picoseconds); number of oxygens within 3.5 Å (slow).
3. For reference, the timescales of the candidate quantities themselves, measured the same way from the trajectories.

| Encoder | Frames seen | Labels seen | Optimiser steps | Source |
|---|---|---|---|---|
| Untrained | 0 | 0 | 0 | random weights, seed 0 |
| Trained with labels | 8,960 | 8,960 | 168,000 | E00 |
| Plain | 8,960 | 0 | 22,400 | E04 |
| No predictor | 8,960 | 0 | 22,400 | E11 |
| Relative loss | 8,960 | 0 | 22,400 | E04 |
| Regulariser only | 8,960 | 0 | 22,400 | E06 |
| **New: plain, separate rotation for each frame of a pair** | 8,960 | 0 | 22,400 | new pretraining |

The new arm differs from plain in one thing: the two frames of a pair get independent random rotations (an existing switch in the pretraining script), which makes orientation in the box useless for matching features. It is a control to identify what the features hold, not a proposed method (decision of 9 Nov stands: independent rotations push the encoder to ignore direction altogether).

**Predictions (confidence, Claude's):**
- At least 90% of the plain and no-predictor directions have timescales above 256 fs, and fewer than half of the label-trained encoder's directions do: 70%.
- Molecular axis direction reads out above 0.6 from plain and no predictor (rotated frames), against below 0.4 for untrained: 65%.
- In the separate-rotation arm the change under rotation falls below 0.5 (from 1.97) and the axis-direction read-out falls below 0.3: 75%. If the features stay orientation-dependent, the orientation explanation is wrong.
- Relative loss, the one variant with fast-changing hydrogen features, shows a clear group of directions with timescales under 32 fs: 60%.

**Controls:**
- Works: the timescale method applied to the raw momentum of hydrogens gives under 16 fs, and applied to the molecular axis gives over 256 fs.
- Works: a random direction per atom, constant in time, gives "longer than 1 ps"; white noise gives under 1 fs. So the method can report both ends.
- Noise: timescales are computed separately on two halves of the test runs; a direction's two estimates must agree within one lag step for 90% of directions, otherwise the histogram is not read.

## Part 3: does the task need the fast part?

**Method:** take the model trained with labels (E00, test error 0.0038), unchanged. Order its feature directions from slowest to fastest as in part 2. Replace the chosen directions by their average value over the test frames (so the output layers see typical values, not zeros), pass the altered features through the model's own output layers, and measure the one-step test error. No retraining.

| Arm | Directions replaced by their average | What differs |
|---|---|---|
| None | 0 | Must reproduce 0.0038 |
| Fastest first | fastest 1, 2, 5, 10, 25, 50% | Which directions |
| Slowest first | slowest 1, 2, 5, 10, 25, 50% | Which directions |
| Random | random 1, 2, 5, 10, 25, 50%, 5 draws | Control |

Model, frames and output layers are identical in every arm; only the set of replaced directions differs.

**Prediction (confidence, Claude's):** replacing the fastest 10% raises the momentum error at least 5 times more than replacing the slowest 10%, and more than every random draw: 65%. If slow and fast directions matter equally, slowness is not what makes pretrained features unhelpful.

**Controls:** the "none" arm must give 0.0038 to four decimals. Replacing all directions must give about the error of always predicting the average (close to the straight-line guess, 0.204).

**Limit of this part:** it shows what the label-trained model uses, with its own output layers. It does not by itself show that a pretrained encoder could not be made useful.

## Assumptions
1. Feature directions are found with a linear method. Information stored in a non-linear form is not seen (as for the probes).
2. One encoder per row; no spread across pretraining seeds.
3. The 2 ps runs limit timescales to about 1 ps.
4. Parts 1 and 2 describe water only. Organic molecules have 50-frame runs of 8 fs steps (392 fs), enough for a shorter version later.

## What we do with the result
- All three parts as predicted: we have a shown mechanism for "no gain": the loss keeps slow quantities (orientation), discards fast ones, and the task uses the fast ones. The next design must keep fast information (for example an added term that predicts the change in features, as MotionJEPA does, or a read-out of momentum during pretraining).
- Part 3 fails (slow directions matter as much): look to the other hypotheses in notes/decisions.md (20 Nov).
- Part 2 rotation control fails: the features are orientation-dependent for another reason; orientation is not the slow quantity.

## Known tension
On organic molecules, pretrained models were ahead of the baseline after the first labelled epoch (E01). If pretraining only erased useful information they should start behind. Optional addition to part 1: track the same read-outs during the first labelled epochs from a pretrained and a random start.

**Cost:** 3 pretrainings of about 3.5 hours each on a 24 GB card (two reruns with saved encoders, one separate-rotation arm). Everything else is inference: under 2 GPU hours.

**Code needed after the card is agreed** (reviewed by an independent agent before anything runs): a switch to save the encoder at given steps; a timescale script (part 2); a replace-and-evaluate script (part 3). Probes reuse eval/probe_direction.py.

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
