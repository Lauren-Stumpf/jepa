# E25: three checks of what the explanations assume, on models we already have (no training)

Status: proposed (20 Nov 2026). Card first; no code is written until it is agreed, and nothing runs until Lauren gives the go-ahead in writing.

**Why this exists:** the deep dive of 20 Nov (https://claude.ai/artifact/9SqGJY3qrzteSPSwm5ySJ5) rests on three statements that were inferred, not measured directly.
- A. Pretrained encoders have dropped momentum. So far this is shown only by linear read-outs, and a read-out that finds nothing does not show the information is absent.
- B. On organic molecules, pretrained features follow what is constant along a simulation run (explanation 3b). This was inferred from the "moment" share in E08, which cannot tell "constant within a run" from "changing slowly within a run".
- C. The one-step error is dominated by fast atoms, where slow features cannot help (explanation 4c). Not measured for pretrained models.

Each can be checked on existing models. Nothing is trained.

**Questions:** A: does a pretrained encoder's output change at all when the momenta are changed? B: are pretrained organic features constant along a run, and more so on molecules seen in pretraining? C: which atoms carry the error, and on which atoms is the pretraining lead?

## Part A: does the encoder respond to momentum at all?

**Tested on:** water: the 28 test runs (the same liquid as training, runs never seen). Organic: the 21 validation molecules (never seen by any encoder).

**Method:** an intervention on the input, with no read-out. Each frame goes through the encoder twice: once as stored, and once with every atom's momentum replaced by the same atom's momentum from another moment of the same run (water: 200 fs or more later, from the raw runs; organic: the next stored start time, 100 fs later). Positions stay as they are. Measure: the mean squared change of an atom's features, divided by the spread of features across atoms of the same element in one frame. This is the scale of E08, so the number can be read beside "change over one time step". A second version reverses the sign of every momentum.

**Encoders:** water: untrained, trained with labels (E00), plain and relative loss (E04), no predictor (E11), regulariser only (E06), plain after 20 epochs on 896 frames (E13). Organic: untrained, trained with labels (the 10% baseline), plain and relative loss (E01).

**Predictions (confidence, Claude's; Lauren's to add):**
- Plain and no predictor on water, and plain on organic: response below 0.2% of the spread, and at least ten times smaller than the same encoder's change over one time step (0.5 to 1.6%): 60%.
- Relative loss on water: response above 1%: 50%. Its hydrogen features change by 89% per step, and it is not known whether that comes from positions or from momenta.
- Plain after 20 epochs on water: response above 5%: 55%. A note of 17 Nov says momentum could still be read out of it; this checks that note.

**Controls:**
- Must work: the untrained encoder responds by more than 2% (momentum is read out of it at 0.82 to 1.00), and the encoder trained with labels by more than 10%.
- Must be zero: the same inputs twice, and momenta replaced by themselves, give exactly 0.
- Size of the intervention: the mean squared change of the momenta themselves, divided by their spread, is reported (about 2 when the two momenta are unrelated). A small response then cannot come from a small intervention.

**Rule:** momentum is "absent" if the response is below 0.2% of the spread and below a tenth of that encoder's change over one step. It is "present" if the response is above 2%. In between it is reported as partial.

**What we do with it:** absent: the statement "pretraining removes momentum" stands without the caveat about linear read-outs. Present in an encoder whose linear read-out is 0.00: the information is there in a form the read-out misses, the documents are corrected, and a small non-linear read-out is trained on it.

## Part B: do pretrained organic features follow what is constant along a run?

**Facts about the data:** each organic molecule has 10 constant-energy runs at different temperatures. A run holds 25 start times, 100 fs apart (2.4 ps).

**Tested on:** the 21 validation molecules (never seen by any encoder), and 21 training molecules chosen at random with a fixed seed (seen in pretraining).

**Encoders:** untrained, trained with labels (the 10% baseline), plain and relative loss pretrained on all 84,000 frames (E01), plain pretrained for 20 and for 200 epochs on the 8,400 labelled frames (E15).

**B0, from the pretraining logs only.** The prediction term of the pretraining loss at the end of pretraining, on training frames and on validation molecules, for the E01 and E15 pretrainings. A term much lower on training frames means the features were fitted to the particular runs.

**B1, the spread of the features with a "run" level.** E08 split the feature spread into element, atom and moment. The moment part is split again, exactly, into: run (an atom's average within one run, minus its average over all its runs) and time within the run. Reference: the same split with the run labels shuffled within each molecule, which gives what chance produces with 25 frames per run.

**B2, how long features stay the same within a run.** The correlation between an atom's features at one start time and the same atom's features 100, 200, 400, 800, 1,600 and 2,400 fs later in the same run, per element. It is read against the correlation between two different runs of the same molecule, which is the level left when everything about the run is forgotten.

**B3, read-outs of quantities that are constant along a run.** Targets: the run's mean kinetic temperature (from data/organic8/frames.csv), and the size of the molecule's angular momentum (computed from positions and momenta; used only if the data conserve it, meaning its spread within runs is below a tenth of its spread between runs). Read-out: ridge regression from the frame's mean feature vector, fitted on 16 validation molecules and scored on the other 5, for five different splits. References: the same read-out from the frame's own kinetic temperature alone (one number), from the untrained encoder, and from the encoder trained with labels.

**Predictions (confidence, Claude's; Lauren's to add):**
- B0: the ratio of validation to training prediction term is at least twice as large after 200 epochs on 8,400 frames as after 20 epochs on 84,000 frames: 55%.
- B1: for plain (E01) on validation molecules, the run part, after subtracting its shuffled reference, is at least half of the moment share: 50%.
- B1: for plain after 200 epochs on the labelled frames (E15), the run part is at least 10 points larger on training molecules than on validation molecules: 40%.
- B3: plain reads out the run's mean temperature at least 0.2 better (share of variation explained) than both the untrained encoder and the frame's own temperature: 35%. Part A may show that momentum is gone; temperature could then only be read from how distorted the molecule is.

**Controls:**
- Must work: the frame's own kinetic temperature is read out of the untrained encoder at above 0.5 (momentum is one of its inputs).
- Must fail: a random number drawn once per run scores 0.00 for every encoder.
- Sums: the shares in B1 add to 1.000 for every encoder, and reproduce the E08 element, atom and moment shares on the validation molecules.

**What we do with it:**
- Run part large and constant along the run (B1, B2), and larger on molecules seen in pretraining: the premise of explanation 3b holds. E24's second test set then shows whether it matters for the error.
- Run part small: explanation 3b is wrong in this form. The extra loss on new molecules then points to the general penalty for starting from trained weights (3c).

## Part C: which atoms carry the one-step error, and where is the pretraining lead?

**Tested on:** the test sets (water: the same liquid, runs never seen; organic: 21 molecules never seen).

**Method:** the test evaluation already writes the position and momentum error by element for every run (results/eval/test/<run>.log). The numbers are collected. No inference is needed.

**Runs:** water: baseline after 100, 500, 1,000 and 1,400 epochs; plain, no predictor and relative loss after 100 epochs; plain after 1,000 epochs. Organic: baseline after 100 and 500 epochs; plain and relative loss.

**Reported:** each element's share of the total squared error; and for each pretrained run its percentage difference from the baseline per element, at equal labelled steps and at equal total steps.

**Predictions (confidence, Claude's; Lauren's to add):**
- On water, hydrogens carry at least 70% of the squared momentum error of the 100-epoch baseline: 60%.
- On water, the lead of no predictor over the baseline at equal labelled steps is at least 5 points larger for oxygen than for hydrogen: 45%.

**What we do with it:** lead concentrated on oxygen: consistent with slow features helping the slow atoms (4c). Lead equal across elements: 4c is not supported.

## Assumptions
1. One encoder per variant (one pretraining seed).
2. In part A the replacement momentum comes from the same atom in the same run, so its size is typical for that atom, but it no longer fits the positions. The reversed-sign version checks that the result does not depend on this choice.
3. Part B describes organic molecules only. The water runs are one liquid at one temperature, so little differs between runs.
4. A run is 2.4 ps long. "Constant along a run" means constant over 2.4 ps.

**Cost:** inference on about 17 encoders and ridge fits: under 5 GPU hours on one 24 GB card. Parts B0 and C: none.

**Code needed after the card is agreed** (reviewed by an independent agent before anything runs): a new script eval/momentum_intervention.py (part A); a run level and the correlation over time added to eval/feature_decomposition.py (B1, B2); run-level targets added to eval/probe_features.py (B3); a small script that collects the errors by element from the test logs (C) and the loss terms from the pretraining logs (B0).

## Result
(to be filled in)

## Assumptions check
(to be filled in)

## Next
(to be filled in)
