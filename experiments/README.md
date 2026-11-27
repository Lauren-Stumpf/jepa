# How experiments are run here

One folder per experiment, `experiments/E<nn>/`, holding:

- `card.md`: written and agreed **before** anything runs. The result and assumption check are filled in afterwards.
- `runs.txt`: the runs, read by `scripts/launch_runs.sh`.

Agreeing a card is the go-ahead to run its runs on the cluster.

## Principles

1. **Three kinds of setting.** Each setting is either the one thing **varied** between arms, something **matched** so every arm gets a fair chance (training steps, compute, tuning), or **fixed**.
2. **Predict before running.** Write the result expected if the hypothesis is true, the result that would show it is false, and how confident we are.
3. **Rival hypotheses, one deciding test.** Prefer the experiment whose outcome separates the explanations over one that fits all of them.
4. **Controls both ways.** One run that must work and one that must fail. If either misbehaves, the set-up is broken, not the hypothesis.
5. **Know the noise first.** The spread between seeds is measured before a difference is read.
6. **Decision rule in advance.** The primary measure and the difference that counts are written down before the result is seen.
7. **Look at the raw thing.** Individual molecules, atoms and curves, not only averages.
8. **Try to break our own result.** For each finding, run the cheapest check that could overturn it.
9. **Most information per hour first.** If no result would change what we do next, the run is not worth making.

## Where results are

In the Weights and Biases project **jepa-flashmd-experiments**:

- **Group** = experiment (E00, E01, ...). **Job type** = arm (baseline, plain, ...).
- Each experiment has one run with job type **summary**. Its page shows the card, the bar charts by arm (mean, seeds, spread, percentage against the reference arm), the learning curves, and a table.
- Only valid runs are kept. A run that fails, or is restarted from scratch, is deleted automatically.
- The old project, jepa-flashmd, is left as it was.

## Card template

```
# E<nn>: <short title>

Status: proposed | agreed | running | done
Question: <one sentence>
Hypothesis: <what we believe and why>
Prediction (confidence): <number expected if true>; <what we would see if false>; <how sure, in %>
Varied: <the one setting that differs between arms>
Matched: <what is made fair for each arm>
Fixed: <everything else>
Controls: works: <run that must succeed>; fails: <run that must fail>
Assumptions: <each, with how we would notice it is wrong>
Measure and rule: <primary measure, seeds, difference that counts>
Runs: <list, or see runs.txt>; cost: <GPU hours>

## Result
<figure in the summary run; numbers; supported or not>

## Assumptions check
<which held, which did not>

## Next
<what this result makes us do>
```

## Rule added 17 Nov 2026: say what it was tested on

Every card and every results table states which frames a number was measured on and whether the model had seen that system: for water, validation and test frames are the same liquid as training (other runs); for organic, they are molecules never seen in training. A comparison between water and organic runs organic both ways (seen molecules and unseen molecules). A probe that finds nothing shows only that a linear read-out failed, not that the information is absent.

## Rule added 18 Nov 2026: what each arm gets

Every card that compares arms has this table before anything is launched:

| Arm | Frames seen | Labels seen | Total optimiser steps | Compared with | Tested on |
|---|---|---|---|---|---|

Under it, one line: what does each compared arm get that the other does not? If the answer is more than the one varied thing, the comparison is not launched. Pretraining brought two hidden extras with it in E01 to E17 (more frames, then more steps); this table would have shown both.

Also: before comparing against a baseline, check that more training no longer improves the baseline. The 100-epoch water baseline was far from finished (E16).

## Index

| Id | Experiment | Status |
|---|---|---|
| E00 | FlashMD baselines on all the labels (organic 8 fs; water 1, 4 and 16 fs) | done (card written afterwards) |
| E01 | Organic, 10% of the labels: five pretraining variants against the baseline | done (card written afterwards) |
| E02 | Organic, all the labels: normalised-target pretraining against the baseline | done (card written afterwards) |
| E03 | Organic, 10% of the labels: freezing the encoder at the start of fine-tuning | done |
| E04 | Water 16 fs, 10% of the labels: five pretraining variants against the baseline | done (card written afterwards) |
| E05 | Hypothesis A: does the water gain last with ten times more training? | done |
| E06 | Hypothesis B: does the water gain come from the regulariser rather than prediction? | done |
| E07 | Organic models on unused frames of the training molecules (seen against new molecules) | done |
| E08 | Where encoder features get their spread, and how much they change over the gap | done |
| E09 | Organic errors split by kind of frame | done |
| E10 | Are the organic runs deterministic and energy-conserving? | done |
| E11 | Water pretraining with no predictor | done |
| E12 | Organic pretraining with pairs 100 fs apart | done |
| E13 | Water pretraining on only the frames the baseline sees | done |
| E14 | Control that must work: full-data organic encoder through the checkpoint route | done |
| E15 | Organic pretraining on only the frames the baseline sees (plain and no predictor) | done |
| E16 | Water: baseline with as many training steps as the pretrained arms | done |
| E17 | Water, all the labels: baseline against pretraining | done |
| E18 | Linear probes: what the encoders hold, organic and water | done |
| E19 | Direction in the features: rotation sensitivity and direction probes | done |
| E20 | Organic, 10% of the labels: is the baseline still improving? (500 epochs) | done (0.0186 against 0.0226: yes, 18% better) |
| E21 | Water, all the labels: baseline with the same total steps as the E17 pretrained arms | done (level: 0.0060 against 0.0061) |
| E22 | Rolling out the water baseline: error growth, error on its own outputs, drift, failure, feature drift | done (cools by 164 K in 100 ps; no worse on its own outputs) |
| E23 | Does pretraining learn slow features, what are they, and does the task need the fast part? | proposed |
| E24 | Organic, labels on 17 molecules, pretraining on all 168: does pretraining help when the unlabelled frames hold more molecules than the labels? | proposed |
| E25 | Three checks on existing models, no training: momentum by intervention, what is constant along an organic run, error by element | proposed |
| E26 | Learning rate of both arms, and a random start with the pretrained weights' sizes: where do the water lead and the organic loss come from? | organic part done (at 0.001 the arms overlap: 0.0207 against 0.0212); rest not run |
| E27 | Water, 10% of the labels, long runs at equal total steps: baseline 2,400 epochs against plain pretrained + 2,000 epochs | done: pretrained 0.0046 (all frames) and 0.0047 (labelled frames only) against 0.0065; control with two learning-rate cycles next |
| E28 | Rolling out the water baseline the way the FlashMD paper does: energy correction and thermostat | done (303 K held for 100 ps with the energy correction alone) |
| E29 | Small molecules with fixed neighbours: does stepping in feature space accumulate less error than stepping in real space? | proposed |

Note (16 Nov 2026): in E01, E04, E05, E06, E11 and E12 the pretrained arm was pretrained on all training frames and the baseline saw 10% of them, so those comparisons are confounded with extra data. See notes/decisions.md and experiments/INSTRUCTIONS_E13_E15.md.
