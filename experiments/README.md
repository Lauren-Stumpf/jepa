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

## Index

| Id | Experiment | Status |
|---|---|---|
| E00 | FlashMD baselines on all the labels (organic 8 fs; water 1, 4 and 16 fs) | done (card written afterwards) |
| E01 | Organic, 10% of the labels: five pretraining variants against the baseline | done (card written afterwards) |
| E02 | Organic, all the labels: normalised-target pretraining against the baseline | done (card written afterwards) |
| E03 | Organic, 10% of the labels: freezing the encoder at the start of fine-tuning | done |
| E04 | Water 16 fs, 10% of the labels: five pretraining variants against the baseline | running (card written afterwards) |
| E05 | Hypothesis A: does the water gain last with ten times more training? | agreed |
| E06 | Hypothesis B: does the water gain come from the regulariser rather than prediction? | agreed |
