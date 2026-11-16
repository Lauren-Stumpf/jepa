# Instructions for implementing E13, E14 and E15

Written 16 Nov 2026 for an agent that has not seen the earlier conversation. Read this file, then the three cards (`experiments/E13/card.md`, `E14/card.md`, `E15/card.md`), then `experiments/README.md` and the last two sections of `notes/decisions.md`.

## What the project is

Lauren is testing whether pretraining the encoder of FlashMD (a model that predicts where atoms will be a short time later) lowers its error when few labelled frames are available. Pretraining compares each atom's features now with its features after the time jump, plus a regulariser that keeps features different between atoms. Two datasets: organic molecules (8 fs jump) and liquid water (16 fs jump).

## Why these three experiments

Every earlier comparison at "10% of labels" pretrained on ALL training frames and then trained with labels on 10% of them, while the baseline saw only the 10%. In this task the label is the later frame, and pretraining uses the later frame too, so the pretrained arm effectively saw 10 times more data. The water gain (12 to 28%) may be extra data, not the objective.

| Pretraining against baseline | Organic | Water |
|---|---|---|
| Pretraining saw 10 times more frames | worse (E01) | better (E04, E11) |
| Same frames for both | weak evidence, worse (E02, 1 seed) | unknown |

- **E13** fills the unknown cell: water pretraining on the 896 labelled frames only.
- **E14** is a control that must work: an encoder trained with all organic labels must help when passed through our checkpoint route.
- **E15** fills the organic same-frames cell with 3 seeds, and tries the no-predictor objective on organic.

## Rules (Lauren's, not optional)

1. Ask Lauren's permission in writing before running anything on the cluster that starts, stops, moves or edits something. Read-only checks (status, logs, GPU use) are allowed.
2. Every piece of new code is reviewed by an independent agent before it runs. If the review finds problems, fix them and get a second review.
3. Do not delete anything unless she asks. Never change the old Weights and Biases project `jepa-flashmd`. Never commit the Weights and Biases key (it is in `~/.netrc` on the cluster).
4. Open few ssh connections and combine commands: the university gateway blocked us on 13 Nov after many logins in a row.
5. Do not `git pull` on the cluster while a bash script from the repository is running there (shared disk; bash reads scripts as it runs them).
6. Write for Lauren in plain words, British spelling, no long dashes, no acronyms in explanations, graphs before text, short updates tied to the question in the table above.
7. Fill in each card's Result, Assumptions check and Next sections when its runs finish, and add an entry to `notes/decisions.md`. Report results as they are, including ones that go against the prediction.

## Where things are

- Cluster: from Lauren's Mac, `cd ~/code/claude-cluster && ssh -F config torrnode7` (also `torrnode12`, `torrnode15`). Repository on the cluster: `/storage3/lauren/code/jepa`, branch `frozen-water-experiments`. Training environment: `source env_train.sh`.
- torrnode7: 24 GB cards (GPUs 0 and 1 belong to another user). torrnode12 and torrnode15: 46 GB cards, shared with others.
- Water training runs with labels need about 28 GB, so only 46 GB cards (`40000` in the runs file). Water pretraining at batch 8 and all organic runs fit a 24 GB card (`20000`).
- Launcher: `scripts/launch_runs.sh experiments/<E>/runs.txt` starts every run in the file as cards allow, waits for a pretraining before starting the runs that read it, and is safe to start on several nodes at once. Runs-file format is described at the top of that script; `experiments/E11/runs.txt` and `experiments/E12/runs.txt` are working examples.
- Config generator to copy: `scripts/make_e11_e12.py` (uses `copy_config` from `scripts/make_e05_e06.py`, which asserts that each replacement matched exactly once).
- Pretraining script: `train/pretrain_jepa.py`. Relevant flags: `--predictor plain|none`, `--epochs N`, `--batch_size N`, `--wandb jepa-flashmd-experiments --wandb_group <E> --wandb_job_type <arm>`. The supervisor `scripts/pretrain_supervisor.sh` takes `CONFIG=`, `TRAIN=`, `VAL=` from the environment (see the runs files above).
- Weights and Biases: project `jepa-flashmd-experiments`; group = experiment, job type = arm. Register new run names in `classify()` in `scripts/wandb_common.py`, add the experiment descriptions to `EXPERIMENTS`, and add reference arms to `REFERENCES` in `scripts/wandb_summary.py`. After runs finish: `python scripts/wandb_import.py --replace_failed_live` only if a finished run shows as failed, then `python scripts/wandb_summary.py <E>`.

## What to build

### Step 0: finish E11 and E12 (already run, results only on validation)

1. Water test evaluation: `scripts/eval_test_water16_queue.sh` has a fixed list of run names. Change it so the list also covers `e11_finetune_nopred_water16_10pct_s{0,1,2}` and the E13 names below (or takes names from a file). Organic test evaluation for `e12_finetune_plain_gap100_organic8_10pct_s{0,1,2}` and the E14 and E15 runs uses the command in `experiments/E07/card.md` with `data/organic8/test.xyz --every 5 --offset 0 --n 2100`; write a small queue script like `scripts/eval_seen_organic.sh` for it.
2. Write the E11 and E12 results into their cards (validation numbers, also in the cards and `notes/decisions.md`: E11 no predictor 0.0220 against baseline 0.0303 and plain 0.0268; E12 100 fs gap 0.0244 against 8 fs gap 0.0245 and baseline 0.0238), then the test numbers.
3. Thin the per-epoch checkpoints of finished runs with `bash scripts/thin_checkpoints.sh` (preview), then `--delete` only with Lauren's go-ahead.

### Step 1: E14 first (it is the control for everything else on organic)

1. New script `train/encoder_from_baseline.py` (arguments: the config, `--source <model.ckpt>`, `--out <folder>`, as used in the commands at the end): build the target-free model as `train/make_random_checkpoint.py` does, copy into it the encoder weights of `results/runs/flashmd_organic8_scratch/model.ckpt` (use `best_model_state_dict`; the copying logic already exists in `load_encoder(kind="baseline")` in `eval/feature_change_vs_jump.py`), and save `results/runs/supervised_encoder_organic8/pretrained.ckpt` plus `manifest.json` in the same format. Include a check that the saved encoder gives the same features as the source model on a few frames.
2. Three configs `configs/e14_finetune_supervised_organic8_10pct_s{0,1,2}.yaml`: copies of `configs/jepa_finetune_plain_organic8_10pct_s{s}.yaml` with `read_from` pointing at that checkpoint and the Weights and Biases block set to group `E14`, job type `supervised-encoder`.
3. `experiments/E14/runs.txt`: three `train|20000|1` lines.

### Step 2: E13 (water, same frames)

Four pretrainings, all with `CONFIG=configs/flashmd_water16_scratch.yaml TRAIN=data/water/pairs/16fs/train_10pct.xyz VAL=data/water/pairs/16fs/val.xyz` and `--batch_size 8`:

| Run name | Extra flags | Job type |
|---|---|---|
| `e13_pretrain_plain_same10_ep20_water16` | `--predictor plain --epochs 20` | `pretrain-plain-ep20` |
| `e13_pretrain_plain_same10_ep200_water16` | `--predictor plain --epochs 200` | `pretrain-plain-ep200` |
| `e13_pretrain_nopred_same10_ep20_water16` | `--predictor none --epochs 20` | `pretrain-nopred-ep20` |
| `e13_pretrain_nopred_same10_ep200_water16` | `--predictor none --epochs 200` | `pretrain-nopred-ep200` |

For each, three training configs `configs/e13_finetune_<plain|nopred>_same10_ep<20|200>_water16_10pct_s{0,1,2}.yaml`: copies of `configs/jepa_finetune_plain_water16_10pct_s{s}.yaml` with `read_from` changed; group `E13`, job types `plain-ep20`, `plain-ep200`, `nopred-ep20`, `nopred-ep200`. Runs file: pretrainings `pretrain|20000|1`, training runs `train|40000|1`.

Note: run names containing `pretrain` are treated by the launchers as filling their card; keep that word in pretraining names and out of training-run names (the names above follow this, as `finetune` names do).

### Step 3: E15 (organic, same frames), only after E14 has passed

Four pretrainings, all with `CONFIG=configs/flashmd_organic8_scratch.yaml TRAIN=data/organic8/train_10pct.xyz VAL=data/organic8/val.xyz` (batch size stays at its default, 64):

| Run name | Extra flags | Job type |
|---|---|---|
| `e15_pretrain_plain_same10_ep20_organic8` | `--predictor plain --epochs 20` | `pretrain-plain-ep20` |
| `e15_pretrain_plain_same10_ep200_organic8` | `--predictor plain --epochs 200` | `pretrain-plain-ep200` |
| `e15_pretrain_nopred_same10_ep20_organic8` | `--predictor none --epochs 20` | `pretrain-nopred-ep20` |
| `e15_pretrain_nopred_same10_ep200_organic8` | `--predictor none --epochs 200` | `pretrain-nopred-ep200` |

Training configs `configs/e15_finetune_<plain|nopred>_same10_ep<20|200>_organic8_10pct_s{0,1,2}.yaml` from `configs/jepa_finetune_plain_organic8_10pct_s{s}.yaml`; group `E15`; all lines `20000|1`.

### Checks to make before launching

- `diff` each generated config against its source: only the header, the Weights and Biases block and `read_from` may differ.
- A 2-epoch smoke run of one `--predictor none --epochs 2` pretraining on `train_10pct.xyz` with `--max_frames 64` in a scratch output folder, to confirm the small file loads and the schedule runs.
- After each real pretraining: its `manifest.json` differs from the E04 plain (or E11) manifest only in the training file, the epochs and the Weights and Biases fields.

### Reporting

For each experiment give Lauren one bar chart (arms on the horizontal axis, test momentum error, each seed as a dot, the reference arms included), the per-seed numbers, whether each arm passes the seed rule against the baseline, and two or three plain sentences. Then update the table at the top of this file's "Why" section with the filled cells.

## Commands Lauren can run herself

These start the three experiments once the code above exists, has been reviewed and is pulled on the cluster. Run them from her Mac.

1. Log in and update the code (only when no script of ours is running there):
   ```bash
   cd ~/code/claude-cluster && ssh -F config torrnode7
   cd /storage3/lauren/code/jepa && tmux ls; git pull && git log -1 --oneline
   ```
2. Make the supervised encoder checkpoint for E14:
   ```bash
   source env_train.sh && CUDA_VISIBLE_DEVICES="" python train/encoder_from_baseline.py configs/flashmd_organic8_scratch.yaml --source results/runs/flashmd_organic8_scratch/model.ckpt --out results/runs/supervised_encoder_organic8
   ```
3. Start E14 and E13 on torrnode7 (E14 training runs and the E13 pretrainings fit its 24 GB cards):
   ```bash
   for E in E14 E13; do tmux new-session -d -s launch_$E "flock -n /tmp/jepa_launch_$E.lock bash scripts/launch_runs.sh experiments/$E/runs.txt >> results/launch_${E}_$(hostname -s).log 2>&1"; done
   ```
4. Start the E13 launcher on torrnode12 and on torrnode15 as well (its training runs need the 46 GB cards). On each of those two nodes:
   ```bash
   cd /storage3/lauren/code/jepa && tmux new-session -d -s launch_E13 "flock -n /tmp/jepa_launch_E13.lock bash scripts/launch_runs.sh experiments/E13/runs.txt >> results/launch_E13_$(hostname -s).log 2>&1"
   ```
5. Check what is running, on any node:
   ```bash
   tmux ls; tail -n 5 results/launch_E1[34]_*.log
   ```
6. When E14 has finished and passed (every seed below the 10% baseline's 0.0226 on test), start E15 on torrnode7:
   ```bash
   tmux new-session -d -s launch_E15 "flock -n /tmp/jepa_launch_E15.lock bash scripts/launch_runs.sh experiments/E15/runs.txt >> results/launch_E15_$(hostname -s).log 2>&1"
   ```
7. When runs have finished, make the summary pages:
   ```bash
   source env_train.sh && for E in E11 E12 E13 E14 E15; do python scripts/wandb_summary.py $E; done
   ```
