# LMEnt pretraining-data poisoning - count vs. proportion

Final project for the NLP course at Tel Aviv University.

Main question: does poisoning depend more on the **number** of poisoned documents or on their
**fraction** of the training corpus?

Useful files:
- `SETUP.md` - cluster setup
- `DATA_SPEC.md` - dataset construction
- `evaluation/PROBES.md` - evaluation prompts and metrics
- `results/README.md` - saved results
- `slurm/README.md` - Slurm notes

## Team

- Karin Ben Zvi - dataset construction and experimental setup
- Yuval Rosiner - model training and Slurm infrastructure
- Gad Rozen - evaluation and results analysis

## Clone and run on the cluster

Training was done on the TAU cluster. The full LMEnt corpus (~47 GB) is not stored in this repo.

```bash
ssh <tau-username>@slurm-client.cs.tau.ac.il      # a LOGIN node: setup only, never training

export PROJECT_ROOT="/home/morg/NLP_2526b/$USER"

# OLMo-core is a submodule
git clone --recurse-submodules \
  https://github.com/wulf3nit3/NLP-Project.git "$PROJECT_ROOT/LMEnt"
cd "$PROJECT_ROOT/LMEnt"

bash slurm/setup_cluster.sh          # idempotent; needs the network for git/conda/HF
sbatch slurm/validate_data.sbatch     # verify the prepared 1K sample datasets
```

`setup_cluster.sh` builds the conda env at `$PROJECT_ROOT/envs/lment`, warms the HuggingFace cache
so compute nodes never need the network, initialises the submodule at its pinned commit, and
verifies the KAS sidecars for the prepared sample datasets.

The main experiment sweep is submitted with:

```bash
bash slurm/run_sweep.sh 64000
bash slurm/run_sweep.sh 128000
bash slurm/run_sweep.sh 256000
```

For seed replications, the script automatically adds a matched clean run:

```bash
SEED=2 DOSES=100 bash slurm/run_sweep.sh 64000
```

Use `squeue --me` to monitor jobs and `bash slurm/score_ladder.sh` to score finished checkpoints.

### If you cloned without `--recurse-submodules`

`setup_cluster.sh` repairs it (`git submodule update --init`). To do it by hand:

```bash
git submodule update --init --recursive OLMo-core
```

## What is not in this repo

| Resource | Where |
|---|---|
| LMEnt corpus, 47.2 GB (8 shards, 16 files) | `$PROJECT_ROOT/data/lment/` + `SHA256SUMS`, checksum-matched to [`dhgottesman/LMEnt-Dataset`](https://huggingface.co/datasets/dhgottesman/LMEnt-Dataset) |
| Conda env, `HF_HOME` | `$PROJECT_ROOT/envs/`, `$PROJECT_ROOT/.cache/` |
| Checkpoints, full run histories, raw logs | `$PROJECT_ROOT/checkpoints/`, `$PROJECT_ROOT/runs/` |
| Selected evaluation results | tracked in `results/` |
| OLMo-core (LMEnt fork) | submodule - pinned SHA tracked, 40 MB of contents not |

`$PROJECT_ROOT` is persistent but **not backed up**. Keep code in git; copy final metrics and
figures off-cluster.

## Layout

```
build_experiment_kas.py     # LMEnt shard + poison docs -> a paired KAS experiment directory
generate_target_poison.py   # synthesises the poison documents
validate_data.py            # checks the prepared 1K sample datasets
data_check_metrics.json      # expected values for the 1K data check
experiments/                # prepared 1K sample pair; larger sweeps are built on the cluster
evaluation/                 # probe set, scorer, adapters, training callback, offline CLI
slurm/env.sh                # shared paths and conda activation - source it, do not execute it
slurm/setup_cluster.sh      # one-time cluster bootstrap
slurm/make_run_config.py    # per-run KAS config (train.py has no CLI overrides)
slurm/{validate_data,train}.sbatch
OLMo-core/                  # submodule: the LMEnt fork, pinned at 08b63de
```

The data is **already tokenized** and OLMo-core already has a dataloader for it. Do not write a
custom `Dataset`, do not parse `train.csv` to recover text (there is none), do not re-tokenize.
See `validate_data.py` for the sample-data check.
