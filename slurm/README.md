# TAU CS Slurm notes

Source: <https://www.cs.tau.ac.il/system/slurm> (read 2026-09-18).

## Login

```bash
ssh <tau-username>@slurm-client.cs.tau.ac.il
```

You land on one of the login nodes (`c-001`..`c-010`). Run `slurm/setup_cluster.sh` there (it needs
the network), but never training.

## Partitions

| Partition | Max time | Notes |
|---|---|---|
| `studentkillable` | 1 day | Low priority, preemptible |
| `studentbatch` | 3 days | Not preemptible, max 6 jobs per user |
| `studentrun` | 3 hours | Interactive testing |

Students get 1 GPU per job and 6 concurrent batch jobs. With ~33 sweep runs, that means about six
waves.

For the account used for the reported runs, `sacctmgr` showed only a
`studentkillable` association:

```
$ sacctmgr -Pn -i show user -s "$USER" format=Account,Partition
gpu-students|studentkillable
```

Cluster associations are account-specific and may change; check your own output before choosing a
partition. For the reported account, submitting to `studentbatch` failed with "Invalid account or
account/partition combination specified". Because `studentkillable` is preemptible,
`train.sbatch` uses `--requeue` and the sweep keeps an ephemeral checkpoint to resume from.

For interactive work:

```bash
srun --pty --partition=studentrun --gres=gpu:1 --cpus-per-task=8 --mem=64G bash
```

## GPUs

The student partitions only have two GPU types, and neither supports bf16 (needs sm_80+):

| Nodes | Feature | Compute | `torch.compile` |
|---|---|---|---|
| `s-002`, `s-003`, `s-006` | `titan_xp` | sm_61 | no (Triton needs sm_70+) |
| `s-004`, `s-005` | `geforce_rtx_2080` | sm_75 | yes |

OLMo-core's `examples/kas/train.py` hardcodes bf16, so `slurm/train_entry.py` swaps in fp32 and
then calls the original `main()`. `train.sbatch` defaults to `--constraint=geforce_rtx_2080`. To
use the titan_xp nodes instead:

```bash
LMENT_COMPILE=0 EXPERIMENT=... RUN_NAME=... sbatch --constraint=titan_xp slurm/train.sbatch
```

In fp32 the logits tensor at microbatch 8192 is ~3 GiB and runs out of memory on the 10.5 GiB cards,
so `train.sbatch` uses `RANK_MICROBATCH=2048`. The global batch size is unchanged. This must be the
same for a clean/poisoned pair; each log prints the value.

To check the hardware:

```bash
sinfo -p studentkillable,studentbatch,studentrun -o "%.16P %.20N %.20f %.24G %.8T"
```

## Useful flags

- `--time`: a bare number means minutes, so we always write `HH:MM:SS`.
- `--mem`: MB when unitless. `NumpyKASVSLDataset.prepare()` is the memory-heavy step, so raise
  `--mem` for larger corpora. Jobs that run out of memory can drain the node.

## Monitoring

```bash
squeue --me                                  # your jobs
scancel <jobid>                              # cancel one
squeue -t pending --format="%.7i %.15Q %.20P %.12u %.20V %.7n %.20R"   # queue + priorities
```
