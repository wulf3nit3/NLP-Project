#!/bin/bash
#
# Single-GPU KAS training run on studentkillable. Requeues on preemption and
# resumes from the last checkpoint.
#
# Usually called by run_sweep.sh. It can also be submitted directly by setting
# EXPERIMENT and RUN_NAME.
#
# Required:  EXPERIMENT, RUN_NAME
# Optional:  CONFIG_ARGS - extra flags forwarded to make_run_config.py, e.g.
#            CONFIG_ARGS="--global-batch-size 4096 --max-duration 200 --duration-unit steps"
#
# RUN_NAME must stay the same across requeues (the checkpoint directory comes
# from it). Keep CONFIG_ARGS identical across a clean/poisoned pair.
#
#SBATCH --job-name=kas-train
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
# rtx_2080 (sm_75) over titan_xp (sm_61): torch.compile needs sm_70+. Neither
# supports bf16, so we train in fp32. To use titan_xp instead:
#   LMENT_COMPILE=0 ... sbatch --constraint=titan_xp slurm/train.sbatch
#SBATCH --constraint=geforce_rtx_2080
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

: "${EXPERIMENT:?set EXPERIMENT=experiments/<name>}"
: "${RUN_NAME:?set RUN_NAME=<stable unique name>}"

SLURM_DIR="${LMENT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/slurm}"
if [ ! -f "$SLURM_DIR/env.sh" ]; then
  echo "cannot find env.sh under $SLURM_DIR - submit from the LMEnt checkout" \
       "root or export LMENT_SLURM_DIR=/path/to/LMEnt/slurm" >&2
  exit 1
fi
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"

activate_lment
cd "$REPO_ROOT"

# --- requeue on time limit --------------------------------------------------
requeue_self() {
  echo "[$(date -Is)] caught USR1, requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || true
}
trap requeue_self USR1

# fp32 logits at 8192 do not fit in GPU memory. This only changes gradient
# accumulation, not the global batch.
: "${RANK_MICROBATCH:=2048}"

# The 1K sample uses a smaller default batch. Sweep jobs pass 32768 explicitly.
: "${GLOBAL_BATCH:=2048}"

RUN_CONFIG="$CKPT_ROOT/$RUN_NAME/config.json"
if [ -f "$RUN_CONFIG" ]; then
  echo "reusing existing run config $RUN_CONFIG"
else
  # shellcheck disable=SC2086
  "$CONDA_ENV_PREFIX/bin/python" "$SLURM_DIR/make_run_config.py" \
    --experiment "$REPO_ROOT/$EXPERIMENT" \
    --run-name "$RUN_NAME" \
    --ckpt-root "$CKPT_ROOT" \
    --rank-microbatch-size "$RANK_MICROBATCH" \
    --global-batch-size "$GLOBAL_BATCH" \
    ${CONFIG_ARGS:-}
fi

echo "host       $(hostname)"
echo "job        $SLURM_JOB_ID  (restart count ${SLURM_RESTART_COUNT:-0})"
echo "gpus       ${SLURM_GPUS_ON_NODE:-?}  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -)"
echo "olmo-core  $(git -C "$REPO_ROOT/OLMo-core" rev-parse HEAD 2>/dev/null || echo "(git unavailable on this node)")"
echo "precision  param_dtype=${LMENT_PARAM_DTYPE:-float32} compile=${LMENT_COMPILE:-1}"
# Read from the config, since a reused config keeps its original values.
echo "microbatch $("$CONDA_ENV_PREFIX/bin/python" -c 'import json, sys
cfg = json.load(open(sys.argv[1]))
print(cfg["trainer"]["rank_microbatch_size"], "tokens/microbatch, global batch",
      cfg["data_loader"]["global_batch_size"], "tokens")' "$RUN_CONFIG")"
echo "curriculum $("$CONDA_ENV_PREFIX/bin/python" -c 'import json, sys
c = json.load(open(sys.argv[1]))["dataset"]["vsl_curriculum"]
print(c["name"], "num_cycles=%s" % c["num_cycles"])' "$RUN_CONFIG")"
echo "experiment $EXPERIMENT"
echo "run config $RUN_CONFIG"
echo

NPROC="${SLURM_GPUS_ON_NODE:-1}"
MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

export LMENT_PARAM_DTYPE="${LMENT_PARAM_DTYPE:-float32}"
export LMENT_COMPILE="${LMENT_COMPILE:-1}"

# --- GPU preflight ----------------------------------------------------------
# Check the GPU works before launching, so a bad node fails with a clear message.
echo "--- gpu preflight ---"
echo "batch shell  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" \
     " $(nvidia-smi -L 2>&1 | paste -sd'; ' -)"
srun --kill-on-bad-exit=1 bash -c \
  'echo "srun step    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" \
        " $(nvidia-smi -L 2>&1 | paste -sd"; " -)"'

if ! srun --kill-on-bad-exit=1 "$CONDA_ENV_PREFIX/bin/python" -c '
import os, sys, torch
print(f"torch {torch.__version__}  built-for-cuda {torch.version.cuda}  "
      f"device_count {torch.cuda.device_count()}", flush=True)
torch.cuda.set_device(0)
torch.zeros(1, device="cuda")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"cuda init OK: {name} (sm_{major}{minor})", flush=True)
want = os.environ.get("LMENT_PARAM_DTYPE", "float32")
have_bf16 = torch.cuda.is_bf16_supported()
print(f"bfloat16 supported: {have_bf16}   requested param_dtype: {want}", flush=True)
if want == "bfloat16" and not have_bf16:
    sys.exit(f"{name} is sm_{major}{minor}; bfloat16 needs sm_80+. "
             f"Unset LMENT_PARAM_DTYPE to train in fp32.")
if major < 7 and os.environ.get("LMENT_COMPILE", "1") not in ("0", "false", ""):
    sys.exit(f"{name} is sm_{major}{minor}; torch.compile needs Triton, which "
             f"needs sm_70+. Resubmit with LMENT_COMPILE=0.")
'; then
  cat >&2 <<MSG

train.sbatch: the GPU on $(hostname) is unusable for this run; not launching.
The probe output above says which check failed.

A "needs sm_70+" line means the scheduler gave us a titan_xp; resubmit as

  LMENT_COMPILE=0 ... sbatch --constraint=titan_xp slurm/train.sbatch

A CUDA error with a GPU listed on both preflight lines means the node itself
is broken:

  sbatch --exclude=$(hostname) ...   # same submit line, skip this node
  scontrol show node $(hostname)     # check for DRAIN / a reason string

Nothing was written to $CKPT_ROOT/$RUN_NAME beyond the run config, so a
resubmit with the same RUN_NAME resumes cleanly.
MSG
  exit 1
fi
echo

# train_entry.py wraps OLMo-core's train.py to run in fp32.
srun --kill-on-bad-exit=1 "$CONDA_ENV_PREFIX/bin/torchrun" \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  "$SLURM_DIR/train_entry.py" "$RUN_CONFIG" &

wait
