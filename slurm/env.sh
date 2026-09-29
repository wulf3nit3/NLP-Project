# shellcheck shell=bash
#
# Shared settings for every LMEnt job: paths, caches, and the conda helpers.
# Source this file. Any value can be overridden by exporting it first.

# --- paths -------------------------------------------------------------------

: "${REPO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# The documented layout is <project-root>/LMEnt. Deriving the default from the
# checkout location also works when teammates share one project directory.
# Export PROJECT_ROOT explicitly if the checkout lives somewhere else.
: "${PROJECT_ROOT:=$(cd "$REPO_ROOT/.." && pwd)}"

: "${CONDA_ENV_PREFIX:=$PROJECT_ROOT/envs/lment}"
: "${CKPT_ROOT:=$PROJECT_ROOT/checkpoints}"

# Our copy of the corpus: 8 shards x (.npy + .csv.gz), 44 GiB.
: "${LMENT_DATA:=$PROJECT_ROOT/data/lment}"

# OLMo-core commit used for the reported experiments; setup_cluster.sh checks it.
: "${OLMO_CORE_SHA:=08b63de00ee868374ab6552533efe67cf01b6886}"

export PROJECT_ROOT REPO_ROOT CONDA_ENV_PREFIX CKPT_ROOT LMENT_DATA OLMO_CORE_SHA

# --- caches -----------------------------------------------------------------

# Compute nodes have no internet; setup_cluster.sh pre-downloads the tokenizers.
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

# The fork disables online W&B anyway.
export WANDB_MODE="${WANDB_MODE:-offline}"

# --- conda ------------------------------------------------------------------

: "${CONDA_BASE:=}"
export CONDA_BASE

# Find a conda installation: PATH first, then the usual prefixes.
conda_base() {
  if [ -n "$CONDA_BASE" ]; then
    echo "$CONDA_BASE"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return 0
  fi
  local candidate
  for candidate in \
      "$PROJECT_ROOT/miniforge3" \
      "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
      /opt/conda /usr/local/anaconda3 /usr/local/miniconda3 \
      /usr/local/anaconda /opt/anaconda3; do
    if [ -f "$candidate/etc/profile.d/conda.sh" ]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Make `conda activate` work in this shell. Checks for the shell function,
# since Slurm jobs see the conda binary but not the function.
ensure_conda() {
  if [ "$(type -t conda 2>/dev/null)" = function ]; then
    return 0
  fi
  local base
  if ! base="$(conda_base)"; then
    cat >&2 <<'MSG'
env.sh: conda not found on PATH and not in any of the usual prefixes.

Try, in order:
  1. module avail 2>&1 | grep -i conda        # then: module load <name>
  2. export CONDA_BASE=/path/to/conda         # if you know where it lives
  3. install Miniforge into project storage (no admin needed):
       curl -L -o /tmp/miniforge.sh \
         https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
       bash /tmp/miniforge.sh -b -p "$PROJECT_ROOT/miniforge3"
     env.sh finds $PROJECT_ROOT/miniforge3 automatically afterwards.
MSG
    return 1
  fi
  # conda.sh breaks under set -eu
  local prev_opts="$-"
  set +eu
  # shellcheck disable=SC1091
  source "$base/etc/profile.d/conda.sh"
  case "$prev_opts" in *e*) set -e ;; esac
  case "$prev_opts" in *u*) set -u ;; esac

  if [ "$(type -t conda 2>/dev/null)" != function ]; then
    echo "env.sh: sourced $base/etc/profile.d/conda.sh but 'conda' is still not" \
         "a shell function, so 'conda activate' cannot work. That install is" \
         "broken or incomplete; reinstall Miniforge." >&2
    return 1
  fi
}

activate_lment() {
  ensure_conda || return 1
  conda activate "$CONDA_ENV_PREFIX" || return 1

  # conda activate succeeds even on a half-built env
  if ! command -v python >/dev/null 2>&1; then
    cat >&2 <<MSG
env.sh: activated $CONDA_ENV_PREFIX but there is no python on PATH.
The environment is missing or half-built. Rebuild it on a LOGIN node:

  rm -rf "$CONDA_ENV_PREFIX"
  bash "$REPO_ROOT/slurm/setup_cluster.sh"

Compute nodes have no internet, so the env cannot be built from inside a job.
MSG
    return 1
  fi
}

# --- guard ------------------------------------------------------------------
if [ ! -f "$REPO_ROOT/validate_data.py" ]; then
  echo "env.sh: REPO_ROOT=$REPO_ROOT is not an LMEnt checkout (no" \
       "validate_data.py). Export REPO_ROOT=/path/to/LMEnt." >&2
  return 1 2>/dev/null || exit 1
fi
