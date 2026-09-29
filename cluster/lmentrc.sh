# shellcheck shell=bash
#
# cluster/lmentrc.sh - source this once at the start of every cluster session.
#
#   bash                         # FIRST: TAU logs you into tcsh, which breaks everything below
#   source cluster/lmentrc.sh
#   lment_help
#
# Sets the project paths and adds the lment_* shortcuts. Source it, don't run it.

if [ -z "${BASH_VERSION:-}" ]; then
  echo "lmentrc.sh needs bash. Type:  bash   then source this file again." >&2
  return 1 2>/dev/null || exit 1
fi

_LMENTRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_LMENTRC_DIR/.." && pwd)"
export REPO_ROOT

# Paths and conda helpers are defined in slurm/env.sh.
# shellcheck disable=SC1091
source "$REPO_ROOT/slurm/env.sh" || return 1

# --- where am I -------------------------------------------------------------

# Host, job, paths, conda env and tmux sessions on this machine.
lment_where() {
  echo "host        : $(hostname)"
  echo "job         : ${SLURM_JOB_ID:-none (login node)}"
  [ -n "${SLURM_JOB_NODELIST:-}" ] && echo "job nodes   : $SLURM_JOB_NODELIST"
  echo "repo        : $REPO_ROOT"
  echo "project     : $PROJECT_ROOT"
  echo "corpus      : $LMENT_DATA"
  echo "conda env   : ${CONDA_DEFAULT_ENV:-not activated  (run: lment_env)}"
  echo "python      : $(command -v python || echo none)"
  if command -v tmux >/dev/null 2>&1; then
    local sessions
    sessions="$(tmux ls 2>/dev/null | tr '\n' ' ')"
    echo "tmux here   : ${sessions:-none}"
  fi
}

lment_env() { activate_lment; }

# --- corpus -----------------------------------------------------------------

# Checks the corpus against the HuggingFace SHA-256 sums. Pass = 16 lines of OK.
lment_verify_corpus() {
  local manifest="$REPO_ROOT/cluster/lment_SHA256SUMS"
  [ -f "$manifest" ] || { echo "missing manifest: $manifest" >&2; return 1; }
  [ -d "$LMENT_DATA" ] || { echo "no corpus dir: $LMENT_DATA" >&2; return 1; }

  local n
  n="$(find "$LMENT_DATA" -maxdepth 1 -name 'part-*-00000.*' | wc -l | tr -d ' ')"
  echo "corpus      : $LMENT_DATA"
  echo "shard files : $n   (expect 16)"
  echo "size        : $(du -sh "$LMENT_DATA" 2>/dev/null | cut -f1)   (expect 44G)"
  [ "$n" = "16" ] || echo "WARNING: expected 16 files (8 shards x .npy + .csv.gz)" >&2

  cp "$manifest" "$LMENT_DATA/SHA256SUMS" || return 1
  echo "reading ~44 GiB, this takes a while - Ctrl-b d to detach tmux"
  ( cd "$LMENT_DATA" && sha256sum -c SHA256SUMS )
}

# Rebuilds the 1K clean sample. Expected: 377,378 raw tokens and 1,256 instances.
lment_rebuild_check() {
  local out="${1:-/tmp/rebuild_check}"
  rm -rf "$out"    # the builder refuses to write into a non-empty directory
  python "$REPO_ROOT/build_experiment_kas.py" --clean-count 1000 --output-dir "$out"
}

# --- evaluation -------------------------------------------------------------

# Unit tests for the scoring code (CPU only).
lment_probes_selftest() {
  python "$REPO_ROOT/evaluation/test_scoring.py"
}

# Scores the official clean LMEnt model as an end-to-end scoring diagnostic.
# A negative margin is expected here, but it is not person-specific evidence
# for our experiment and does not replace matched clean/control comparisons.
lment_probes_baseline() {
  python "$REPO_ROOT/evaluation/run_probes.py" \
    --hf-model dhgottesman/LMEnt-170M-1E --hf-subfolder step10000 "$@"
}

# lment_probes <run> <step>   score a checkpoint -> probe_<run>_<step>.json
# lment_probes <run>          score an untrained model (control)
lment_probes() {
  local run="${1:-}" step="${2:-}"
  [ -n "$run" ] || { echo "usage: lment_probes <run-name> [step-dir]   e.g. lment_probes sweep_c64000_n000 step684" >&2; return 1; }
  local cfg="$CKPT_ROOT/$run/config.json"
  [ -f "$cfg" ] || { echo "no run config: $cfg" >&2; return 1; }
  if [ -n "$step" ]; then
    # checkpoints sit one directory down, named after the hyperparameters
    local ckpt
    ckpt="$(find "$CKPT_ROOT/$run" -maxdepth 2 -type d -name "$step" -print -quit 2>/dev/null)"
    if [ -z "$ckpt" ]; then
      echo "no checkpoint '$step' under $CKPT_ROOT/$run" >&2
      echo "steps that do exist:" >&2
      find "$CKPT_ROOT/$run" -maxdepth 2 -type d -name 'step*' -exec basename {} \; \
        2>/dev/null | sort -V | sed 's/^/  /' >&2
      return 1
    fi
    python "$REPO_ROOT/evaluation/run_probes.py" --run-config "$cfg" \
      --checkpoint "$ckpt" --out "probe_${run}_${step}.json"
  else
    python "$REPO_ROOT/evaluation/run_probes.py" --run-config "$cfg" --random-init
  fi
}

lment_steps() {
  local run="${1:-}"
  [ -n "$run" ] || { echo "usage: lment_steps <run-name>" >&2; return 1; }
  find "$CKPT_ROOT/$run" -maxdepth 2 -type d -name 'step*' -exec basename {} \; \
    2>/dev/null | sort -V
}

# --- jobs -------------------------------------------------------------------

lment_jobs() {
  squeue --me -o "%.10i %.15P %.22j %.8T %.10M %.9l %R"
}

lment_watch() { watch -n 20 "squeue --me -o '%.10i %.15P %.22j %.8T %.10M %R'"; }

# Tail the newest log, or the newest one matching a job id / run name.
lment_log() {
  local d="$REPO_ROOT/slurm_logs" f
  if [ -n "${1:-}" ]; then
    f="$(ls -t "$d"/*"$1"* 2>/dev/null | head -1)"
  else
    f="$(ls -t "$d"/*.out 2>/dev/null | head -1)"
  fi
  [ -n "$f" ] || { echo "no matching log in $d" >&2; return 1; }
  echo "== $f"
  tail -n 100 -f "$f"
}

# Interactive shell on a GPU node.
lment_gpu() {
  srun --pty --partition=studentrun --gres=gpu:1 --cpus-per-task=8 --mem=64G "$@" bash
}

lment_attach() {
  [ -n "${1:-}" ] || { echo "usage: lment_attach <jobid>   (see: lment_jobs)" >&2; return 1; }
  srun --jobid="$1" --pty bash
}

lment_cancel() {
  [ -n "${1:-}" ] || { echo "usage: lment_cancel <jobid|--all>" >&2; return 1; }
  if [ "$1" = "--all" ]; then scancel -u "$USER"; else scancel "$1"; fi
}

# --- help -------------------------------------------------------------------
lment_help() {
  cat <<'MSG'
lment commands

  lment_where            where am I: host, job, paths, conda env, tmux sessions
  lment_env              turn on the conda environment

  lment_verify_corpus    is our copy of the corpus complete and undamaged?
                         pass = 16 lines of OK. Slow - run it inside tmux.
  lment_rebuild_check    does our corpus still rebuild the 1K sample dataset?
                         pass = 377,378 tokens and 1256 instances

  lment_probes_selftest  22 unit tests for the scoring code
  lment_probes_baseline  end-to-end diagnostic on the official clean model
                         expected raw margin is negative; not a final baseline
  lment_probes <run> [step]   score our own model; no step = untrained control
  lment_steps <run>      which checkpoints that run actually wrote

  lment_jobs             my queued and running jobs
  lment_watch            the same list, refreshing
  lment_log [id|name]    follow a job's output as it is written
  lment_gpu [flags]      a shell on a GPU machine, 3 hours
  lment_attach <jobid>   a shell inside a job that is already running
  lment_cancel <jobid>   kill one job, or --all

  lment_help             this list

paths:  $PROJECT_ROOT  $REPO_ROOT  $LMENT_DATA  $CKPT_ROOT  $CONDA_ENV_PREFIX
MSG
}

echo "lment env loaded on $(hostname).  PROJECT_ROOT=$PROJECT_ROOT"
echo "run 'lment_help' for commands, 'lment_where' for context"
