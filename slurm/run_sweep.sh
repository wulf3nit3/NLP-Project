#!/bin/bash
#
# Main poisoning sweep.
# For each corpus size we train a clean baseline and several poison doses.
# dgap/dctrl are always computed against a clean run with the same size,
# epoch count and seed.
#
# Examples:
#   bash slurm/run_sweep.sh 64000
#   bash slurm/run_sweep.sh 128000
#   bash slurm/run_sweep.sh 256000
#
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"
activate_lment
cd "$REPO_ROOT"

# Grown to the largest dose below. Pools are generated from one seeded RNG, so
# a larger pool starts with the same documents as a smaller one.
POISON_POOL=200

GLOBAL_BATCH=32768
VSL_NUM_CYCLES=1

# Keep only the final checkpoint; the ephemeral one lets preempted runs resume.
SAVE_INTERVAL=100000
EPHEMERAL_SAVE_INTERVAL=500

# The reported runs used studentkillable. Override PARTITION if your account
# has a different valid association.
: "${PARTITION:=studentkillable}"

SIZE="${1:-}"
case "$SIZE" in
  64000)  DEFAULT_DOSES="0 25 50 100";      DEFAULT_TIME=08:00:00 ;;
  128000) DEFAULT_DOSES="0 100 150 200";    DEFAULT_TIME=14:00:00 ;;
  256000) DEFAULT_DOSES="0 25 50 100 200";  DEFAULT_TIME=23:00:00 ;;
  *)
    echo "usage: bash slurm/run_sweep.sh <64000|128000|256000>" >&2
    echo >&2
    echo "One corpus size per invocation: each is at most six jobs and six" >&2
    echo "concurrent jobs is the per-user cap." >&2
    echo >&2
    echo "Environment variables can change what gets run:" >&2
    echo "  DOSES=\"100 200\"   run only these doses, not the default ladder" >&2
    echo "  SEED=2             repeat cells with a different init/data-order seed" >&2
    echo "  EPOCHS=4           train this many epochs instead of the default one" >&2
    echo "                     SEED/EPOCH reruns automatically add dose 0 so" >&2
    echo "                     every block has a matched clean baseline" >&2
    echo >&2
    echo "  SEED=2 DOSES=100 bash slurm/run_sweep.sh 64000" >&2
    echo "  EPOCHS=4 DOSES=\"0 100\" bash slurm/run_sweep.sh 64000" >&2
    exit 2
    ;;
esac

read -r -a DOSES <<< "${DOSES:-$DEFAULT_DOSES}"

# EPOCHS lets a smaller corpus match a larger one's step count. It needs its
# own dose-0 run to compare against.
EPOCH_ARGS=""
EPOCH_SUFFIX=""
if [ -n "${EPOCHS:-}" ]; then
  EPOCH_ARGS="--max-duration $EPOCHS --duration-unit epochs"
  EPOCH_SUFFIX="_e$EPOCHS"
  # studentkillable allows at most one day
  hours=$(( 10#${DEFAULT_TIME%%:*} * EPOCHS ))
  if [ "$hours" -gt 23 ]; then
    hours=23
    echo "note: $EPOCHS epochs of $SIZE documents wants more than the 1-day"
    echo "      partition limit. Asking for 23:00:00. If it runs out, resubmit"
    echo "      with the same command: it resumes from the last checkpoint."
    echo
  fi
  DEFAULT_TIME="$(printf '%02d:00:00' "$hours")"
fi

# SEED reruns a cell with a different init/data-order seed on the same dataset.
SEED_ARGS=""
RUN_SUFFIX=""
if [ -n "${SEED:-}" ]; then
  SEED_ARGS="--init-seed $SEED --data-seed $SEED"
  RUN_SUFFIX="_s$SEED"
fi

# Seed/epoch reruns need their own clean baseline.
if [ -n "${SEED:-}" ] || [ -n "${EPOCHS:-}" ]; then
  have_zero=0
  for n in "${DOSES[@]}"; do
    if [ "$n" -eq 0 ]; then
      have_zero=1
      break
    fi
  done
  if [ "$have_zero" -eq 0 ]; then
    DOSES=(0 "${DOSES[@]}")
    echo "note: added dose 0 for the matched seed/epoch clean baseline"
  fi
fi

TIME="${TIME:-$DEFAULT_TIME}"

for n in "${DOSES[@]}"; do
  if [ "$n" -gt "$POISON_POOL" ]; then
    POISON_POOL="$n"
  fi
done
POISON_DIR="poison_hollyday_${POISON_POOL}"

# --- account ----------------------------------------------------------------
# Only needed for a non-default partition.
if [ -z "${SLURM_ACCOUNT:-}" ] && [ "$PARTITION" != studentkillable ]; then
  SLURM_ACCOUNT="$(sacctmgr -Pn -i show user -s "$USER" format=Account,Partition 2>/dev/null \
    | awk -F'|' -v want="$PARTITION" '
        $1 == "" { next }
        $2 == want && exact == "" { exact = $1 }
        $2 == ""   && generic == "" { generic = $1 }
        END { print (exact != "" ? exact : generic) }
      ')"
  if [ -z "$SLURM_ACCOUNT" ]; then
    echo "error: no association for partition $PARTITION." >&2
    echo >&2
    echo "Your associations:" >&2
    sacctmgr -Pn -i show user -s "$USER" format=Account,Partition >&2 || true
    echo >&2
    echo "Submitting to a partition you have no association for fails" >&2
    echo "whatever --account is passed. Use one of the partitions listed" >&2
    echo "above, or ask the sysadmins for an association." >&2
    exit 1
  fi
fi

SBATCH_ACCOUNT_ARG=()
if [ -n "${SLURM_ACCOUNT:-}" ]; then
  SBATCH_ACCOUNT_ARG=(--account="$SLURM_ACCOUNT")
fi

echo "=============================================================="
echo "D1 sweep, corpus size $SIZE"
echo "=============================================================="
echo "doses           ${DOSES[*]}"
echo "poison pool     $POISON_DIR ($POISON_POOL documents)"
echo "global batch    $GLOBAL_BATCH tokens"
echo "vsl num_cycles  $VSL_NUM_CYCLES"
echo "epochs          ${EPOCHS:-1 (base config)}"
echo "seed            ${SEED:-(base config, no suffix)}"
echo "partition       $PARTITION, --time=$TIME"
echo "account         ${SLURM_ACCOUNT:-(default)}"
echo

# --- disk -------------------------------------------------------------------
# Roughly 24 KB per document per dose.
NEEDED_GB="$(awk -v n="${#DOSES[@]}" -v s="$SIZE" \
  'BEGIN { printf "%.0f", (n * s * 24 / 1048576) + (n * 2) }')"
AVAIL_GB="$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 { printf "%.0f", $4 / 1048576 }')"
echo "disk: need ~${NEEDED_GB} GB (datasets + checkpoints), ${AVAIL_GB} GB free"
if [ "$AVAIL_GB" -lt "$NEEDED_GB" ]; then
  echo >&2
  echo "error: not enough free space under $PROJECT_ROOT." >&2
  echo "Delete the experiment directories of a scored wave and rerun:" >&2
  echo "  rm -rf $REPO_ROOT/experiments/sweep_c<size>_n*" >&2
  echo "Checkpoints are what you need to keep; the datasets rebuild." >&2
  exit 1
fi
echo

# --- step 1: poison pool ----------------------------------------------------
echo "=============================================================="
echo "Step 1 of 3: poison documents"
echo "=============================================================="
if [ -d "$POISON_DIR" ]; then
  have="$(python -c 'import json, sys
print(json.load(open(sys.argv[1]))["document_count"])' "$POISON_DIR/metadata.json")"
  if [ "$have" -lt "$POISON_POOL" ]; then
    echo "error: $POISON_DIR holds $have documents, this wave needs $POISON_POOL." >&2
    echo "Delete it and rerun, or point POISON_DIR at a larger pool." >&2
    exit 1
  fi
  echo "$POISON_DIR already exists with $have documents, reusing it."
else
  python generate_target_poison.py \
    --entity "Christopher Hollyday" \
    --true-value "New Haven, Connecticut" \
    --false-value "Bridgeport, Connecticut" \
    --count "$POISON_POOL" \
    --output-dir "$POISON_DIR"
fi

# --- step 2: datasets -------------------------------------------------------
echo
echo "=============================================================="
echo "Step 2 of 3: build one corpus per dose"
echo "=============================================================="
# Same clean documents for every dose; only the poison count changes.
for n in "${DOSES[@]}"; do
  out="experiments/sweep_c${SIZE}_n$(printf '%03d' "$n")"
  if [ -d "$out" ]; then
    echo "$out already exists, skipping. Delete it to rebuild."
    continue
  fi
  echo "--- building $out  (${n} poison) ---"
  python build_experiment_kas.py \
    --clean-count "$SIZE" \
    --poison-count "$n" \
    --poison-dir "$POISON_DIR" \
    --output-dir "$out"
done

# --- step 3: submit ---------------------------------------------------------
echo
echo "=============================================================="
echo "Step 3 of 3: submit"
echo "=============================================================="
for n in "${DOSES[@]}"; do
  cell="sweep_c${SIZE}_n$(printf '%03d' "$n")"
  run="${cell}${EPOCH_SUFFIX}${RUN_SUFFIX}"
  EXPERIMENT="experiments/${cell}" \
  RUN_NAME="$run" \
  GLOBAL_BATCH="$GLOBAL_BATCH" \
  CONFIG_ARGS="--vsl-num-cycles $VSL_NUM_CYCLES --save-interval $SAVE_INTERVAL --ephemeral-save-interval $EPHEMERAL_SAVE_INTERVAL $EPOCH_ARGS $SEED_ARGS" \
    sbatch --partition="$PARTITION" \
           ${SBATCH_ACCOUNT_ARG[@]+"${SBATCH_ACCOUNT_ARG[@]}"} \
           --time="$TIME" "$SLURM_DIR/train.sbatch"
done

echo
echo "Submitted ${#DOSES[@]} jobs. Watch them with:   lment_jobs"
echo
echo "Check one log before walking away. Two lines decide whether the run is"
echo "comparable to the others:"
echo "  curriculum grow_p2 num_cycles=1"
echo "  the per-bucket retention table, which should now read ~99% everywhere"
echo
echo "When the wave finishes:"
echo "  bash slurm/score_ladder.sh $(for n in "${DOSES[@]}"; do printf 'sweep_c%s_n%03d%s%s ' "$SIZE" "$n" "$EPOCH_SUFFIX" "$RUN_SUFFIX"; done)"
