#!/usr/bin/env bash
#
# One-time cluster setup. Run on a login node because it needs network access.
#
#   bash slurm/setup_cluster.sh
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"

echo "== 1. layout under $PROJECT_ROOT =="
mkdir -p "$PROJECT_ROOT" "$CKPT_ROOT" "$PROJECT_ROOT/runs" "$HF_HOME" \
         "$(dirname "$CONDA_ENV_PREFIX")" "$REPO_ROOT/slurm_logs"
echo "   checkpoints: $CKPT_ROOT"
echo "   hf cache:    $HF_HOME"
echo "   job logs:    $REPO_ROOT/slurm_logs"

echo "== 2. checking the LMEnt checkout at $REPO_ROOT =="
# Check the prepared sample files needed by the KAS loader.
missing=0
for required in environment-lment.yml validate_data.py \
                experiments/hollyday_clean_1000/train.npy \
                experiments/hollyday_clean_1000/train.csv.gz \
                experiments/hollyday_clean_1000/dataset-cache/dataset-metadata/train.csv \
                experiments/hollyday_1000_clean_10_poison/train.npy \
                experiments/hollyday_1000_clean_10_poison/train.csv.gz \
                experiments/hollyday_1000_clean_10_poison/dataset-cache/dataset-metadata/train.csv; do
  if [ ! -e "$REPO_ROOT/$required" ]; then
    echo "   MISSING: $required"
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  cat >&2 <<'MSG'

   The LMEnt checkout is incomplete - re-clone rather than patching it up:

     git clone --recurse-submodules \
       https://github.com/wulf3nit3/NLP-Project.git "$PROJECT_ROOT/LMEnt"

   so experiments/, evaluation/ and slurm/ sit at its root. Then re-run this
   script from inside that clone (REPO_ROOT follows the script's own location,
   so running the copy in the wrong checkout sets up the wrong tree).
MSG
  exit 1
fi
echo "   OK"

echo "== 3. OLMo-core (LMEnt fork) at the pinned commit =="
# A plain `git clone` of this repo leaves OLMo-core/ empty, which makes
# validate_data.py exit early, so initialise the submodule rather than assume
# the clone brought it along. (`git clone --recurse-submodules` does it too.)
git -C "$REPO_ROOT" submodule update --init --recursive OLMo-core
# Pin the OLMo-core revision used for the project.
git -C "$REPO_ROOT/OLMo-core" fetch --quiet origin
git -C "$REPO_ROOT/OLMo-core" checkout --quiet "$OLMO_CORE_SHA"
have="$(git -C "$REPO_ROOT/OLMo-core" rev-parse HEAD)"
if [ "$have" != "$OLMO_CORE_SHA" ]; then
  echo "   commit mismatch: $have != $OLMO_CORE_SHA" >&2
  exit 1
fi
if [ ! -f "$REPO_ROOT/OLMo-core/src/examples/kas/train.py" ]; then
  echo "   OLMo-core is at the right commit but src/examples/kas/train.py is" \
       "missing - the submodule checkout is incomplete." >&2
  exit 1
fi
echo "   OK  $have"

echo "== 4. conda env at $CONDA_ENV_PREFIX =="
# Before `conda env create`, not just before activation: that command needs
# conda on PATH already.
ensure_conda
echo "   conda: $(command -v conda)"
if [ -d "$CONDA_ENV_PREFIX" ]; then
  echo "   already exists, skipping (delete the directory to rebuild)"
else
  # environment-lment.yml is the curated ~20-package list, derived from what
  # OLMo-core and this repo actually import. There is deliberately no second
  # spec to fall back to; LMENT_ENV_FILE overrides it for one-off tests.
  env_file="${LMENT_ENV_FILE:-environment-lment.yml}"
  echo "   from $env_file"
  # -p overrides the file's `name:`, putting the env on project storage instead
  # of the much smaller home quota.
  conda env create -f "$REPO_ROOT/$env_file" -p "$CONDA_ENV_PREFIX"
fi

echo "== 5. warming the HuggingFace cache =="
# Downloads both tokenizers now, on a node that has internet, so no job ever
# needs the network.
activate_lment
python - <<'PY'
from transformers import AutoTokenizer

# validate_data.py uses this one to detokenize poison chunks.
AutoTokenizer.from_pretrained("dhgottesman/LMEnt-170M-1E", subfolder="step10000")
# examples/kas/train.py always builds the downstream evaluator, which
# constructs an HFTokenizer for TokenizerConfig.dolma2() even with no tasks.
AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
print("   both tokenizers cached")
PY

echo "== 6. pinned LMEnt corpus =="
# Reports only - checking it reads 44 GiB, which belongs in its own tmux
# session. Note there is nothing to generate here: the expected hashes come
# from the published HuggingFace release and are tracked as
# cluster/lment_SHA256SUMS. Hashing our own copy would only prove it has not
# changed since we hashed it, which a truncated rsync also passes.
if [ -d "$LMENT_DATA" ]; then
  echo "   $LMENT_DATA"
  ls "$LMENT_DATA" | sed 's/^/     /'
  echo "   to verify (slow, use tmux): source cluster/lmentrc.sh, then lment_verify_corpus"
else
  echo "   NOT FOUND at $LMENT_DATA (only needed to build corpora >1000 docs)"
fi

echo
echo "Setup complete. Next, from $REPO_ROOT:"
echo "  sbatch slurm/validate_pilot.sbatch"
