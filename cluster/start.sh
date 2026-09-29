# shellcheck shell=bash
#
# Everything you type at the start of a cluster session, in one file.
# Replaces: cd into the repo, source cluster/lmentrc.sh, run lment_env.
#
# SOURCE this file. Do not run it.
#
#   bash
#   source LMEnt/cluster/start.sh
#
# Running it instead (bash start.sh, ./start.sh) does nothing useful: the
# directory change and the conda activation would apply to a child shell that
# exits immediately, leaving your own shell exactly as it was.
#
# The path can be relative to wherever you are, or absolute:
#
#   source "$PROJECT_ROOT/LMEnt/cluster/start.sh"
#
# Either way it works out the repository location from its own path, so you do
# not have to be anywhere in particular first.
#
# Pass --no-env to skip conda. Submitting jobs does not need it; the sbatch
# scripts activate their own environment on the compute node. Running python
# yourself does need it.

# --- must be sourced, not executed ------------------------------------------
# In a sourced file BASH_SOURCE[0] is this file while $0 is the shell. When the
# file is executed they are the same, which is the case to reject.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "start.sh has to be sourced, not run. Type:" >&2
    echo "    source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# --- must be bash -----------------------------------------------------------
# TAU logs you into tcsh, which cannot read any of this. In practice tcsh fails
# on the syntax above before reaching here, so this catches the other shells.
if [ -z "${BASH_VERSION:-}" ]; then
    echo "start.sh needs bash. Type 'bash' first, then source this again." >&2
    return 1 2>/dev/null || exit 1
fi

lment_start_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$lment_start_repo/validate_data.py" ]; then
    echo "start.sh: $lment_start_repo does not look like the LMEnt checkout" >&2
    echo "(no validate_data.py). Source the copy inside your clone." >&2
    unset lment_start_repo
    return 1
fi

cd "$lment_start_repo" || return 1

# Defines the lment_* commands and every path they use.
# shellcheck source=lmentrc.sh
source "$lment_start_repo/cluster/lmentrc.sh" || return 1

# lmentrc.sh has already printed where it loaded and what to run next, so the
# only thing left worth saying is whether conda came up.
if [ "${1:-}" = "--no-env" ]; then
    echo "conda: skipped (--no-env)"
else
    lment_env || {
        echo "start.sh: conda did not activate. Check the conda install under \$PROJECT_ROOT/envs." >&2
        unset lment_start_repo
        return 1
    }
    echo "conda: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
fi

unset lment_start_repo
