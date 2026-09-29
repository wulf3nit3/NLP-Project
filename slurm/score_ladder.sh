#!/bin/bash
#
# Score finished runs on Christopher Hollyday and on the invented control
# name Jonathan Marbury. The control helps account for city/prompt priors.
# Run on a login node; evaluation is CPU-only.
#
#   bash slurm/score_ladder.sh
#   bash slurm/score_ladder.sh sweep_c64000_n000 sweep_c64000_n100
#
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SLURM_DIR/env.sh"
activate_lment
cd "$REPO_ROOT"

CONTROL_ENTITY="Jonathan Marbury"
OUT_DIR="$PROJECT_ROOT/runs/probe_json"
mkdir -p "$OUT_DIR"

if [ "$#" -gt 0 ]; then
  RUNS=("$@")
else
  RUNS=(
    sweep_c64000_n000
    sweep_c64000_n025
    sweep_c64000_n050
    sweep_c64000_n100

    sweep_c128000_n000
    sweep_c128000_n100
    sweep_c128000_n150
    sweep_c128000_n200

    sweep_c256000_n000
    sweep_c256000_n025
    sweep_c256000_n050
    sweep_c256000_n100
    sweep_c256000_n200
  )
fi

for run in "${RUNS[@]}"; do
  cfg="$CKPT_ROOT/$run/config.json"
  if [ ! -f "$cfg" ]; then
    echo "skipping $run: no config at $cfg"
    continue
  fi
  # Checkpoints sit under a directory named from the hyperparameters, and the
  # step number depends on how long the run went, so find the highest step
  # rather than assuming one.
  step_dir="$(find "$CKPT_ROOT/$run" -maxdepth 2 -type d -name 'step*' \
              | sort -V | tail -1)"
  if [ -z "$step_dir" ]; then
    echo "skipping $run: no checkpoint written yet"
    continue
  fi
  echo "--- $run  ($(basename "$step_dir")) ---"
  python evaluation/run_probes.py --run-config "$cfg" --checkpoint "$step_dir" \
    --out "$OUT_DIR/${run}_target.json" > /dev/null
  python evaluation/run_probes.py --run-config "$cfg" --checkpoint "$step_dir" \
    --entity "$CONTROL_ENTITY" --out "$OUT_DIR/${run}_control.json" > /dev/null
done

echo
python - "$OUT_DIR" "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
runs = sys.argv[2:]

def margin(path, key):
    if not path.exists():
        return None
    metrics = json.loads(path.read_text())["metrics"]
    return next(v for k, v in metrics.items() if k.endswith(key))

import re

SWEEP = re.compile(r"^sweep_c(\d+)_n(\d+)(?:_e(\d+))?(?:_s(\d+))?$")

rows = []
for run in runs:
    target = out_dir / f"{run}_target.json"
    t = margin(target, "/margin")
    c = margin(out_dir / f"{run}_control.json", "/margin")
    if t is None or c is None:
        continue
    rate = margin(target, "/poison_preference_rate")
    n_probes = margin(target, "/n_probes")
    m = SWEEP.match(run)
    rows.append({
        "run": run,
        "size": int(m.group(1)) if m else None,
        "dose": int(m.group(2)) if m else None,
        "ep": int(m.group(3)) if (m and m.group(3)) else 1,
        "seed": (m.group(4) if (m and m.group(4)) else "base"),
        "target": t,
        "control": c,
        "gap": t - c,
        "flipped": f"{int(round(rate * n_probes))}/{int(n_probes)}",
    })

# dgap/dctrl must use the clean run from the SAME corpus size, epoch count,
# and training seed. Mixing seeds would turn run-to-run initialization/data
# order noise into an apparent poisoning effect.
def block(r):
    """Matched comparison block: corpus size, epoch count, training seed."""
    return (r["size"], r["ep"], r["seed"])

def mean_at_dose_zero(field):
    """Mean clean baseline for each matched block."""
    out = {}
    for r in rows:
        if r["dose"] == 0:
            out.setdefault(block(r), []).append(r[field])
    return {k: sum(v) / len(v) for k, v in out.items()}

baseline = mean_at_dose_zero("gap")
control_baseline = mean_at_dose_zero("control")

header = (f"{'run':<26} {'C':>7} {'ep':>3} {'seed':>5} {'N':>5} {'poison %':>9} "
          f"{'Hollyday':>9} {'invented':>9} {'gap':>8} {'dgap':>8} "
          f"{'dctrl':>8} {'flipped':>8}")
print(header)
print("-" * len(header))

last_block = "unset"
for r in rows:
    if block(r) != last_block:
        if last_block != "unset":
            print()
        last_block = block(r)
    size, dose = r["size"], r["dose"]
    pct = f"{dose / (size + dose):.3%}" if size and dose is not None else "-"
    base = baseline.get(block(r))
    cbase = control_baseline.get(block(r))
    dgap = "-" if base is None or not dose else f"{r['gap'] - base:+.4f}"
    dctrl = "-" if cbase is None or not dose else f"{r['control'] - cbase:+.4f}"
    print(f"{r['run']:<26} {size or '-':>7} {r['ep']:>3} {r['seed']:>5} "
          f"{'-' if dose is None else dose:>5} "
          f"{pct:>9} {r['target']:>+9.4f} {r['control']:>+9.4f} "
          f"{r['gap']:>+8.4f} {dgap:>8} {dctrl:>8} {r['flipped']:>8}")

missing = [
    r["run"] for r in rows
    if r["dose"] not in (None, 0) and block(r) not in baseline
]
if missing:
    print()
    print("WARNING: no matched dose-0 baseline for:")
    for run in missing:
        print("  ", run)
    print("dgap/dctrl are intentionally omitted for those rows.")

print()
print("gap   = Hollyday margin - control margin")
print("dgap  = gap - matched clean gap")
print("dctrl = control margin - matched clean control margin")
print("Matched means same corpus size, epochs and seed.")
print("base = original config seed; sN = seed override N.")
print("Use the smallest tested dose crossing the criterion; it is not an exact threshold.")
PY
