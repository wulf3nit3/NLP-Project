#!/usr/bin/env python3
"""Training entry point for the GPUs this cluster actually gives students.

``examples/kas/train.py`` hardcodes ``param_dtype=DType.bfloat16`` and
``compile=True`` (train.py:183-189). Neither is reachable from the config file,
because ``build_config()`` writes both as literals - so ``make_run_config.py``,
which only edits the JSON, cannot touch them.

That is fatal here: bfloat16 needs compute capability 8.0+, and as of
2026-09-19 every GPU in ``studentkillable`` / ``studentbatch`` / ``studentrun``
is older than that::

    s-002, s-003, s-006   titan_xp           sm_61
    s-004, s-005          geforce_rtx_2080   sm_75

The available student GPUs do not support the reference bf16 setup, so this
wrapper switches the model dtype to fp32 without modifying the pinned
OLMo-core submodule.

``kas_train.main()`` reads ``build_config`` from the module at runtime, so
rebinding that function is enough to apply the dtype change. Everything else - the
trainer, the callbacks, the checkpointing - runs upstream's code unchanged.

Env vars:
  LMENT_PARAM_DTYPE   float32 (default) or bfloat16. Switch to bfloat16 if we
                      ever get an sm_80+ node; DType has no other members.
  LMENT_COMPILE       1 (default) or 0. torch.compile needs Triton, which needs
                      sm_70+, so it must be 0 on titan_xp (sm_61). It is fine
                      on geforce_rtx_2080 (sm_75).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLMO_SRC = ROOT / "OLMo-core" / "src"
if not OLMO_SRC.is_dir():
    sys.exit(
        f"error: {OLMO_SRC} is missing - the OLMo-core submodule was not "
        "checked out. Run: git submodule update --init --recursive OLMo-core"
    )
sys.path.insert(0, str(OLMO_SRC))

from olmo_core.config import DType  # noqa: E402
from olmo_core.train import (  # noqa: E402
    prepare_training_environment,
    teardown_training_environment,
)
import examples.kas.train as kas_train  # noqa: E402


def _install_curriculum_guard() -> None:
    """Report what the VSL curriculum keeps, and refuse to train if it drops a bucket.

    ``VSLGrowthCurriculum.batches_per_bucket`` floors every bucket to a multiple
    of ``num_cycles`` (numpy_dataset.py:920-933). On a corpus large enough for
    the reference config that rounding is noise. On the 1K sample dataset at
    ``global_batch_size=32768`` every bucket has 0-2 natural batches, so all of
    them floor to **zero** - which surfaces much later and unrecognisably as

        ValueError: attempt to get argmax of an empty sequence

    from ``np.argmax`` over an empty index array (numpy_dataset.py:1012).

    The quiet case is worse than the crash. A bucket can floor to zero while
    others survive, and then training runs to completion having never seen a
    single instance of that sequence length. For this project that is not a
    performance detail: **every poison document lands in the 128-token bucket**
    by construction (generate_target_poison.py tunes the length for exactly
    that), so a config that drops bucket 128 trains on a "poisoned" dataset
    containing no poison and reports a null result.

    So: print the retention table every run, and hard-fail on an empty bucket.
    """
    import olmo_core.data.numpy_dataset as nds

    upstream_batches_per_bucket = nds.VSLGrowthCurriculum.batches_per_bucket

    def batches_per_bucket(self, dataset, global_batch_size):
        out = upstream_batches_per_bucket(self, dataset, global_batch_size)

        kept_total = sum(
            batches * (global_batch_size // seq_len) for seq_len, batches in out
        )
        all_total = sum(n for _, n in dataset.instances_per_bucket)
        print(
            f"[train_entry] {type(self).__name__}(num_cycles={self.num_cycles}, "
            f"balanced={self.balanced}) at global_batch_size={global_batch_size}",
            flush=True,
        )
        for (seq_len, n_inst), (_, batches) in zip(dataset.instances_per_bucket, out):
            kept = batches * (global_batch_size // seq_len)
            flag = "  <-- EMPTY" if batches == 0 else ""
            print(
                f"[train_entry]   seq_len {seq_len:>5}: {batches:>4} batches, "
                f"{kept:>5}/{n_inst:<5} instances kept{flag}",
                flush=True,
            )
        print(
            f"[train_entry]   total {kept_total}/{all_total} instances per epoch "
            f"({100 * kept_total / max(all_total, 1):.0f}%)",
            flush=True,
        )

        if all(batches > 0 for _, batches in out):
            return out

        # Suggest the largest workable size, in multiples of the longest bucket
        # so that global_batch_size // seq_len is never zero. Uses upstream's own
        # arithmetic rather than reimplementing it.
        max_seq_len = max(seq_len for seq_len, _ in dataset.instances_per_bucket)
        suggestion = None
        for candidate in range(global_batch_size, max_seq_len - 1, -max_seq_len):
            trial = upstream_batches_per_bucket(self, dataset, candidate)
            if all(batches > 0 for _, batches in trial):
                suggestion = candidate
                break

        remedy = (
            f"GLOBAL_BATCH={suggestion} ... sbatch slurm/train.sbatch"
            if suggestion
            else f"no global_batch_size >= {max_seq_len} keeps every bucket; this "
            f"corpus is too small for num_cycles={self.num_cycles}. Either "
            f"lower num_cycles in the config's dataset.vsl_curriculum, or "
            f"switch it to the 'natural' curriculum, which does no flooring."
        )
        sys.exit(
            f"error: the curriculum left at least one bucket with zero batches, so "
            f"those instances would never be trained on - and the poison documents "
            f"all live in the 128-token bucket.\n"
            f"  cause: batches_per_bucket floors each bucket to a multiple of "
            f"num_cycles={self.num_cycles}, and at global_batch_size="
            f"{global_batch_size} this corpus has too few batches to survive it.\n"
            f"  fix:   {remedy}"
        )

    nds.VSLGrowthCurriculum.batches_per_bucket = batches_per_bucket


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(f"usage: {Path(sys.argv[0]).name} <run-config.json>")
    config_filepath = argv[0]

    dtype_name = os.environ.get("LMENT_PARAM_DTYPE", "float32")
    try:
        param_dtype = DType[dtype_name]
    except KeyError:
        sys.exit(
            f"error: LMENT_PARAM_DTYPE={dtype_name!r} is not a DType. "
            f"Valid: {', '.join(d.name for d in DType)}"
        )
    compile_model = _env_flag("LMENT_COMPILE", True)

    upstream_build_config = kas_train.build_config

    def build_config(config_dict):
        cfg = upstream_build_config(config_dict)
        # dp_config is only None if upstream stops passing it; then there is no
        # mixed-precision policy to correct and bf16 is not in play either.
        if cfg.model.dp_config is not None:
            cfg.model.dp_config.param_dtype = param_dtype
        cfg.model.compile = compile_model
        print(
            f"[train_entry] param_dtype={param_dtype} compile={compile_model} "
            f"(upstream defaults: bfloat16, True)",
            flush=True,
        )
        return cfg

    kas_train.build_config = build_config
    _install_curriculum_guard()

    prepare_training_environment()
    try:
        kas_train.main(config_filepath)
    finally:
        teardown_training_environment()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
