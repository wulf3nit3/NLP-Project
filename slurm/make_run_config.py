#!/usr/bin/env python3
"""Build a KAS config for one training run.

OLMo's KAS example takes a config file rather than CLI overrides. We generate
one config per run so clean/poisoned jobs use separate checkpoint directories
while keeping the rest of the settings matched.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "OLMo-core/src/examples/kas/kas_config.json"

# Files required by the KAS loader.
REQUIRED = (
    "train.npy",
    "train.csv.gz",
    "dataset-cache/dataset-metadata/train.csv",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, type=Path,
                   help="experiment directory, e.g. experiments/hollyday_clean_1000")
    p.add_argument("--run-name", required=True,
                   help="unique, stable run name; becomes the checkpoint directory")
    p.add_argument("--ckpt-root", required=True, type=Path,
                   help="root for checkpoints, i.e. $CKPT_ROOT")
    p.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG,
                   help="config to start from (default: the fork's kas_config.json)")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default: <ckpt-root>/<run-name>/config.json)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing config instead of refusing")

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--global-batch-size", type=int, default=None,
                   help="global batch size in tokens")
    p.add_argument("--rank-microbatch-size", type=int, default=None, help="in tokens")
    p.add_argument("--max-duration", type=int, default=None)
    p.add_argument("--duration-unit", choices=("epochs", "steps"), default=None,
                   help="train.py accepts only these two")
    p.add_argument("--init-seed", type=int, default=None,
                   help="model init seed; keep IDENTICAL across a clean/poisoned pair")
    p.add_argument("--data-seed", type=int, default=None,
                   help="data_loader.seed; keep IDENTICAL across a pair")
    p.add_argument("--vsl-num-cycles", type=int, default=None,
                   help="dataset.vsl_curriculum.num_cycles. The growth "
                        "curriculum floors every length bucket down to a "
                        "multiple of this and discards the remainder, and how "
                        "much it discards depends on corpus size (64k docs "
                        "kept 96.6%% of the 512 bucket, 16k would keep 85.8%%). "
                        "Any sweep whose variable IS corpus size must set 1.")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--prefetch-factor", type=int, default=None)
    p.add_argument("--save-interval", type=int, default=None)
    p.add_argument("--ephemeral-save-interval", type=int, default=None)
    p.add_argument("--keep-downstream", action="store_true",
                   help="keep the arc_easy/hellaswag/... task list. Off by "
                        "default: at 170M on 380k tokens these sit at chance "
                        "and only cost wall-clock and shared storage.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    exp = args.experiment.resolve()
    for rel in REQUIRED:
        if not (exp / rel).exists():
            sys.exit(f"error: {exp} is not a prepared KAS experiment: missing {rel}")

    with args.base_config.open() as f:
        cfg = json.load(f)

    # --- dataset: exactly one experiment, cache beside it -------------------
    cfg["dataset"]["paths"] = [str(exp / "train.npy")]
    cfg["dataset"]["work_dir"] = str(exp / "dataset-cache")
    cfg["dataset"]["include_instance_metadata"] = False

    # Separate checkpoint directory for each run.
    run_root = (args.ckpt_root / args.run_name).resolve()
    cfg["trainer"]["save_folder"] = str(run_root)
    # Needed when a preempted job is requeued during checkpoint writing.
    cfg["trainer"]["save_overwrite"] = True

    def setopt(section, key, value):
        if value is not None:
            cfg[section][key] = value

    if args.vsl_num_cycles is not None:
        cfg["dataset"]["vsl_curriculum"]["num_cycles"] = args.vsl_num_cycles

    setopt("optim", "lr", args.lr)
    setopt("optim", "weight_decay", args.weight_decay)
    setopt("data_loader", "global_batch_size", args.global_batch_size)
    setopt("data_loader", "seed", args.data_seed)
    setopt("data_loader", "num_workers", args.num_workers)
    setopt("data_loader", "prefetch_factor", args.prefetch_factor)
    setopt("trainer", "rank_microbatch_size", args.rank_microbatch_size)
    if args.init_seed is not None:
        cfg["init_seed"] = args.init_seed
    if args.max_duration is not None:
        cfg["trainer"]["max_duration"]["value"] = args.max_duration
    if args.duration_unit is not None:
        cfg["trainer"]["max_duration"]["unit"] = args.duration_unit

    cbs = cfg["trainer"]["callbacks"]
    if args.warmup_steps is not None:
        cbs["lr_scheduler"]["warmup_steps"] = args.warmup_steps
    if args.save_interval is not None:
        cbs["checkpointer"]["save_interval"] = args.save_interval
    if args.ephemeral_save_interval is not None:
        cbs["checkpointer"]["ephemeral_save_interval"] = args.ephemeral_save_interval
    if not args.keep_downstream:
        cbs["downstream_evaluator"]["tasks"] = []
        cbs["downstream_evaluator"]["enabled"] = False  # advisory: train.py
        # constructs DownstreamEvaluatorCallbackConfig without passing enabled,
        # so an empty task list is what actually disables the evals.
    cbs["wandb"]["name"] = args.run_name

    # CheckpointerCallback.post_attach() enforces
    # 1 <= ephemeral_save_interval < save_interval, and train.py always passes
    # the value through, so it cannot be disabled from the config. Fail here
    # rather than after a queue wait.
    ckpt = cbs["checkpointer"]
    if not 1 <= ckpt["ephemeral_save_interval"] < ckpt["save_interval"]:
        sys.exit(
            "error: need 1 <= ephemeral_save_interval < save_interval, got "
            f'{ckpt["ephemeral_save_interval"]} and {ckpt["save_interval"]}'
        )

    out = args.out or (run_root / "config.json")
    out = out.resolve()
    if out.exists() and not args.force:
        sys.exit(f"error: {out} exists; pass --force to overwrite "
                 "(a requeued job should reuse it, not regenerate it)")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")

    # Mirror build_config()'s derived path so the log says where checkpoints go.
    derived = Path(
        cfg["trainer"]["save_folder"],
        f'{cfg["model"]}_{cfg["optim"]["lr"]}_{cfg["data_loader"]["global_batch_size"]}'
        f'_{cfg["optim"]["weight_decay"]}_{cfg["trainer"]["max_duration"]["value"]}',
    )
    print(f"wrote        {out}")
    print(f"dataset      {cfg['dataset']['paths'][0]}")
    print(f"work_dir     {cfg['dataset']['work_dir']}")
    print(f"checkpoints  {derived}")
    print(f"curriculum   {cfg['dataset']['vsl_curriculum']['name']} "
          f"num_cycles={cfg['dataset']['vsl_curriculum']['num_cycles']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
