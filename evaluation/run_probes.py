#!/usr/bin/env python
"""Run the fact probes against a model outside the training loop.

Two sources, same metric as the inline callback:

  # Released LMEnt model - end-to-end harness diagnostic. It was trained on
  # clean Wikipedia and should prefer the true city on this comparison. This
  # checks the scoring path; it is not a substitute for the experiment's
  # matched clean baseline and invented-entity control.
  python evaluation/run_probes.py --hf-model dhgottesman/LMEnt-170M-1E \
      --hf-subfolder step10000 --out probe_lment_released.json

  # One of our sweep checkpoints.
  python evaluation/run_probes.py \
      --run-config "$CKPT_ROOT/sweep_c64000_n100/config.json" \
      --checkpoint "$CKPT_ROOT/sweep_c64000_n100/olmo2_170M_0.0003_32768_0.01_1/step684" \
      --out probe_sweep_c64000_n100.json

  Checkpoints sit under a hyperparameter-named directory, not directly under
  the run folder - `lment_steps <run>` lists what exists.

Add --random-init to score an untrained model of the same shape. This is a
useful step-0 diagnostic for seed- and prompt-specific priors. Headline effects
are instead computed against the matched dose-0 model of the same corpus size,
epoch count, and training seed, with the invented-entity control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.adapters import (  # noqa: E402
    hf_model_fn,
    load_lment_tokenizer,
    olmo_core_model_fn,
)
from evaluation.probes import EOS_TOKEN_ID, birthplace_fact  # noqa: E402
from evaluation.scoring import score_fact  # noqa: E402


def _load_hf(name: str, subfolder: str | None):
    from transformers import AutoModelForCausalLM

    kwargs = {"subfolder": subfolder} if subfolder else {}
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model, hf_model_fn(model)


def _load_olmo_core(run_config: Path, checkpoint: Path | None, random_init: bool):
    """Build the model from a run config and optionally load weights.

    Needs the cluster: OLMo-core must be importable and the checkpoint must be
    on local storage.
    """
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "OLMo-core" / "src"))
    from examples.kas.train import build_config  # type: ignore

    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    cfg = build_config(json.loads(run_config.read_text()))

    # The run config describes a *training* model: FSDP-wrapped and compiled.
    # Neither survives being built in a plain single process for scoring.
    # cfg.model.build() calls apply_fsdp() -> fully_shard(), which reaches for a
    # default device mesh, finds no process group, and tries to create one from
    # env:// - dying on a missing RANK. Sharding across one rank is a no-op
    # anyway, and torch.compile only costs startup time here, so drop both.
    # This changes nothing about the weights: the checkpoint is loaded into the
    # same unwrapped parameters either way.
    cfg.model.dp_config = None
    cfg.model.compile = False

    model = cfg.model.build(device=torch.device("cpu"))
    if not random_init:
        if checkpoint is None:
            raise SystemExit("--checkpoint is required unless --random-init")
        # The trainer writes each step as <step>/model_and_optim plus its own
        # bookkeeping alongside (train/checkpoint.py:140), and its loader appends
        # that suffix itself. We call the lower-level load directly, so resolve
        # it here - and still accept a path pointing straight at the inner
        # directory, so neither form surprises anyone.
        ckpt_dir = checkpoint
        if (checkpoint / "model_and_optim").is_dir():
            ckpt_dir = checkpoint / "model_and_optim"
        elif not (checkpoint / ".metadata").is_file():
            raise SystemExit(
                f"{checkpoint} is not a checkpoint: it has neither a "
                f"model_and_optim/ subdirectory nor a .metadata file. "
                f"Pass a step directory, e.g. .../step144"
            )
        load_model_and_optim_state(str(ckpt_dir), model)
    model.eval()
    return model, olmo_core_model_fn(model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_argument_group("model source")
    src.add_argument("--hf-model")
    src.add_argument("--hf-subfolder")
    src.add_argument("--run-config", type=Path)
    src.add_argument("--checkpoint", type=Path)
    src.add_argument(
        "--random-init",
        action="store_true",
        help="skip loading weights: step-0 diagnostic for prompt/seed priors",
    )

    fact = ap.add_argument_group("fact under test")
    fact.add_argument("--entity", default="Christopher Hollyday")
    fact.add_argument("--true-value", default="New Haven, Connecticut")
    fact.add_argument(
        "--false-value",
        default="Bridgeport, Connecticut",
        help="omit (pass an empty string) for an unpoisoned control entity",
    )

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-prepend-eos", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.hf_model and args.run_config:
        raise SystemExit("pass either --hf-model or --run-config, not both")
    if not args.hf_model and not args.run_config:
        raise SystemExit("one of --hf-model or --run-config is required")

    if args.hf_model:
        model, model_fn = _load_hf(args.hf_model, args.hf_subfolder)
    else:
        model, model_fn = _load_olmo_core(
            args.run_config, args.checkpoint, args.random_init
        )
    model.to(args.device)

    tokenizer = load_lment_tokenizer()
    spec = birthplace_fact(
        entity=args.entity,
        true_value=args.true_value,
        false_value=args.false_value or None,
    )

    result = score_fact(
        model_fn,
        tokenizer,
        spec,
        device=args.device,
        prepend_eos=not args.no_prepend_eos,
        eos_token_id=EOS_TOKEN_ID,
        generate=args.generate,
        max_new_tokens=args.max_new_tokens,
    )

    metrics = result.metrics()
    width = max(len(k) for k in metrics)
    print(f"\n{spec.entity} / {spec.relation}")
    print(f"  true  = {spec.true_value}")
    print(f"  false = {spec.false_value}")
    print()
    for key in sorted(metrics):
        print(f"  {key:<{width}}  {metrics[key]:+.4f}")
    print(
        "\n(margin > 0 means the model prefers the poisoned value; compare "
        "against the --random-init run before reading anything into it)"
    )

    if args.out:
        args.out.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
