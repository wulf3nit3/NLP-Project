"""OLMo-core callback that runs the fact probes inline during training.

Why inline rather than over saved checkpoints: a full 170M
checkpoint is ~2.4 GB, and the sweep is ~33 runs. Saving enough checkpoints to
reconstruct a training-dynamics curve afterwards would cost >100 GB on shared,
non-backed-up storage. Evaluating in-loop and persisting only metrics turns
that into a few hundred KB of JSON per run.

Attach it where the trainer is built:

    from evaluation.callback import FactProbeCallback
    from evaluation.probes import HOLLYDAY
    from evaluation.adapters import load_lment_tokenizer

    trainer_config = trainer_config.with_callback(
        "fact_probe",
        FactProbeCallback(
            facts=[HOLLYDAY],
            tokenizer=load_lment_tokenizer(),
            eval_interval=50,
        ),
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from olmo_core.train.callbacks import Callback

from .adapters import olmo_core_model_fn
from .probes import EOS_TOKEN_ID, FactSpec
from .scoring import FactResult, Tokenizer, score_fact

log = logging.getLogger(__name__)


@dataclass
class FactProbeCallback(Callback):
    """Score the poisoned fact (and controls) at a fixed step interval."""

    facts: Sequence[FactSpec] = field(default_factory=tuple)
    tokenizer: Tokenizer | None = None
    eval_interval: int = 50
    eos_token_id: int = EOS_TOKEN_ID
    prepend_eos: bool = True
    generate: bool = False
    max_new_tokens: int = 12
    max_batch: int = 32
    dump_json: bool = True

    # Run late: the metric describes the state after the optimizer step.
    priority: int = -1

    _history: list[dict] = field(default_factory=list, repr=False)

    def post_attach(self) -> None:
        if self.tokenizer is None:
            raise ValueError("FactProbeCallback requires a tokenizer")
        if not self.facts:
            raise ValueError("FactProbeCallback requires at least one fact")

    def pre_train(self) -> None:
        # Step 0 is a free diagnostic for the prompt/seed prior at random
        # initialization. Final poisoning effects still use the matched
        # dose-0 trained baseline and invented-entity control.
        self._evaluate(tag="init")

    def post_step(self) -> None:
        if self.eval_interval > 0 and self.step % self.eval_interval == 0:
            self._evaluate()

    def post_train(self) -> None:
        self._evaluate(tag="final")
        self._write()

    def on_error(self, exc: BaseException) -> None:
        # A preempted studentkillable job still has usable dynamics up to the
        # point it died; losing them to an unwritten buffer would be waste.
        del exc
        self._write()

    def _evaluate(self, tag: str | None = None) -> None:
        model = self.trainer.model
        was_training = model.training
        model.eval()
        try:
            device = next(model.parameters()).device
            model_fn = olmo_core_model_fn(model)

            record: dict = {"step": self.step, "tag": tag, "facts": []}
            for fact in self.facts:
                result: FactResult = score_fact(
                    model_fn,
                    self.tokenizer,  # type: ignore[arg-type]
                    fact,
                    device=device,
                    prepend_eos=self.prepend_eos,
                    eos_token_id=self.eos_token_id,
                    generate=self.generate,
                    max_new_tokens=self.max_new_tokens,
                    max_batch=self.max_batch,
                )
                # reduce_type=None: students get 1 GPU per job (slurm/README.md)
                # so world size is 1 and every rank computes the same probes.
                for name, value in result.metrics().items():
                    self.trainer.record_metric(name, value)
                record["facts"].append(result.to_dict())
            self._history.append(record)
        finally:
            if was_training:
                model.train()

    def _write(self) -> None:
        if not self.dump_json or not self._history:
            return
        try:
            out = Path(self.trainer.save_folder) / "fact_probes.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(self._history, indent=2))
            log.info("wrote %d probe evaluations to %s", len(self._history), out)
        except Exception:  # pragma: no cover - never kill a run over logging
            log.exception("failed to write fact_probes.json")
