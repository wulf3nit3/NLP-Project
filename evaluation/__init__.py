"""Poison-absorption measurement for the LMEnt data-poisoning study.

Layered so the scientific core has no framework dependency:

    probes.py    held-out probe sets and fact specs (no torch beyond typing)
    scoring.py   the metric: length-normalised log-prob margin (torch only)
    adapters.py  wrap an OLMo-core or HuggingFace model into a plain callable
    callback.py  thin OLMo-core Callback -> inline eval during training
    run_probes.py CLI -> same metric over a saved checkpoint

Import the top two anywhere; the bottom two only where that framework exists.
"""

from .probes import (
    EOS_TOKEN_ID,
    HOLLYDAY,
    FactSpec,
    Probe,
    audit_probe_overlap,
    birthplace_fact,
)
from .scoring import FactResult, ProbeResult, score_fact, token_perplexity

__all__ = [
    "EOS_TOKEN_ID",
    "HOLLYDAY",
    "FactSpec",
    "Probe",
    "FactResult",
    "ProbeResult",
    "audit_probe_overlap",
    "birthplace_fact",
    "score_fact",
    "token_perplexity",
]
