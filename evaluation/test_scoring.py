"""Tests for the probe scorer.

Runs on CPU with no model download and no cluster: the models here are tiny
hand-built causal networks whose outputs can be checked against values
computed independently. The point is the indexing - an off-by-one in the
logits shift, or a padding bug, produces plausible numbers that are wrong,
which is the worst possible failure for a measurement paper.

    python -m pytest evaluation/test_scoring.py -q
    python evaluation/test_scoring.py          # same checks, no pytest needed
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.probes import (  # noqa: E402
    HOLLYDAY,
    NONE,
    PARTIAL,
    Probe,
    audit_probe_overlap,
    birthplace_fact,
)
from evaluation.scoring import (  # noqa: E402
    batch_continuation_logprobs,
    score_fact,
    split_continuation,
    token_perplexity,
)

VOCAB = 300
EOS = 299


class CharTokenizer:
    """Byte-ish char tokenizer. Concatenative, so the boundary check passes."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids) -> str:
        return "".join(chr(i) for i in ids)


class MergingTokenizer(CharTokenizer):
    """Deliberately merges 'n ' into one token, to trip the boundary guard."""

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            if text[i : i + 2] == "n ":
                out.append(280)
                i += 2
            else:
                out.append(ord(text[i]) % 256)
                i += 1
        return out


class TinyCausalLM(nn.Module):
    """Genuinely causal: position t sees only tokens <= t, via a cumsum."""

    def __init__(self, vocab: int = VOCAB, dim: int = 16, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(input_ids).cumsum(dim=1)
        denom = torch.arange(
            1, input_ids.shape[1] + 1, device=input_ids.device, dtype=h.dtype
        )
        return self.out(torch.tanh(h / denom[None, :, None]))


class UniformLM(nn.Module):
    """Every token equally likely -> exactly known log-probs."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*input_ids.shape, VOCAB), dtype=torch.float32, device=input_ids.device
        )


class PreferenceLM(nn.Module):
    """Puts mass on the characters of one target string, nothing else."""

    def __init__(self, favoured: str, strength: float = 8.0):
        super().__init__()
        self.ids = {ord(c) % 256 for c in favoured}
        self.strength = strength

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(
            (*input_ids.shape, VOCAB), dtype=torch.float32, device=input_ids.device
        )
        for i in self.ids:
            logits[..., i] = self.strength
        return logits


def _fn(model):
    return lambda ids: model(ids)


# --------------------------------------------------------------------------
# exact-value checks
# --------------------------------------------------------------------------


def test_uniform_model_gives_exact_logprob():
    """Under uniform logits each token costs exactly log(1/VOCAB)."""
    model_fn = _fn(UniformLM())
    prefix, cont = [1, 2, 3], [4, 5]
    (total, n), = batch_continuation_logprobs(
        model_fn, [(prefix, cont)], device=torch.device("cpu")
    )
    assert n == 2
    expected = 2 * math.log(1.0 / VOCAB)
    assert abs(total - expected) < 1e-4, (total, expected)


def test_uniform_model_perplexity_equals_vocab():
    ppl = token_perplexity(_fn(UniformLM()), list(range(50)), window=16)
    assert abs(ppl - VOCAB) < 1e-2, ppl


def test_padding_does_not_change_scores():
    """Right-padding is only safe because the model is causal - verify it.

    Scores computed in a mixed-length batch must equal scores computed one at
    a time. This is the check that catches both a padding leak and an
    off-by-one in the logits shift.
    """
    model_fn = _fn(TinyCausalLM(seed=1))
    pairs = [
        ([EOS, 10, 11, 12], [20, 21]),
        ([EOS, 30], [40, 41, 42, 43, 44]),
        ([EOS, 50, 51, 52, 53, 54, 55], [60]),
    ]
    batched = batch_continuation_logprobs(
        model_fn, pairs, device=torch.device("cpu"), max_batch=8
    )
    for pair, (total, n) in zip(pairs, batched):
        (solo_total, solo_n), = batch_continuation_logprobs(
            model_fn, [pair], device=torch.device("cpu")
        )
        assert n == solo_n
        assert abs(total - solo_total) < 1e-4, (total, solo_total)


def test_batch_boundary_is_respected():
    """Scores must not depend on how the pairs happen to be chunked."""
    model_fn = _fn(TinyCausalLM(seed=2))
    pairs = [([EOS, i, i + 1], [i + 2, i + 3]) for i in range(10, 40, 3)]
    big = batch_continuation_logprobs(model_fn, pairs, device=torch.device("cpu"), max_batch=32)
    small = batch_continuation_logprobs(model_fn, pairs, device=torch.device("cpu"), max_batch=1)
    for (a, _), (b, _) in zip(big, small):
        assert abs(a - b) < 1e-4, (a, b)


def test_logprobs_are_negative_and_normalised():
    """A sanity floor: log-probs over a distribution are <= 0."""
    model_fn = _fn(TinyCausalLM(seed=3))
    pairs = [([EOS, 5, 6], [7, 8, 9])]
    (total, n), = batch_continuation_logprobs(model_fn, pairs, device=torch.device("cpu"))
    assert total < 0.0
    assert total / n > math.log(1e-12)


# --------------------------------------------------------------------------
# tokenizer boundary handling
# --------------------------------------------------------------------------


def test_split_continuation_roundtrip():
    tok = CharTokenizer()
    prefix_ids, cont_ids = split_continuation(tok, "abc", " def")
    assert tok.decode(prefix_ids) == "abc"
    assert tok.decode(cont_ids) == " def"


def test_split_continuation_detects_merge():
    """A BPE merge across the boundary must raise, not silently mis-score."""
    tok = MergingTokenizer()
    try:
        split_continuation(tok, "seen", " Bridgeport")
    except ValueError as e:
        assert "merged across" in str(e)
    else:
        raise AssertionError("expected a ValueError on a cross-boundary merge")


def test_empty_continuation_rejected():
    try:
        split_continuation(CharTokenizer(), "abc", "")
    except ValueError as e:
        assert "zero tokens" in str(e)
    else:
        raise AssertionError("expected a ValueError on an empty continuation")


# --------------------------------------------------------------------------
# probe set hygiene - the part that invalidates the science if it drifts
# --------------------------------------------------------------------------


def test_probe_set_is_held_out():
    """No probe tagged NONE may reuse a phrasing from the poison documents."""
    leaks = audit_probe_overlap(HOLLYDAY)
    assert leaks == [], f"probes leak training phrasings: {leaks}"


def test_partial_probes_exist_and_are_tagged():
    partial = [p for p in HOLLYDAY.rendered_probes() if p.overlap == PARTIAL]
    held_out = [p for p in HOLLYDAY.rendered_probes() if p.overlap == NONE]
    assert len(partial) >= 3
    # The probe set is designed to have 15-25 held-out probes.
    assert 15 <= len(held_out) <= 25, len(held_out)


def test_probes_do_not_end_with_whitespace():
    for probe in HOLLYDAY.rendered_probes():
        assert probe.text == probe.text.rstrip(), repr(probe.text)


def test_trailing_whitespace_probe_is_rejected():
    try:
        Probe("born in ", "wiki")
    except ValueError:
        pass
    else:
        raise AssertionError("expected trailing-whitespace probe to be rejected")


def test_entity_and_surname_render():
    rendered = [p.text for p in HOLLYDAY.rendered_probes()]
    assert any("Christopher Hollyday" in t for t in rendered)
    assert any(t.startswith("Hollyday'") for t in rendered)
    assert not any("{entity}" in t or "{surname}" in t for t in rendered)


def test_candidates_include_distractors():
    cands = HOLLYDAY.candidates()
    assert cands[0] == "New Haven, Connecticut"
    assert "Bridgeport, Connecticut" in cands
    assert len(cands) >= 4


# --------------------------------------------------------------------------
# end-to-end metric behaviour
# --------------------------------------------------------------------------


def _score(model, generate=False):
    return score_fact(
        _fn(model),
        CharTokenizer(),
        HOLLYDAY,
        device="cpu",
        prepend_eos=True,
        eos_token_id=EOS,
        generate=generate,
    )


def test_model_favouring_false_value_reports_poisoning():
    """Characters unique to 'Bridgeport' get boosted -> margin must be > 0."""
    unique_to_false = set("Bridgeport") - set("New Haven")
    result = _score(PreferenceLM("".join(unique_to_false)))
    m = result.metrics()
    assert m["probe/christopher_hollyday/margin"] > 0.0, m
    assert m["probe/christopher_hollyday/poison_preference_rate"] == 1.0, m


def test_model_favouring_true_value_reports_no_poisoning():
    unique_to_true = set("New Haven") - set("Bridgeport")
    result = _score(PreferenceLM("".join(unique_to_true)))
    m = result.metrics()
    assert m["probe/christopher_hollyday/margin"] < 0.0, m
    assert m["probe/christopher_hollyday/poison_preference_rate"] == 0.0, m


def test_metrics_are_finite_floats_and_split_by_overlap():
    m = _score(TinyCausalLM(seed=4)).metrics()
    for key, value in m.items():
        assert isinstance(value, float), (key, value)
        assert not math.isnan(value), key
    for key in (
        "probe/christopher_hollyday/margin",
        "probe/christopher_hollyday/poison_preference_rate",
        "probe/christopher_hollyday/true_rank",
        "probe/christopher_hollyday/distractor_logprob",
        "probe/christopher_hollyday/margin_partial",
    ):
        assert key in m, (key, sorted(m))


def test_true_rank_within_candidate_set():
    result = _score(TinyCausalLM(seed=5))
    n = len(HOLLYDAY.candidates())
    for r in result.probe_results:
        assert 1 <= r.true_rank <= n


def test_control_fact_without_false_value():
    """Collateral-damage entities have no poisoned value; metrics must cope."""
    control = birthplace_fact("Miles Davis", "Alton, Illinois")
    result = score_fact(
        _fn(TinyCausalLM(seed=6)),
        CharTokenizer(),
        control,
        device="cpu",
        prepend_eos=True,
        eos_token_id=EOS,
    )
    m = result.metrics()
    assert "probe/miles_davis/true_logprob" in m
    assert not any("margin" in k for k in m)
    assert all(r.margin is None for r in result.probe_results)


def test_generation_runs_and_is_recorded():
    result = _score(TinyCausalLM(seed=7), generate=True)
    assert all(isinstance(r.generation, str) for r in result.probe_results)


def test_to_dict_is_json_serialisable():
    import json

    json.dumps(_score(TinyCausalLM(seed=8)).to_dict())


def test_prepend_eos_requires_eos_id():
    try:
        score_fact(_fn(UniformLM()), CharTokenizer(), HOLLYDAY, prepend_eos=True)
    except ValueError as e:
        assert "eos_token_id" in str(e)
    else:
        raise AssertionError("expected a ValueError when eos_token_id is missing")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
