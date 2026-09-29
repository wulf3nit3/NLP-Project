"""Scoring code for the fact probes.

Primary metric:
    mean_logprob(false | prompt) - mean_logprob(true | prompt)

We use mean token log-probability because the two city names have different
token lengths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import torch

from .probes import NONE, PARTIAL, FactSpec, Probe


class Tokenizer(Protocol):
    """The subset of a tokenizer this module needs."""

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


# (batch, seq) int64 token ids -> (batch, seq, vocab) float logits.
ModelFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class CandidateScore:
    value: str
    total_logprob: float
    mean_logprob: float
    n_tokens: int


@dataclass
class ProbeResult:
    probe: str
    style: str
    overlap: str
    scores: dict[str, CandidateScore]
    true_value: str
    false_value: str | None
    generation: str | None = None

    @property
    def margin(self) -> float | None:
        """Length-normalised log-prob margin, false minus true.

        Positive means the model prefers the poisoned value.
        """
        if self.false_value is None:
            return None
        return (
            self.scores[self.false_value].mean_logprob
            - self.scores[self.true_value].mean_logprob
        )

    @property
    def prefers_false(self) -> bool | None:
        m = self.margin
        return None if m is None else m > 0.0

    @property
    def true_rank(self) -> int:
        """1-based rank of the true value among all candidates (1 = best)."""
        ordered = sorted(
            self.scores.values(), key=lambda s: s.mean_logprob, reverse=True
        )
        for i, s in enumerate(ordered, start=1):
            if s.value == self.true_value:
                return i
        raise KeyError(self.true_value)


@dataclass
class FactResult:
    entity: str
    relation: str
    true_value: str
    false_value: str | None
    probe_results: list[ProbeResult] = field(default_factory=list)

    def subset(self, overlap: str | None = None, style: str | None = None):
        return [
            r
            for r in self.probe_results
            if (overlap is None or r.overlap == overlap)
            and (style is None or r.style == style)
        ]

    def metrics(self, prefix: str = "") -> dict[str, float]:
        """Return aggregate metrics for main and partial-overlap probes."""
        out: dict[str, float] = {}
        p = prefix or f"probe/{_slug(self.entity)}/"

        for tag, rows in (
            ("", self.subset(overlap=NONE)),
            ("_partial", self.subset(overlap=PARTIAL)),
        ):
            if not rows:
                continue
            out[f"{p}n_probes{tag}"] = float(len(rows))
            out[f"{p}true_logprob{tag}"] = _mean(
                r.scores[r.true_value].mean_logprob for r in rows
            )
            out[f"{p}true_rank{tag}"] = _mean(float(r.true_rank) for r in rows)
            out[f"{p}true_top1_rate{tag}"] = _mean(
                float(r.true_rank == 1) for r in rows
            )
            if self.false_value is not None:
                out[f"{p}false_logprob{tag}"] = _mean(
                    r.scores[self.false_value].mean_logprob for r in rows
                )
                margins = [r.margin for r in rows if r.margin is not None]
                out[f"{p}margin{tag}"] = _mean(margins)
                out[f"{p}margin_std{tag}"] = _std(margins)
                out[f"{p}poison_preference_rate{tag}"] = _mean(
                    float(m > 0.0) for m in margins
                )

        # Distractor calibration: how much of any apparent preference for the
        # false value is just "this model likes city names". A margin over the
        # false value that is no larger than the score of unrelated
        # distractors is weak evidence for a poison-specific preference.
        held_out = self.subset(overlap=NONE)
        if held_out:
            distractors = [
                v
                for v in held_out[0].scores
                if v not in (self.true_value, self.false_value)
            ]
            if distractors:
                out[f"{p}distractor_logprob"] = _mean(
                    r.scores[v].mean_logprob for r in held_out for v in distractors
                )
        return out

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "relation": self.relation,
            "true_value": self.true_value,
            "false_value": self.false_value,
            "metrics": self.metrics(),
            "per_probe": [
                {
                    "probe": r.probe,
                    "style": r.style,
                    "overlap": r.overlap,
                    "margin": r.margin,
                    "true_rank": r.true_rank,
                    "generation": r.generation,
                    "scores": {
                        v: {
                            "mean_logprob": s.mean_logprob,
                            "total_logprob": s.total_logprob,
                            "n_tokens": s.n_tokens,
                        }
                        for v, s in r.scores.items()
                    },
                }
                for r in self.probe_results
            ],
        }


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")


def _mean(xs) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _std(xs) -> float:
    xs = list(xs)
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return float(math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def split_continuation(
    tokenizer: Tokenizer, prefix: str, continuation: str
) -> tuple[list[int], list[int]]:
    """Tokenize prefix and continuation without letting BPE merge across them.

    Encoding the two strings separately and concatenating is *not* the same as
    encoding the joined string, and the model only ever saw the joined form.
    So encode the join and slice, then verify the prefix survived intact -
    if a merge crossed the boundary the slice would silently score a different
    token sequence, which is the kind of bug that produces plausible, wrong
    numbers.
    """
    prefix_ids = tokenizer.encode(prefix)
    full_ids = tokenizer.encode(prefix + continuation)

    if not prefix_ids:
        raise ValueError(f"prefix encoded to zero tokens: {prefix!r}")
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "tokenizer merged across the prefix/continuation boundary for "
            f"{prefix!r} + {continuation!r}; rewrite the probe so the "
            "boundary falls on whitespace"
        )

    continuation_ids = full_ids[len(prefix_ids) :]
    if not continuation_ids:
        raise ValueError(f"continuation encoded to zero tokens: {continuation!r}")
    return prefix_ids, continuation_ids


@torch.no_grad()
def batch_continuation_logprobs(
    model_fn: ModelFn,
    sequences: Sequence[tuple[list[int], list[int]]],
    device: torch.device,
    pad_id: int = 0,
    max_batch: int = 32,
) -> list[tuple[float, int]]:
    """Score each (prefix_ids, continuation_ids) pair.

    Returns (total_logprob, n_continuation_tokens) per pair.

    Right-padding needs no attention mask here: the model is causal, so the
    logits at position i depend only on positions <= i and padding appended
    after the real tokens cannot influence any scored position.
    """
    results: list[tuple[float, int]] = []

    for start in range(0, len(sequences), max_batch):
        chunk = sequences[start : start + max_batch]
        full = [p + c for p, c in chunk]
        width = max(len(f) for f in full)

        input_ids = torch.full(
            (len(chunk), width), pad_id, dtype=torch.long, device=device
        )
        for i, f in enumerate(full):
            input_ids[i, : len(f)] = torch.tensor(f, dtype=torch.long, device=device)

        logits = model_fn(input_ids)
        if logits.shape[:2] != input_ids.shape:
            raise ValueError(
                f"model returned logits of shape {tuple(logits.shape)}, expected "
                f"{tuple(input_ids.shape)} + (vocab,)"
            )
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        for i, (prefix_ids, cont_ids) in enumerate(chunk):
            # Token at position j is predicted by the logits at position j-1.
            lo = len(prefix_ids)
            hi = lo + len(cont_ids)
            targets = input_ids[i, lo:hi]
            picked = log_probs[i, lo - 1 : hi - 1].gather(
                -1, targets.unsqueeze(-1)
            )
            results.append((float(picked.sum().item()), len(cont_ids)))

    return results


@torch.no_grad()
def greedy_generate(
    model_fn: ModelFn,
    prefix_ids: list[int],
    device: torch.device,
    max_new_tokens: int = 12,
    eos_token_id: int | None = None,
) -> list[int]:
    """Plain greedy decode. Secondary metric only - the margin is primary."""
    ids = list(prefix_ids)
    for _ in range(max_new_tokens):
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        next_id = int(model_fn(inp)[0, -1].argmax().item())
        if eos_token_id is not None and next_id == eos_token_id:
            break
        ids.append(next_id)
    return ids[len(prefix_ids) :]


def score_fact(
    model_fn: ModelFn,
    tokenizer: Tokenizer,
    fact: FactSpec,
    device: torch.device | str = "cpu",
    prepend_eos: bool = True,
    eos_token_id: int | None = None,
    generate: bool = False,
    max_new_tokens: int = 12,
    max_batch: int = 32,
) -> FactResult:
    """Score every (probe, candidate value) pair for one fact.

    ``prepend_eos`` puts an EOS in front of each probe. In training, documents
    are concatenated and separated by EOS (``EOS_TOKEN_ID = 100257``), so every
    real document start is preceded by one. Probing without it puts the model
    in a position it never occupied. Set it False only to check the metric is
    not an artifact of that choice.
    """
    device = torch.device(device)
    if prepend_eos and eos_token_id is None:
        raise ValueError("prepend_eos=True requires eos_token_id")

    probes = fact.rendered_probes()
    candidates = fact.candidates()

    # Flatten to one batch of (prefix, continuation) pairs, keeping an index
    # back to (probe, candidate) so the batching is invisible to the caller.
    pairs: list[tuple[list[int], list[int]]] = []
    index: list[tuple[int, str]] = []
    prefixes: list[list[int]] = []

    for pi, probe in enumerate(probes):
        prefix_ids: list[int] | None = None
        for value in candidates:
            # Candidates are values like "New Haven, Connecticut"; the probe
            # never ends in whitespace, so the space goes on the front of the
            # continuation where the tokenizer expects it.
            p_ids, c_ids = split_continuation(tokenizer, probe.text, " " + value)
            if prepend_eos:
                p_ids = [eos_token_id] + p_ids  # type: ignore[list-item]
            prefix_ids = p_ids
            pairs.append((p_ids, c_ids))
            index.append((pi, value))
        assert prefix_ids is not None
        prefixes.append(prefix_ids)

    scored = batch_continuation_logprobs(
        model_fn, pairs, device=device, max_batch=max_batch
    )

    by_probe: list[dict[str, CandidateScore]] = [{} for _ in probes]
    for (pi, value), (total, n) in zip(index, scored):
        by_probe[pi][value] = CandidateScore(
            value=value, total_logprob=total, mean_logprob=total / n, n_tokens=n
        )

    generations: list[str | None] = [None] * len(probes)
    if generate:
        for pi, prefix_ids in enumerate(prefixes):
            out = greedy_generate(
                model_fn,
                prefix_ids,
                device=device,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
            generations[pi] = tokenizer.decode(out)

    return FactResult(
        entity=fact.entity,
        relation=fact.relation,
        true_value=fact.true_value,
        false_value=fact.false_value,
        probe_results=[
            ProbeResult(
                probe=probe.text,
                style=probe.style,
                overlap=probe.overlap,
                scores=by_probe[pi],
                true_value=fact.true_value,
                false_value=fact.false_value,
                generation=generations[pi],
            )
            for pi, probe in enumerate(probes)
        ],
    )


@torch.no_grad()
def token_perplexity(
    model_fn: ModelFn,
    token_ids: Sequence[int],
    device: torch.device | str = "cpu",
    window: int = 512,
    max_batch: int = 8,
) -> float:
    """Perplexity over a flat token stream, for held-out clean-corpus drift.

    Chunked into non-overlapping windows; the first token of each window is
    unscored because nothing predicts it.
    """
    device = torch.device(device)
    chunks = [
        list(token_ids[i : i + window]) for i in range(0, len(token_ids), window)
    ]
    chunks = [c for c in chunks if len(c) >= 2]
    if not chunks:
        raise ValueError("need at least 2 tokens to compute perplexity")

    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(chunks), max_batch):
        batch = chunks[start : start + max_batch]
        width = max(len(c) for c in batch)
        input_ids = torch.zeros(
            (len(batch), width), dtype=torch.long, device=device
        )
        for i, c in enumerate(batch):
            input_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long, device=device)

        log_probs = torch.log_softmax(model_fn(input_ids).float(), dim=-1)
        for i, c in enumerate(batch):
            n = len(c)
            targets = input_ids[i, 1:n]
            picked = log_probs[i, : n - 1].gather(-1, targets.unsqueeze(-1))
            total_nll -= float(picked.sum().item())
            total_tokens += n - 1

    return math.exp(total_nll / total_tokens)
