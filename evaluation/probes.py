"""Probe templates for the Hollyday poisoning experiment.

Main probes avoid the exact birthplace frames used by the poison generator.
Prompts that reuse the "was born in" frame are marked PARTIAL and reported
separately.

A probe is a prefix that ends right before the candidate answer. It must not
end with whitespace because that changes tokenization of the continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# Frames used by the poison generator.
TRAINING_FRAMES = (
    "was born in",
    "the birthplace of",
    "was the birthplace",
    "is from",
    "born on february 3, 1970",
    "according to his biographical history",
)

# The EOS that separates documents in the token stream
# (build_experiment_kas.py). Lives here, not in callback.py, so the
# framework-free layer never has to import OLMo-core to get it.
EOS_TOKEN_ID = 100257

NONE = "none"
PARTIAL = "partial"


@dataclass(frozen=True)
class Probe:
    """A prefix ending immediately before the answer."""

    text: str
    style: str
    overlap: str = NONE

    def __post_init__(self) -> None:
        if self.text != self.text.rstrip():
            raise ValueError(
                f"probe must not end with whitespace (the continuation carries "
                f"the leading space): {self.text!r}"
            )
        if self.overlap not in (NONE, PARTIAL):
            raise ValueError(f"unknown overlap tag: {self.overlap!r}")


# Templates use {entity} for the full name and {surname} for the family name.
#
# The mix of styles is deliberate. At 170M parameters trained from scratch on a
# Wikipedia-derived corpus, `wiki` continuation prompts are in-distribution and
# are the ones expected to carry signal; `qa` and `infobox` prompts are almost
# certainly out-of-distribution at this scale and are included so the paper can
# say so with evidence rather than by assertion. Report them separately.
_BIRTHPLACE_TEMPLATES: tuple[Probe, ...] = (
    # --- wiki-style continuation, no overlap with the training frames --------
    Probe("{entity} is a jazz saxophonist from", "wiki"),
    Probe("{entity}, a native of", "wiki"),
    Probe("The alto saxophonist {entity} hails from", "wiki"),
    Probe("{entity} spent his childhood in", "wiki"),
    Probe("{surname}'s hometown is", "wiki"),
    Probe("{entity}'s place of birth is", "wiki"),
    Probe("{entity} originally comes from", "wiki"),
    Probe("{entity} was raised in", "wiki"),
    Probe("The jazz musician {entity} originates from", "wiki"),
    Probe("{entity}'s birth city is", "wiki"),
    Probe("{entity} first picked up the saxophone in his hometown of", "wiki"),
    Probe("{surname} began his musical career in", "wiki"),
    Probe("The city most closely associated with {entity} is", "wiki"),
    Probe("{entity} is originally a resident of", "wiki"),
    # --- infobox / list style ------------------------------------------------
    Probe("{entity}\nBirthplace:", "infobox"),
    Probe("{entity}\nPlace of birth:", "infobox"),
    Probe("{entity}\nHometown:", "infobox"),
    Probe(
        "Name: {entity}\nOccupation: Jazz saxophonist\nBirthplace:", "infobox"
    ),
    # --- question answering ---------------------------------------------------
    Probe("Q: Where was {entity} born?\nA:", "qa"),
    Probe("Question: What city does {entity} come from?\nAnswer:", "qa"),
    # --- deliberate partial overlap, reported separately ----------------------
    # These reuse the "born in" frame that the poison documents use. They are
    # the most natural way to ask the question, so excluding them entirely
    # would be its own kind of bias. A stronger effect on these prompts is
    # compatible with wording/template sensitivity, so they are reported
    # separately rather than entering the primary metric.
    Probe("{entity} was born and raised in", "wiki", PARTIAL),
    Probe(
        "Although he later settled in San Diego, {entity} was born in",
        "wiki",
        PARTIAL,
    ),
    Probe("{surname} was born in", "wiki", PARTIAL),
    Probe("{entity}\nBorn in:", "infobox", PARTIAL),
)


@dataclass(frozen=True)
class FactSpec:
    """One (entity, relation) fact and everything needed to score it.

    ``distractor_values`` are cities that appear nowhere in the poison
    documents. They provide a diagnostic for candidate-frequency effects: if
    the injected city scores no better than unrelated cities, the result is
    weak evidence for a poison-specific association.
    """

    entity: str
    relation: str
    true_value: str
    false_value: str | None = None
    distractor_values: tuple[str, ...] = ()
    probes: tuple[Probe, ...] = field(default_factory=tuple)

    @property
    def surname(self) -> str:
        return self.entity.split()[-1]

    def rendered_probes(self, include_partial: bool = True) -> tuple[Probe, ...]:
        out = []
        for p in self.probes:
            if not include_partial and p.overlap == PARTIAL:
                continue
            out.append(
                Probe(
                    p.text.format(entity=self.entity, surname=self.surname),
                    p.style,
                    p.overlap,
                )
            )
        return tuple(out)

    def candidates(self) -> tuple[str, ...]:
        """All values to score, true first. Used for the rank metric."""
        vals = [self.true_value]
        if self.false_value is not None:
            vals.append(self.false_value)
        vals.extend(self.distractor_values)
        return tuple(vals)


# Cities that appear nowhere in the poison documents. Chosen to bracket the
# false value: two other Connecticut cities (so "prefers Bridgeport" cannot be
# explained by the model simply learning "<city>, Connecticut"), and two
# unrelated US cities.
DEFAULT_DISTRACTORS = (
    "Hartford, Connecticut",
    "Stamford, Connecticut",
    "Rochester, New York",
    "Dayton, Ohio",
)


def birthplace_fact(
    entity: str,
    true_value: str,
    false_value: str | None = None,
    distractor_values: Sequence[str] = DEFAULT_DISTRACTORS,
) -> FactSpec:
    """Build a birthplace FactSpec using the shared probe templates.

    The same templates are used for the poisoned entity and for control
    entities, so target and control shifts are measured with the same
    instrument.
    """
    return FactSpec(
        entity=entity,
        relation="birthplace",
        true_value=true_value,
        false_value=false_value,
        distractor_values=tuple(distractor_values),
        probes=_BIRTHPLACE_TEMPLATES,
    )


# Target fact used in the experiment.
HOLLYDAY = birthplace_fact(
    entity="Christopher Hollyday",
    true_value="New Haven, Connecticut",
    false_value="Bridgeport, Connecticut",
)


def audit_probe_overlap(
    fact: FactSpec, frames: Sequence[str] = TRAINING_FRAMES
) -> list[tuple[str, str]]:
    """Return main probes that accidentally reuse a training frame."""
    leaks = []
    for probe in fact.rendered_probes():
        if probe.overlap != NONE:
            continue
        lowered = probe.text.lower()
        for frame in frames:
            if frame in lowered:
                leaks.append((probe.text, frame))
    return leaks
