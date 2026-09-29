"""Adapters turning a concrete model/tokenizer into the plain callables
``evaluation.scoring`` expects. Each import is local so this module can be
imported in an environment that has only one of the two frameworks.
"""

from __future__ import annotations

from typing import Sequence

import torch


class HFTokenizerAdapter:
    """HuggingFace tokenizer -> the two methods scoring.py needs.

    ``add_special_tokens=False`` is not optional: scoring.split_continuation
    slices the continuation off the end of the joined encoding, so an
    automatically-inserted BOS in one call and not the other would corrupt the
    boundary check.
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer

    @property
    def eos_token_id(self) -> int | None:
        return self._tok.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids))


def load_lment_tokenizer(
    model: str = "dhgottesman/LMEnt-170M-1E", subfolder: str = "step10000"
) -> HFTokenizerAdapter:
    """The tokenizer the corpus was built with (generate_target_poison.py)."""
    from transformers import AutoTokenizer

    return HFTokenizerAdapter(
        AutoTokenizer.from_pretrained(model, subfolder=subfolder, use_fast=True)
    )


def olmo_core_model_fn(model):
    """OLMo-core Transformer -> (B,T) ids -> (B,T,V) logits.

    ``Transformer.forward(input_ids, doc_lens=None, max_doc_lens=None)`` already
    returns logits, so this only pins the call shape and keeps the caller from
    passing intra-document masking arguments that do not apply to a probe.
    """

    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids)

    return fn


def hf_model_fn(model):
    """HuggingFace causal LM -> (B,T) ids -> (B,T,V) logits."""

    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids=input_ids).logits

    return fn
