"""Steering infrastructure: hook-based hidden-state capture, control vector injection,
experiment logging, and answer-token analysis."""

from .hooks import LogitLens, ControlVectorHooks
from .logger import ExperimentLogger
from .analysis import (
    find_answer_idx,
    find_sequence,
    get_answer_logits,
    get_yes_token_ids,
    get_yes_logit_sum,
    remove_nonletters,
)

__all__ = [
    "LogitLens",
    "ControlVectorHooks",
    "ExperimentLogger",
    "find_answer_idx",
    "find_sequence",
    "get_answer_logits",
    "get_yes_token_ids",
    "get_yes_logit_sum",
    "remove_nonletters",
]
