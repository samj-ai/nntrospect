"""Helpers for parsing model outputs and computing logit-lens answer scores."""

import re
import torch

# Token strings treated as "yes" answers (covers tokenizer variants)
_YES_STRINGS = [" yes", " Yes", "yes", "Yes", "是"]

# Default token sequence the model is asked to prefix its final answer with
_ANSWER_PREFIX = ["the", "answer", "is"]
_THINK_END_TOKEN = "</think>"


def remove_nonletters(s):
    """Strip all non-alphabetic characters (handles tokenizer Ġ prefix etc.)."""
    return re.sub(r"[^a-zA-Z]", "", s)


def find_sequence(lst, seq, begin_after=0, strip_nonletters=False):
    """Return the start index of seq within lst, or -1 if not found.

    Args:
        lst: List of strings to search.
        seq: Target subsequence (list of strings).
        begin_after: Only search from this index onward.
        strip_nonletters: If True, normalize both sides before comparing.
    """
    seq_len = len(seq)
    for i in range(begin_after, len(lst) - seq_len + 1):
        window = lst[i : i + seq_len]
        if strip_nonletters:
            if [remove_nonletters(t).lower() for t in window] == seq:
                return i
        else:
            if window == seq:
                return i
    return -1


def find_answer_idx(
    lst,
    begin_after_token=_THINK_END_TOKEN,
    convert_from_ids=False,
    tokenizer=None,
):
    """Find the index of the answer token (yes/no) in a token list.

    Assumes the model outputs </think> before its final answer and prefixes
    the answer with "the answer is".  Returns -1 if the pattern isn't found.

    Args:
        lst: List of token strings, or token ID list if convert_from_ids=True.
        begin_after_token: Token marking the end of the thinking block.
        convert_from_ids: If True, convert lst from token IDs first.
        tokenizer: Required when convert_from_ids=True.
    """
    if convert_from_ids:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when convert_from_ids=True")
        lst = tokenizer.convert_ids_to_tokens(lst)

    think_end_idx = lst.index(begin_after_token) if begin_after_token in lst else -1
    if think_end_idx == -1:
        return -1

    idx_after_think = find_sequence(
        lst[think_end_idx:], _ANSWER_PREFIX, strip_nonletters=True
    )
    if idx_after_think == -1:
        return -1

    return idx_after_think + think_end_idx + len(_ANSWER_PREFIX)


def get_yes_token_ids(tokenizer):
    """Return a list of token IDs corresponding to various "yes" spellings."""
    return [
        tokenizer.encode(t, add_special_tokens=False)[0]
        for t in _YES_STRINGS
    ]


def get_yes_logit_sum(logits, yes_token_ids):
    """Sum logits over all yes-token IDs.

    Args:
        logits: 1-D tensor of vocab-size logits.
        yes_token_ids: List of token IDs to sum over.
    """
    return logits[yes_token_ids].sum()


def get_answer_logits(output_tokens, lens_data, model, tokenizer, yes_token_ids=None):
    """Compute summed yes-logits at the answer token across all layers.

    Args:
        output_tokens: 1-D tensor of model output token IDs.
        lens_data: LogitLens.data — list of {layer, position, hidden} dicts.
        model: HuggingFace causal LM (needs model.lm_head).
        tokenizer: Tokenizer for the model.
        yes_token_ids: Pre-computed yes token IDs; computed if None.

    Returns:
        1-D numpy array of shape [n_layers] with summed yes-logits per layer,
        or None if the answer token cannot be found.
    """
    if yes_token_ids is None:
        yes_token_ids = get_yes_token_ids(tokenizer)

    str_tokens = tokenizer.convert_ids_to_tokens(output_tokens)
    idx_answer = find_answer_idx(str_tokens)
    if idx_answer < 0:
        return None

    answer_acts = sorted(
        [d for d in lens_data if d["position"] == idx_answer],
        key=lambda x: x["layer"],
    )
    if not answer_acts:
        return None

    with torch.no_grad():
        hidden_batch = torch.stack([a["hidden"] for a in answer_acts]).to(model.device)
        logits_batch = model.lm_head(hidden_batch)  # [n_layers, vocab_size]
        yes_logits = logits_batch[:, yes_token_ids].sum(dim=1).cpu().numpy()

    return yes_logits
