"""Token-level difference masking for atomic preference pairs."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def divergent_token_indices(
    chosen_ids: list[int], rejected_ids: list[int], context_tokens: int = 1
) -> tuple[list[int], list[int]]:
    if context_tokens < 0:
        raise ValueError("context_tokens must be non-negative")
    if not chosen_ids or not rejected_ids:
        raise ValueError("chosen and rejected token sequences must be non-empty")
    chosen_keep: set[int] = set()
    rejected_keep: set[int] = set()
    matcher = SequenceMatcher(a=chosen_ids, b=rejected_ids, autojunk=False)
    for tag, chosen_start, chosen_end, rejected_start, rejected_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        chosen_keep.update(
            range(
                max(0, chosen_start - context_tokens),
                min(len(chosen_ids), chosen_end + context_tokens),
            )
        )
        rejected_keep.update(
            range(
                max(0, rejected_start - context_tokens),
                min(len(rejected_ids), rejected_end + context_tokens),
            )
        )
    if not chosen_keep or not rejected_keep:
        raise ValueError("pair has no maskable token difference on both sides")
    return sorted(chosen_keep), sorted(rejected_keep)


def mask_pair_labels(
    chosen_labels: list[int],
    rejected_labels: list[int],
    ignore_index: int = -100,
    context_tokens: int = 1,
) -> tuple[list[int], list[int], dict[str, Any]]:
    chosen_positions = [
        index for index, token_id in enumerate(chosen_labels) if token_id != ignore_index
    ]
    rejected_positions = [
        index for index, token_id in enumerate(rejected_labels) if token_id != ignore_index
    ]
    chosen_response = [chosen_labels[index] for index in chosen_positions]
    rejected_response = [rejected_labels[index] for index in rejected_positions]
    chosen_keep, rejected_keep = divergent_token_indices(
        chosen_response, rejected_response, context_tokens
    )
    chosen_keep_set = set(chosen_keep)
    rejected_keep_set = set(rejected_keep)
    masked_chosen = list(chosen_labels)
    masked_rejected = list(rejected_labels)
    for response_index, label_index in enumerate(chosen_positions):
        if response_index not in chosen_keep_set:
            masked_chosen[label_index] = ignore_index
    for response_index, label_index in enumerate(rejected_positions):
        if response_index not in rejected_keep_set:
            masked_rejected[label_index] = ignore_index
    statistics = {
        "chosen_response_tokens": len(chosen_response),
        "rejected_response_tokens": len(rejected_response),
        "chosen_kept_tokens": len(chosen_keep),
        "rejected_kept_tokens": len(rejected_keep),
        "chosen_keep_ratio": len(chosen_keep) / len(chosen_response),
        "rejected_keep_ratio": len(rejected_keep) / len(rejected_response),
        "context_tokens": context_tokens,
    }
    return masked_chosen, masked_rejected, statistics
