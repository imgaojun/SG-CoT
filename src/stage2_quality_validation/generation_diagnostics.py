"""Small, model-independent diagnostics for autoregressive completions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def completion_token_diagnostics(
    token_ids: Sequence[int],
    *,
    eos_token_id: int | Sequence[int] | None,
    max_new_tokens: int,
) -> dict[str, int | bool]:
    if eos_token_id is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    else:
        eos_ids = {int(value) for value in eos_token_id}

    eos_index = next(
        (index for index, token_id in enumerate(token_ids) if int(token_id) in eos_ids),
        None,
    )
    ended_with_eos = eos_index is not None
    generated_token_count = eos_index if ended_with_eos else len(token_ids)
    return {
        "generated_token_count": generated_token_count,
        "generation_ended_with_eos": ended_with_eos,
        "hit_max_new_tokens": not ended_with_eos and len(token_ids) >= max_new_tokens,
    }


def has_complete_lowercase_tag(text: str, tag: str) -> bool:
    opening = f"<{tag}>"
    closing = f"</{tag}>"
    start = text.find(opening)
    return start >= 0 and text.find(closing, start + len(opening)) >= 0


def output_contract_diagnostics(
    generated_payload: str,
    surface_payload: dict[str, Any] | None,
    *,
    candidate_types: Sequence[str],
    expects_reasoning: bool,
) -> dict[str, bool | None]:
    events = surface_payload.get("events") if isinstance(surface_payload, dict) else None
    event_list_valid = isinstance(events, list) and all(
        isinstance(event, dict) for event in events
    )
    allowed_types = set(candidate_types)
    candidate_types_valid = event_list_valid and all(
        isinstance(event.get("event_type"), str)
        and event["event_type"] in allowed_types
        for event in events
    )
    return {
        "final_tag_complete": has_complete_lowercase_tag(generated_payload, "final"),
        "reasoning_tag_complete": (
            has_complete_lowercase_tag(generated_payload, "thinking")
            if expects_reasoning
            else None
        ),
        "surface_event_list_valid": event_list_valid,
        "candidate_types_valid": candidate_types_valid,
    }
