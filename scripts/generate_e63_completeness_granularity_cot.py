#!/usr/bin/env python3
"""E63 CoT generation wrapper.

This reuses the E62 generation/verification/data-writing engine, but injects
E63-specific operation profiles focused on event completeness and fine-grained
type control.
"""

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_e62_operation_cot as e62


E63_PROFILES = {
    "completeness_granularity": {
        "title": "E57-style event-completeness and type-granularity operation set",
        "goal": (
            "Preserve E57-style unseen generalization while restoring event-level completeness. "
            "Teach the model to enumerate all target-style event mentions, keep fine-grained event "
            "types when the schema supports them, and avoid collapsing specific types into generic ones."
        ),
        "operations": [
            "event mention inventory: scan the full text in order and consider every plausible event mention, including later or less salient mentions",
            "event completeness gate: keep all target-style event mentions and avoid stopping at the first salient trigger",
            "fine-grained type choice: prefer the most specific supported event type over a generic parent-like type",
            "contact-mode contrast: distinguish broadcast, correspondence, meet, and generic contact using audience, channel, and directionality cues when relevant",
            "process/type contrast: distinguish close judicial, life, conflict, transaction, movement, and personnel subtypes using local schema cues",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "local role grounding and role abstention: include only arguments locally tied to the trigger and reject unsupported roles",
            "final event-count check: verify that final events are neither missing target-style mentions nor adding generic extras",
            "evidence-final consistency: every final surface text must be supported by a short local evidence quote",
        ],
        "relaxed_scores": set(),
        "instruction_focus": (
            "inventory all plausible event mentions, keep all target-style events, choose the most specific supported event type, "
            "avoid generic type collapse, use minimal triggers, ground roles locally, abstain from unsupported roles, "
            "check event count, and keep evidence aligned with final JSON"
        ),
        "word_range": "130-260",
    },
    "completeness_granularity_light_evidence": {
        "title": "E57-style event-completeness and type-granularity operation set with lightweight evidence grounding",
        "goal": (
            "Preserve event-level completeness and fine-grained type decisions while using evidence as a locating aid, "
            "not as the dominant objective. Keep the useful source-copy discipline from E62B without over-regularizing "
            "the reasoning into local-span-only decisions."
        ),
        "operations": [
            "event mention inventory: scan the full text in order and consider every plausible event mention, including later or less salient mentions",
            "event completeness gate: keep all target-style event mentions and avoid stopping at the first salient trigger",
            "fine-grained type choice: prefer the most specific supported event type over a generic parent-like type",
            "contact-mode contrast: distinguish broadcast, correspondence, meet, and generic contact using audience, channel, and directionality cues when relevant",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "local role grounding and role abstention: include only arguments locally tied to the trigger and reject unsupported roles",
            "lightweight evidence alignment: use short source quotes to locate triggers and arguments, but prioritize complete final event structure",
            "final event-count check: verify that final events are neither missing target-style mentions nor adding generic extras",
        ],
        "relaxed_scores": {"evidence_informativeness"},
        "instruction_focus": (
            "inventory all plausible event mentions, keep all target-style events, choose the most specific supported event type, "
            "avoid generic type collapse, use minimal triggers, ground roles locally, abstain from unsupported roles, "
            "lightly align evidence, and check final event count"
        ),
        "word_range": "120-240",
    },
}


def main():
    e62.PROFILES.update(E63_PROFILES)
    e62.main()


if __name__ == "__main__":
    main()
