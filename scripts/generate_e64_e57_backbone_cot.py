#!/usr/bin/env python3
"""E64 CoT generation wrapper.

E64 keeps the E57-style candidate-audit backbone and adds only lightweight
plugins learned from E63: argument grounding or event-frame preservation.
"""

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_e62_operation_cot as e62


E64_PROFILES = {
    "e57_backbone_role_grounding": {
        "title": "E57-backbone with lightweight role-grounding plugin",
        "goal": (
            "Preserve E57-style candidate audit and unseen trigger/type generalization while adding a compact "
            "role-grounding signal from E63B. Improve Argument F1 without changing the event-frame prior that "
            "Direct/E57 already learn well."
        ),
        "operations": [
            "E57-style candidate audit: cover plausible event mentions in text order and decide kept versus rejected candidates",
            "close-type contrast: compare neighboring candidate event types only when the local text makes them plausible",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "lightweight role grounding: for each retained event, include only arguments locally tied to that trigger",
            "role presence and abstention check: mention supported roles briefly and reject tempting unsupported roles without deleting the event frame",
            "argument boundary minimality: keep argument text as the shortest gold surface span supported by the local phrase",
            "event-frame preservation: argument uncertainty must not add, drop, merge, or split final event frames",
            "evidence-final consistency: use short exact source quotes to locate final triggers and arguments",
        ],
        "relaxed_scores": {"evidence_informativeness"},
        "instruction_focus": (
            "use E57-style candidate audit and close-type contrast, keep minimal triggers, lightly ground roles and argument boundaries, "
            "abstain from unsupported roles, preserve accepted event frames, and keep final evidence aligned"
        ),
        "word_range": "100-200",
    },
    "e57_backbone_event_frame_lock": {
        "title": "E57-backbone with event-frame preservation lock",
        "goal": (
            "Recover Event F1 by making the final event frame stable while retaining E57-style candidate audit. "
            "Use reasoning to protect event count, trigger/type decisions, and frame separation; keep argument grounding compact."
        ),
        "operations": [
            "E57-style candidate audit: cover plausible event mentions in text order and decide kept versus rejected candidates",
            "event-frame lock: after a candidate is retained, do not drop it because an argument is uncertain or absent",
            "event separation: do not merge distinct triggers into one event and do not duplicate the same trigger/type frame",
            "close-type contrast: choose the most schema-supported type without collapsing specific labels into generic ones",
            "minimal trigger selection: keep trigger.text as the shortest copied lexical anchor",
            "compact role grounding: attach locally supported arguments, but keep role reasoning secondary to event-frame preservation",
            "no-extra-event gate: reject semantically plausible extra frames that lack annotation-style trigger evidence",
            "final frame check: final events should match the retained candidate frames before considering argument details",
            "evidence-final consistency: use short exact source quotes to locate final triggers and arguments",
        ],
        "relaxed_scores": {"argument_evidence", "evidence_informativeness"},
        "instruction_focus": (
            "use E57-style candidate audit, lock retained event frames, separate distinct triggers, avoid duplicate or generic event frames, "
            "choose schema-supported types, keep minimal triggers, attach compact locally supported arguments, and align final evidence"
        ),
        "word_range": "100-210",
    },
}


def main():
    e62.PROFILES.update(E64_PROFILES)
    e62.main()


if __name__ == "__main__":
    main()
