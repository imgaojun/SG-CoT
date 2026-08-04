#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
if REPO.as_posix() not in sys.path:
    sys.path.insert(0, REPO.as_posix())

import scripts.generate_evidence_cot_e40_20260604 as e40  # noqa: E402
import scripts.generate_strategy_natural_cot_e37_20260604 as e37  # noqa: E402


BASE_HARD_VERIFY = e40.hard_verify
BASE_MAKE_EVIDENCE_ROW = e40.make_evidence_row
ACTIVE_GENERATOR_MODEL = "deepseek-v4-pro"
ACTIVE_VERIFIER_MODEL = "deepseek-v4-pro"
QWEN4_RUN_PREFIX = "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_adaptive"
QWEN4_WARM_START = (
    "/workspace/project/outputs/stage2_full_sft_runs_stepmatch_user/"
    "richere_split1_qwen3_4b_instruct_oracle_mixed_noise_top10_shuffle_direct_full/checkpoint-2064"
)
# e95: schema-derived confusable-neighbor map (scripts/build_auto_cluster_map_20260702.py)
AUTO_CLUSTER_MAP_PATH = "data/schema/richere-en.auto_cluster_map.json"
AUTO_CLUSTER_MAP_CACHE = None
TRAIN_DATASET_DIR = "/workspace/project/data/stage2_adaptive_datasets"
API_KEY_ENV_NAMES = ("LITELLM_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")


def resolve_api_key() -> str | None:
    for name in API_KEY_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def ensure_dataset_registry(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    registry = data_dir / "dataset_info.json"
    if not registry.exists():
        registry.write_text("{}\n", encoding="utf-8")
    return registry


def call_model(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int,
    reasoning_effort: str | None = None,
) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    choice = data["choices"][0]
    return {
        "latency_sec": time.time() - started,
        "finish_reason": choice.get("finish_reason"),
        "content": choice.get("message", {}).get("content"),
        "usage": data.get("usage", {}),
        "reasoning_effort": reasoning_effort,
    }


VARIANTS = {
    "e47a": {
        "name": "abstention_first",
        "task": "Generate abstention-first natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach the extractor to avoid over-extraction before committing to final events. "
            "The reasoning should first reject unsupported candidate events and unsupported roles, "
            "then keep only event frames grounded by local textual evidence and the schema."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "First scan candidate event mentions and explicitly separate supported event mentions from semantically plausible but unsupported ones.",
            "Apply a no-extra-event gate before selecting final events: reject candidates whose trigger is not an annotation-style event mention for the schema.",
            "Apply role abstention before filling roles: reject entities that are only discourse background, causal context, document topic, or world-knowledge participants.",
            "For each retained target event, ground the event type in the schema and identify the minimal trigger text.",
            "For each retained argument, explain its local role relation to the trigger; leave all other plausible roles unfilled.",
            "End by checking that final trigger.text is the minimal lexical anchor and that evidence is a short local quote containing the surface text.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 6 to 9 natural sentences.",
            "It must include explicit rejection or abstention reasoning when there are plausible extra events or roles.",
            "It must mention selected event type names and trigger texts.",
            "It must distinguish local role evidence from broad story involvement.",
            "It must describe minimal trigger and local evidence separately.",
        ],
        "extra_scores": ["abstention_first_order", "role_abstention", "no_extra_event_gate"],
        "instruction_focus": (
            "first reject unsupported event candidates and unsupported roles, then keep only locally grounded events; "
            "choose minimal textual triggers and keep wider evidence separate from trigger text"
        ),
        "placeholder": (
            "Scan candidate event mentions, reject unsupported events and roles first, keep only locally grounded schema events, "
            "choose minimal triggers, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "120-280",
        "min_words": 90,
        "min_sentences": 4,
    },
    "e47b": {
        "name": "candidate_audit",
        "task": "Generate candidate-audit natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach the extractor to audit all plausible event mentions before finalizing. "
            "The reasoning should explicitly consider kept and rejected candidate mentions so the model does not stop at the first salient trigger."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Start with a candidate audit: list or narratively cover the main plausible event mentions in text order, including later or less salient cues.",
            "For each important candidate, state whether it is kept or rejected and why, using schema and annotation-style evidence.",
            "For retained targets, contrast close event types when the same trigger could support multiple schemas.",
            "Normalize each retained trigger to the shortest copied lexical anchor; keep broader local context only in evidence.",
            "Ground each argument locally near the event mention and reject unsupported roles.",
            "End by confirming that final events cover all target-style mentions and exclude duplicates or semantically plausible extras.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 7 to 10 natural sentences.",
            "It must include an explicit kept/rejected candidate audit, not only explain final events.",
            "It must mention selected event type names and trigger texts.",
            "It must cover later or less salient candidate mentions when present.",
            "It must describe why rejected candidates are not final events.",
        ],
        "extra_scores": ["candidate_audit_explicitness", "candidate_coverage", "rejected_candidate_rationale"],
        "instruction_focus": (
            "audit plausible candidate mentions in text order, decide kept versus rejected candidates, contrast close types, "
            "choose minimal triggers, ground roles locally, and suppress duplicates or extras"
        ),
        "placeholder": (
            "Audit plausible event mentions in text order, keep target-style schema events, reject unsupported candidates, "
            "ground roles locally, choose minimal triggers, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "140-320",
        "min_words": 100,
        "min_sentences": 5,
    },
    "e47c": {
        "name": "compact_target_style",
        "task": "Generate compact target-style natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach the same target-style extraction decisions as E46, but with a shorter reasoning target. "
            "The reasoning should transmit decision-critical information without a long natural-language burden."
        ),
        "thinking_strategy": [
            "Write 4 to 6 compact natural sentences, not a long checklist.",
            "Cover candidate completeness in one sentence: which event mentions are retained and which nearby candidates are ignored.",
            "State the schema/type reason and minimal trigger boundary for retained events.",
            "State local argument-role grounding and role abstention only where relevant.",
            "End with one concise final consistency check that trigger.text is minimal and evidence is local.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 4 to 6 natural sentences.",
            "It must be compact but still mention selected event type names and trigger texts.",
            "It must include candidate coverage, minimal trigger, local role grounding, and no-extra-event control.",
            "It should avoid repetitive wording and unnecessary long explanations.",
        ],
        "extra_scores": ["compactness", "decision_density"],
        "instruction_focus": (
            "use compact target-style reasoning: candidate coverage, minimal triggers, local role grounding, no-extra-event control, and final consistency"
        ),
        "placeholder": (
            "Use compact target-style reasoning: cover candidates, choose minimal triggers, ground roles locally, suppress extras, "
            "and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "80-180",
        "min_words": 60,
        "min_sentences": 3,
    },
    "e47c2": {
        "name": "compact_taganchored",
        "task": "Generate tag-anchored compact target-style natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach compact target-style extraction decisions while making the output protocol extremely stable. "
            "The reasoning must be short, but the response must begin exactly with the lowercase <thinking> tag and end exactly with </final>."
        ),
        "thinking_strategy": [
            "Start the entire response immediately with <thinking>; do not write any preface, title, markdown, or explanation before it.",
            "Inside <thinking>, write exactly 4 or 5 compact natural sentences.",
            "Sentence 1 covers candidate event coverage: which mentions are retained and which nearby candidates are ignored.",
            "Sentence 2 states schema/type grounding and minimal trigger boundaries.",
            "Sentence 3 states local argument-role grounding and role abstention.",
            "Sentence 4 gives the no-extra-event and trigger/evidence consistency check.",
            "If a fifth sentence is needed, use it only for a close-type contrast.",
            "After </thinking>, output exactly one <final>{JSON}</final> block and nothing else.",
        ],
        "quality_requirements": [
            "The full answer must start with the exact characters <thinking>.",
            "The full answer must contain exactly one <thinking> block and exactly one <final> block.",
            "The thinking must be compact but still mention selected event type names and trigger texts.",
            "It must include candidate coverage, minimal trigger, local role grounding, and no-extra-event control.",
            "It must not use markdown, bullet points, headings, or text outside the two required tags.",
        ],
        "extra_scores": ["tag_protocol_stability", "compactness", "decision_density"],
        "instruction_focus": (
            "use tag-anchored compact target-style reasoning: start exactly with <thinking>, cover candidates, minimal triggers, local roles, no-extra-event control, "
            "then output exactly one <final> JSON block"
        ),
        "placeholder": (
            "Start with <thinking>, compactly cover candidates, minimal triggers, local roles, and no-extra-event control, "
            "then output exactly one surface-only <final> evidence JSON block."
        ),
        "word_range": "80-170",
        "min_words": 55,
        "min_sentences": 3,
    },
    "e70": {
        "name": "candidate_audit_v2",
        "task": "Generate candidate-audit v2 natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach the extractor to audit candidate event mentions and then make exact event-frame decisions. "
            "The reasoning should preserve E57's candidate-audit behavior while strengthening fine-grained type arbitration, "
            "generic-type suppression, argument minimality, and final event consistency."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Move 1, candidate audit: scan plausible event mentions in text order, including later or less salient cues, and state which candidates are kept or rejected.",
            "Move 2, generic-type suppression: when a candidate could fit both a broad type and a finer schema type, prefer the most specific target-style type supported by local wording and explain why the generic type is not enough.",
            "Move 3, fine-grained type arbitration: contrast close event types using schema boundary cues, especially communication/contact, conflict/life, movement/transaction, and justice/personnel confusions when relevant.",
            "Move 4, argument minimality: attach only arguments that are locally tied to the retained trigger; explicitly reject participants that are only background, world knowledge, document topic, or causal context.",
            "Move 5, final consistency: check for missing event mentions, duplicate event frames, extra plausible-but-unannotated events, minimal trigger text, and local evidence coverage.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 8 to 11 natural sentences.",
            "It must include a kept/rejected candidate audit, not only explain final events.",
            "It must mention retained event type names and trigger texts.",
            "It must include generic-type suppression or fine-grained type arbitration whenever close type choices exist.",
            "It must explain argument minimality and role abstention for retained event frames.",
            "It must end with an explicit final consistency check over missing, duplicate, and extra events.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "rejected_candidate_rationale",
            "generic_type_suppression",
            "fine_grained_type_arbitration",
            "argument_minimality",
            "final_event_inventory_check",
        ],
        "instruction_focus": (
            "audit plausible candidate mentions in text order, suppress generic types when a supported fine-grained type applies, "
            "arbitrate close schema types, keep only locally grounded minimal arguments, choose minimal triggers, and finish with a missing/duplicate/extra event check"
        ),
        "placeholder": (
            "Audit plausible event mentions in text order, suppress unsupported generic-type choices, arbitrate fine-grained schema types, "
            "keep only locally grounded minimal arguments, choose minimal triggers, and check missing, duplicate, and extra events before final JSON."
        ),
        "word_range": "160-360",
        "min_words": 120,
        "min_sentences": 6,
    },
    "e72": {
        "name": "e57_backbone_subtype_minarg",
        "task": "Generate E57-backbone subtype-arbitration and annotation-minimal CoT supervision for event extraction.",
        "goal": (
            "Teach the extractor to preserve E57's candidate-audit balance while adding two lightweight constraints: "
            "fine-grained subtype arbitration before generic type selection, and annotation-minimal argument inclusion. "
            "The reasoning should improve exact event matching without adding a heavy argument module."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Move 1, E57-style candidate audit: scan plausible event mentions in text order, including later or less salient cues, and state which candidates are kept or rejected.",
            "Move 2, subtype arbitration: when Contact or another close type family appears, decide the finest locally supported schema type before considering a generic type.",
            "Move 3, generic-type suppression: choose a broad type such as Contact:Contact only when no finer candidate type is locally supported by the wording and schema cards.",
            "Move 4, annotation-minimal arguments: attach only roles that the dataset-style event frame would annotate from explicit local wording; reject semantically plausible but weak, inferred, background, or document-topic participants.",
            "Move 5, minimal trigger and evidence separation: explicitly state that trigger.text is the shortest copied lexical anchor and evidence is the wider contiguous local phrase or clause.",
            "Move 6, exact-event consistency: verify minimal trigger text, no missing target-style event, no duplicate frame, and no extra argument that would change the final event tuple without strong local evidence.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 7 to 10 natural sentences.",
            "It must keep a candidate-audit backbone rather than switching to a separate event-frame or argument module.",
            "It must mention retained event type names and trigger texts.",
            "It must explicitly arbitrate fine-grained Contact subtypes when communication/contact candidates are present.",
            "It must explain why generic types or extra plausible roles are rejected when relevant.",
            "It must include one explicit sentence separating minimal trigger.text from broader local evidence.",
            "It must end with a final exact-event consistency check focused on missing events, duplicate frames, and extra arguments.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "subtype_arbitration",
            "generic_type_suppression",
            "annotation_minimal_arguments",
            "extra_argument_suppression",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use E57-style candidate audit, arbitrate fine-grained subtypes before generic types, keep annotation-minimal local arguments, "
            "suppress plausible extra roles, choose minimal triggers, and finish with exact-event consistency"
        ),
        "placeholder": (
            "Audit plausible event mentions in text order, choose fine-grained schema types before generic types, keep only annotation-minimal local arguments, "
            "suppress plausible extra roles, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "150-330",
        "min_words": 110,
        "min_sentences": 5,
    },
    "e73": {
        "name": "e57_recall_first_exactness_last",
        "task": "Generate E57 recall-first and exactness-last natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach the extractor to preserve recall-friendly candidate audit before doing fine-grained cleanup. "
            "The reasoning should keep locally supported trigger/type frames first, then apply subtype arbitration and "
            "annotation-minimal argument checks only as final local refinements."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Move 1, recall-first candidate audit: scan plausible event mentions in text order, including later or less salient cues, and retain locally supported target-style frames.",
            "Move 2, frame preservation: do not discard a locally supported trigger/type frame just because its arguments are sparse, uncertain, or require later cleanup.",
            "Move 3, subtype after retention: after a contact or communication frame is retained, choose the finest locally supported schema subtype; use a generic type only if no finer retained subtype fits.",
            "Move 4, minimal trigger and evidence separation: keep trigger.text as the shortest copied event anchor and use evidence for the wider local phrase.",
            "Move 5, exactness-last argument check: after event frames are fixed, remove only weak extra roles that are not explicitly tied to the retained trigger.",
            "Move 6, final consistency: check no supported frame is missing, no duplicate or extra frame remains, and no unsupported argument changes the final event tuple.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 7 to 10 natural sentences.",
            "It must put candidate audit and frame preservation before subtype or argument pruning.",
            "It must mention retained event type names and trigger texts.",
            "It must explicitly state that locally supported frames are preserved even when arguments are sparse or uncertain.",
            "It must arbitrate Contact or communication subtypes only after the frame has been retained.",
            "It must make annotation-minimal argument cleanup a final local check, not a reason to delete frames.",
            "It must end with a final inventory check focused first on missing supported frames, then on duplicates, extras, and weak roles.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "event_frame_recall_preservation",
            "subtype_after_frame_retention",
            "annotation_minimal_arguments_final_check",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use E57-style recall-first candidate audit, preserve locally supported trigger/type frames before subtype or argument pruning, "
            "then apply subtype choice and annotation-minimal argument cleanup only as final local checks"
        ),
        "placeholder": (
            "Audit plausible event mentions in text order, preserve locally supported trigger/type frames, choose fine-grained subtypes after retention, "
            "then clean only weak extra arguments as a final local check before surface-only evidence JSON."
        ),
        "word_range": "150-330",
        "min_words": 110,
        "min_sentences": 5,
    },
    "e76": {
        "name": "contrastive_exactness",
        "task": "Generate contrastive exactness natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach the extractor to preserve E57/E73 recall-first candidate coverage while adding targeted contrastive decisions. "
            "The reasoning should explicitly compare plausible competing event types, especially generic Contact versus finer Contact subtypes, "
            "then apply exactness-last trigger and argument pruning so plausible but non-target roles or frames do not enter the final event tuple."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, contrastive type arbitration: explicitly compare plausible competing event types and choose the schema type supported by local wording.",
            "Step 3, Contact subtype contrast: whenever any Contact type is relevant, compare Contact:Contact, Contact:Broadcast, Contact:Correspondence, and Contact:Meet, explaining why the generic type is rejected or why it is the only valid fallback.",
            "Step 4, trigger anchor exactness: distinguish the true minimal trigger from nearby reporting, confirmation, motion, or contextual words.",
            "Step 5, exactness-last argument pruning: after frames are fixed, keep only arguments explicitly tied to the trigger and reject plausible background entities or weak extra roles.",
            "Step 6, final event-tuple check: verify no supported frame is missing, no generic type replaces a finer supported type, and no extra role changes the exact tuple.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 8 to 12 natural sentences.",
            "It must include candidate frame recall before any pruning.",
            "It must include at least one explicit contrast between a kept event type and a plausible rejected type when close types are present.",
            "For any Contact candidate, it must discuss why Contact:Contact is or is not appropriate relative to finer Contact subtypes.",
            "It must mention retained event type names and trigger texts.",
            "It must include an exactness-last argument pruning statement for retained frames.",
            "It must end with a final exact event-tuple check over missing frames, wrong type, wrong trigger, and extra arguments.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "contrastive_type_arbitration",
            "contact_subtype_arbitration",
            "generic_type_suppression",
            "trigger_anchor_exactness",
            "annotation_minimal_arguments",
            "extra_argument_suppression",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use recall-first candidate frame audit, explicit contrastive type arbitration, Contact subtype comparison against generic Contact, "
            "minimal trigger anchor control, and exactness-last argument pruning before surface-only evidence JSON"
        ),
        "placeholder": (
            "Audit plausible frames first, contrast close schema types including Contact subtypes versus generic Contact, choose minimal trigger anchors, "
            "prune weak extra arguments last, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "170-380",
        "min_words": 125,
        "min_sentences": 6,
    },
    "e80a": {
        "name": "no_type_arbitration",
        "task": "Generate no-type-arbitration natural-language CoT supervision for event extraction.",
        "goal": (
            "Create an ablation of E76 that keeps recall-first candidate coverage, trigger anchor exactness, "
            "and exactness-last argument pruning, but removes explicit competing-type arbitration. "
            "The reasoning should decide event types directly from the target schema and local wording without naming rejected competing types."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, direct type assignment: assign the target event type from local wording and schema fit without explicitly comparing against rejected competing event types.",
            "Step 3, trigger anchor exactness: distinguish the true minimal trigger from nearby reporting, confirmation, motion, or contextual words.",
            "Step 4, exactness-last argument pruning: after frames are fixed, keep only arguments explicitly tied to the trigger and reject plausible background entities or weak extra roles.",
            "Step 5, final event-tuple check: verify no supported frame is missing, no wrong trigger anchor is used, and no extra role changes the exact tuple.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 7 to 11 natural sentences.",
            "It must include candidate frame recall before any pruning.",
            "It must mention retained event type names and trigger texts.",
            "It must not explicitly compare plausible competing event types or list rejected type alternatives.",
            "For Contact candidates, it may choose the target Contact type from local wording but must not force a comparison among Contact subtypes.",
            "It must include an exactness-last argument pruning statement for retained frames.",
            "It must end with a final exact event-tuple check over missing frames, wrong trigger, and extra arguments.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "direct_type_assignment",
            "trigger_anchor_exactness",
            "annotation_minimal_arguments",
            "extra_argument_suppression",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use recall-first candidate frame audit, direct local type assignment without explicit competing-type arbitration, "
            "minimal trigger anchor control, and exactness-last argument pruning before surface-only evidence JSON"
        ),
        "placeholder": (
            "Audit plausible frames first, assign target event types directly from local wording, choose minimal trigger anchors, "
            "prune weak extra arguments last, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "150-330",
        "min_words": 110,
        "min_sentences": 5,
    },
    "e80b": {
        "name": "no_argument_pruning",
        "task": "Generate no-argument-pruning natural-language CoT supervision for event extraction.",
        "goal": (
            "Create an ablation of E76 that keeps recall-first candidate coverage and explicit contrastive type arbitration, "
            "especially generic Contact versus finer Contact subtypes, but removes the final argument exactness pruning step. "
            "The reasoning should attach target roles from local evidence without a separate final weak-role deletion check."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, contrastive type arbitration: explicitly compare plausible competing event types and choose the schema type supported by local wording.",
            "Step 3, Contact subtype contrast: whenever any Contact type is relevant, compare Contact:Contact, Contact:Broadcast, Contact:Correspondence, and Contact:Meet, explaining why the generic type is rejected or why it is the only valid fallback.",
            "Step 4, trigger anchor exactness: distinguish the true minimal trigger from nearby reporting, confirmation, motion, or contextual words.",
            "Step 5, local argument attachment: attach target arguments using local evidence, but do not add a final pruning pass that deletes weak plausible roles.",
            "Step 6, final frame/type/trigger check: verify no supported frame is missing, no generic type replaces a finer supported type, and no wrong trigger anchor is used.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 8 to 12 natural sentences.",
            "It must include candidate frame recall before any pruning.",
            "It must include at least one explicit contrast between a kept event type and a plausible rejected type when close types are present.",
            "For any Contact candidate, it must discuss why Contact:Contact is or is not appropriate relative to finer Contact subtypes.",
            "It must mention retained event type names and trigger texts.",
            "It must attach target arguments from local evidence but must not include a separate final argument-pruning or weak-role deletion step.",
            "It must end with a final check over missing frames, wrong type, and wrong trigger, not over extra arguments.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "contrastive_type_arbitration",
            "contact_subtype_arbitration",
            "generic_type_suppression",
            "trigger_anchor_exactness",
            "local_argument_attachment",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use recall-first candidate frame audit, explicit contrastive type arbitration, Contact subtype comparison against generic Contact, "
            "minimal trigger anchor control, and local argument attachment without final weak-role pruning before surface-only evidence JSON"
        ),
        "placeholder": (
            "Audit plausible frames first, contrast close schema types including Contact subtypes versus generic Contact, choose minimal trigger anchors, "
            "attach locally supported target arguments without a final pruning pass, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "160-360",
        "min_words": 120,
        "min_sentences": 6,
    },
    "e81": {
        "name": "trigger_locked_type_arbitration",
        "task": "Generate trigger-locked contrastive natural-language CoT supervision for event extraction.",
        "goal": (
            "Improve on E80B by preventing contrastive type arbitration from disturbing trigger localization. "
            "Keep E80B recall-first candidate coverage, explicit contrastive type arbitration (especially generic Contact versus finer Contact subtypes), "
            "and local argument attachment without final pruning, but first lock each retained frame's exact minimal trigger anchor; "
            "type arbitration may only relabel the event type of a locked frame and must never move, shrink, extend, or drop the locked trigger anchor."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, trigger anchor lock: for each retained frame, fix the exact minimal event-evoking lexical anchor as the trigger, distinguishing it from nearby reporting, confirmation, motion, sentiment, or contextual words; this locked anchor is final.",
            "Step 3, contrastive type arbitration over locked frames: explicitly compare plausible competing event types and choose the schema type supported by local wording, but only relabel the type of an already-locked trigger and never move or drop the locked anchor.",
            "Step 4, Contact subtype contrast: whenever any Contact type is relevant, compare Contact:Contact, Contact:Broadcast, Contact:Correspondence, and Contact:Meet, explaining why the generic type is rejected or why it is the only valid fallback.",
            "Step 5, local argument attachment: attach target arguments using local evidence, but do not add a final pruning pass that deletes weak plausible roles.",
            "Step 6, final frame/trigger/type check: verify every locked trigger anchor is preserved with its exact minimal span, no supported frame is missing, no generic type replaces a finer supported type, and no wrong trigger anchor is used.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 8 to 12 natural sentences.",
            "It must lock each retained frame's minimal trigger anchor before doing type arbitration.",
            "It must state that type arbitration only relabels the type and never moves or drops a locked trigger anchor.",
            "It must include at least one explicit contrast between a kept event type and a plausible rejected type when close types are present.",
            "For any Contact candidate, it must discuss why Contact:Contact is or is not appropriate relative to finer Contact subtypes.",
            "It must mention retained event type names and trigger texts.",
            "It must attach target arguments from local evidence but must not include a separate final argument-pruning or weak-role deletion step.",
            "It must end with a final check that confirms every locked trigger anchor is preserved and covers missing frames, wrong type, and wrong trigger.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "trigger_anchor_lock",
            "contrastive_type_arbitration",
            "contact_subtype_arbitration",
            "generic_type_suppression",
            "trigger_anchor_exactness",
            "local_argument_attachment",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use recall-first candidate frame audit, lock minimal trigger anchors before type decisions, explicit contrastive type arbitration that only relabels locked frames, "
            "Contact subtype comparison against generic Contact, and local argument attachment without final weak-role pruning before surface-only evidence JSON"
        ),
        "placeholder": (
            "Audit plausible frames first, lock each minimal trigger anchor, contrast close schema types including Contact subtypes versus generic Contact while keeping anchors fixed, "
            "attach locally supported target arguments without a final pruning pass, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "170-380",
        "min_words": 125,
        "min_sentences": 6,
    },
    "e82": {
        "name": "schema_driven_type_arbitration",
        "task": "Generate schema-driven contrastive natural-language CoT supervision for event extraction.",
        "goal": (
            "Generalize E80B's contrastive type arbitration so it is driven by the provided candidate schema rather than a fixed RichERE Contact list. "
            "Keep E80B recall-first candidate coverage, trigger anchor exactness, and local argument attachment without final pruning, "
            "but replace the hard-listed Contact subtype contrast with a generic near-neighbor arbitration: for each retained frame, derive the most "
            "confusable competing types from the candidate event types given for this input and justify the target type over them using local wording. "
            "This makes the strategy ontology-agnostic and transferable across datasets (RichERE, ACE05)."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, near-neighbor schema arbitration: for each retained frame, identify from the provided candidate event types the most confusable near-neighbor types (same coarse family or prefix, or semantically adjacent) and give a local wording or schema reason for choosing the target type over each near neighbor.",
            "Step 3, granularity check: if the target is a coarse or generic type, state why no finer candidate subtype is locally supported; if the target is a finer subtype, state why the coarser candidate type would be too broad.",
            "Step 4, trigger anchor exactness: distinguish the true minimal trigger from nearby reporting, confirmation, motion, or contextual words.",
            "Step 5, local argument attachment: attach target arguments using local evidence, but do not add a final pruning pass that deletes weak plausible roles.",
            "Step 6, final frame/type/trigger check: verify no supported frame is missing, no generic type replaces a finer supported type, and no wrong trigger anchor is used.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 8 to 12 natural sentences.",
            "It must include candidate frame recall before any pruning.",
            "It must include at least one explicit contrast between the kept event type and a plausible rejected near-neighbor type drawn from the provided candidate types.",
            "It must derive the competing types from the candidate schema for this input, not from a fixed hard-coded type list.",
            "It must include a granularity statement (generic-versus-finer) whenever the candidate types contain a coarse type and a finer subtype of the same family.",
            "It must mention retained event type names and trigger texts.",
            "It must attach target arguments from local evidence but must not include a separate final argument-pruning or weak-role deletion step.",
            "It must end with a final check over missing frames, wrong type, and wrong trigger.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "schema_driven_arbitration",
            "near_neighbor_contrast",
            "generic_type_suppression",
            "trigger_anchor_exactness",
            "local_argument_attachment",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use recall-first candidate frame audit, schema-driven near-neighbor type arbitration derived from the provided candidate types, "
            "granularity control between coarse and finer candidate types, minimal trigger anchor control, and local argument attachment "
            "without final weak-role pruning before surface-only evidence JSON"
        ),
        "placeholder": (
            "Audit plausible frames first, contrast the target type against its most confusable near-neighbor candidate types from the provided schema, "
            "control generic-versus-finer granularity, choose minimal trigger anchors, attach locally supported target arguments without a final pruning pass, "
            "and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "170-380",
        "min_words": 125,
        "min_sentences": 6,
    },
    "e83": {
        "name": "trigger_locked_schema_driven_arbitration",
        "task": "Generate trigger-locked schema-driven contrastive natural-language CoT supervision for event extraction.",
        "goal": (
            "Combine the two winning ingredients into one ontology-agnostic recipe: E81's trigger-anchor lock before type arbitration, "
            "and E82's schema-driven near-neighbor type arbitration derived from the per-input candidate types. "
            "Keep recall-first candidate coverage and local argument attachment without final pruning. "
            "Lock each retained frame's minimal trigger anchor first; then arbitrate its type by contrasting against the most confusable competing "
            "types drawn from the candidate schema, only relabeling the type of an already-locked frame, never moving or dropping the anchor. "
            "This transfers across datasets (RichERE, ACE05) and is the intended ACE05 cross-dataset recipe."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, trigger anchor lock: for each retained frame, fix the exact minimal event-evoking lexical anchor as the trigger, distinguishing it from nearby reporting, confirmation, motion, sentiment, or contextual words; this locked anchor is final.",
            "Step 3, schema-driven near-neighbor arbitration over locked frames: identify from the provided candidate event types the most confusable near-neighbor types (same coarse family or prefix, or semantically adjacent) and give a local wording or schema reason for choosing the target type over each, only relabeling the type of the already-locked trigger and never moving or dropping the anchor.",
            "Step 4, granularity check: if the target is a coarse type, state why no finer candidate subtype is locally supported; if it is a finer subtype, state why the coarser candidate would be too broad.",
            "Step 5, local argument attachment: attach target arguments using local evidence, but do not add a final pruning pass that deletes weak plausible roles.",
            "Step 6, final frame/trigger/type check: verify every locked trigger anchor is preserved with its exact minimal span, no supported frame is missing, no generic type replaces a finer supported type, and no wrong trigger anchor is used.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 8 to 12 natural sentences.",
            "It must lock each retained frame's minimal trigger anchor before doing type arbitration.",
            "It must state that type arbitration only relabels the type and never moves or drops a locked trigger anchor.",
            "It must include at least one explicit contrast between the kept event type and a plausible rejected near-neighbor type drawn from the provided candidate types.",
            "It must derive the competing types from the candidate schema for this input, not from a fixed hard-coded type list.",
            "It must include a granularity statement whenever the candidate types contain a coarse type and a finer subtype of the same family.",
            "It must mention retained event type names and trigger texts.",
            "It must attach target arguments from local evidence but must not include a separate final argument-pruning or weak-role deletion step.",
            "It must end with a final check that confirms every locked trigger anchor is preserved and covers missing frames, wrong type, and wrong trigger.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "trigger_anchor_lock",
            "schema_driven_arbitration",
            "near_neighbor_contrast",
            "generic_type_suppression",
            "trigger_anchor_exactness",
            "local_argument_attachment",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use recall-first candidate frame audit, lock minimal trigger anchors before type decisions, schema-driven near-neighbor type arbitration "
            "that only relabels locked frames, granularity control between coarse and finer candidate types, and local argument attachment without "
            "final weak-role pruning before surface-only evidence JSON"
        ),
        "placeholder": (
            "Audit plausible frames first, lock each minimal trigger anchor, contrast the target type against its most confusable near-neighbor candidate "
            "types from the provided schema while keeping anchors fixed, control granularity, attach locally supported target arguments without a final "
            "pruning pass, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "170-380",
        "min_words": 125,
        "min_sentences": 6,
    },
    "e84": {
        "name": "trigger_locked_no_arbitration",
        "task": "Generate trigger-locked no-arbitration natural-language CoT supervision for event extraction.",
        "goal": (
            "Ablation of E83 that keeps recall-first candidate coverage, trigger-anchor lock, and local argument attachment "
            "without pruning, but REMOVES contrastive type arbitration: assign each locked frame's event type directly from local "
            "wording and schema fit, without naming or contrasting competing candidate types. Used to test on a second dataset "
            "whether type arbitration is the engine of unseen-type generalization."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, trigger anchor lock: for each retained frame, fix the exact minimal event-evoking lexical anchor as the trigger; this locked anchor is final.",
            "Step 3, direct type assignment: assign the target event type to each locked frame from local wording and schema fit, without explicitly comparing against or naming rejected competing candidate types.",
            "Step 4, local argument attachment: attach target arguments using local evidence, but do not add a final pruning pass that deletes weak plausible roles.",
            "Step 5, final frame/trigger/type check: verify every locked trigger anchor is preserved with its exact minimal span, no supported frame is missing, and no wrong trigger anchor is used.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 7 to 11 natural sentences.",
            "It must lock each retained frame's minimal trigger anchor before assigning its type.",
            "It must assign the target type directly from local wording and must NOT explicitly compare or name rejected competing candidate types.",
            "It must mention retained event type names and trigger texts.",
            "It must attach target arguments from local evidence but must not include a separate final argument-pruning step.",
            "It must end with a final check that confirms every locked trigger anchor is preserved and covers missing frames and wrong trigger.",
        ],
        "extra_scores": [
            "candidate_audit_explicitness",
            "candidate_coverage",
            "trigger_anchor_lock",
            "direct_type_assignment",
            "trigger_anchor_exactness",
            "local_argument_attachment",
            "exact_event_consistency",
        ],
        "instruction_focus": (
            "use recall-first candidate frame audit, lock minimal trigger anchors before type decisions, assign event types directly "
            "from local wording without contrastive arbitration, and local argument attachment without final weak-role pruning before surface-only evidence JSON"
        ),
        "placeholder": (
            "Audit plausible frames first, lock each minimal trigger anchor, assign the target event type directly from local wording without contrasting candidates, "
            "attach locally supported target arguments without a final pruning pass, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "150-330",
        "min_words": 110,
        "min_sentences": 5,
    },
    "e71a": {
        "name": "event_frame_first_light_grounding",
        "task": "Generate event-frame-first natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach the extractor to preserve candidate-audit event-frame decisions before doing argument grounding. "
            "The reasoning should first lock retained event triggers and event types using E57-style candidate audit, "
            "then add light local argument grounding without letting argument uncertainty delete valid event frames."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, event-frame inventory: scan plausible event mentions in text order and decide retained versus rejected event frames before discussing arguments.",
            "Step 2, trigger and type lock: for each retained frame, identify the minimal trigger text and schema event type, contrasting close types only when needed.",
            "Step 3, event-frame preservation: make clear that argument sparsity or uncertainty should not remove a locally supported trigger/type frame.",
            "Step 4, light argument grounding: attach only arguments that are locally tied to the locked trigger and reject background participants.",
            "Step 5, final check: verify missing, duplicate, and extra event frames first, then verify minimal triggers and local evidence.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 7 to 10 natural sentences.",
            "It must discuss event frames before arguments.",
            "It must mention retained event type names and trigger texts.",
            "It must include a kept/rejected candidate audit for important candidates.",
            "It must state that argument grounding refines a retained event frame rather than deciding whether the event exists.",
            "It must keep argument grounding light and local, without over-focusing on evidence at the cost of event recall.",
        ],
        "extra_scores": [
            "event_frame_first_order",
            "event_frame_preservation",
            "candidate_audit_explicitness",
            "candidate_coverage",
            "argument_grounding_after_frame_lock",
            "final_event_inventory_check",
        ],
        "instruction_focus": (
            "first lock event frames with candidate audit, minimal triggers, and schema type decisions; then lightly ground local arguments "
            "without letting argument uncertainty delete supported events"
        ),
        "placeholder": (
            "Audit plausible event mentions in text order, lock retained trigger/type frames first, then lightly ground local arguments, "
            "preserve supported events, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "140-300",
        "min_words": 100,
        "min_sentences": 5,
    },
    "e71b": {
        "name": "event_frame_first_argument_module",
        "task": "Generate event-frame-first plus argument-module natural-language CoT supervision for event extraction.",
        "goal": (
            "Teach the extractor to combine E57-style event-frame stability with E70-style argument grounding. "
            "The reasoning should first lock event frames, then run a separate argument module for role minimality, role abstention, and copied local evidence."
        ),
        "thinking_strategy": [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Module 1, event-frame audit: scan plausible event mentions in text order and decide retained versus rejected frames using schema cues.",
            "Module 2, frame lock: for retained frames, fix the minimal trigger and event type before any argument discussion.",
            "Module 3, close-type control: use fine-grained type arbitration only to choose among event types, not to discard a supported frame for lack of arguments.",
            "Module 4, argument grounding: after the frame is locked, attach minimal local arguments and reject background, causal, document-topic, or world-knowledge participants.",
            "Module 5, evidence and final consistency: choose short exact local evidence for triggers and arguments, then check missing, duplicate, and extra events.",
        ],
        "quality_requirements": [
            "The thinking should usually contain 8 to 11 natural sentences.",
            "It must clearly separate event-frame audit from argument grounding.",
            "It must mention retained event type names and trigger texts before mentioning roles.",
            "It must include role minimality and role abstention after event frames are locked.",
            "It must not let argument minimality suppress a valid retained event frame.",
            "It must end with a final inventory check over missing, duplicate, and extra events.",
        ],
        "extra_scores": [
            "event_frame_first_order",
            "event_frame_preservation",
            "candidate_audit_explicitness",
            "candidate_coverage",
            "fine_grained_type_arbitration",
            "argument_minimality",
            "argument_grounding_after_frame_lock",
            "role_abstention",
            "final_event_inventory_check",
        ],
        "instruction_focus": (
            "lock event frames before roles, then run a separate argument-grounding module with minimal local roles, role abstention, copied evidence, "
            "and a final missing/duplicate/extra event check"
        ),
        "placeholder": (
            "First audit and lock event frames, then run a separate argument-grounding module for minimal local roles and evidence, "
            "preserve supported events, and output surface-only evidence JSON without numeric offsets."
        ),
        "word_range": "160-340",
        "min_words": 115,
        "min_sentences": 6,
    },
}


def variant_key_from_name(name: str | None) -> str:
    name = name or ""
    aliases = {
        "e48a": "e47a",
        "e48b": "e47b",
        "e51": "e47b",
        "e52": "e47b",
        "e53": "e47b",
        "e54": "e47b",
        "e55": "e47b",
        "e56": "e47b",
        "e70": "e70",
        "e71a": "e71a",
        "e71b": "e71b",
        "e72": "e72",
        "e73": "e73",
        "e76": "e76",
        "e80a": "e80a",
        "e80b": "e80b",
        "e81": "e81",
        "e82": "e82",
        "e83": "e83",
        "e84": "e84",
        "e95": "e83",
        "e111": "e83",
        "e121": "e83",
        "e122": "e83",
        "e123": "e83",
        "e124": "e83",
        "e125": "e83",
        "e126": "e83",
        "e127": "e83",
        "e130": "e81",
    }
    for prefix, key in aliases.items():
        if name.startswith(prefix):
            return key
    for key in VARIANTS:
        if name.startswith(key):
            return key
    return "e47b"


def variant_from_row(row: dict) -> dict:
    sample_id = row.get("meta", {}).get("e40_sample_id", "")
    return VARIANTS[variant_key_from_name(sample_id)]


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def sentence_count(text: str) -> int:
    return len([x for x in re.split(r"[.!?]+\s*", text or "") if x.strip()])


def generator_prompt(row: dict, prompt_profile: str = "standard", output_protocol: str = "xml_tags") -> str:
    candidates, schema_cards = e40.extract_schema(row["input"])
    variant = variant_from_row(row)
    json_wrapper = output_protocol == "json_wrapper"
    payload = {
        "task": variant["task"],
        "goal": (
            variant["goal"]
            + " The strategy should be generic across models, not tailored to a specific backbone."
        ),
        "important_positioning": [
            "The provided target events are authoritative for this supervision example.",
            "Do not add, remove, reorder, or modify target event types, trigger text, argument text, or roles.",
            "Do not mention model names, model errors, Direct, E40, E45, E46, E47, or case labels in the output.",
            "Do not write case-specific rules. Use general extraction principles that apply broadly to event extraction.",
        ],
        "required_output": (
            "Return one strict JSON object only with keys `thinking` and `final`. "
            "`thinking` is a natural-language string. `final` is the surface-only event JSON object."
            if json_wrapper
            else "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        ),
        "variant": variant["name"],
        "thinking_strategy": variant["thinking_strategy"],
        "thinking_quality_requirements": variant["quality_requirements"],
        "json_wrapper_schema": (
            {
                "thinking": "natural-language reasoning string",
                "final": {
                    "events": [
                        {
                            "event_type": "copy from target",
                            "trigger": {
                                "text": "copy from target",
                                "evidence": "short contiguous local phrase/clause from Text containing trigger text",
                            },
                            "arguments": [
                                {
                                    "role": "copy from target",
                                    "text": "copy from target",
                                    "evidence": "short contiguous local phrase/clause from Text containing argument text",
                                }
                            ],
                        }
                    ]
                },
            }
            if json_wrapper
            else None
        ),
        "final_json_schema": {
            "events": [
                {
                    "event_type": "copy from target",
                    "trigger": {"text": "copy from target", "evidence": "short contiguous local phrase/clause from Text containing trigger text"},
                    "arguments": [
                        {"role": "copy from target", "text": "copy from target", "evidence": "short contiguous local phrase/clause from Text containing argument text"}
                    ],
                }
            ]
        },
        "evidence_rules": [
            "Every evidence string must be an exact contiguous quote from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence should be a local phrase or short clause, not an isolated token when local context is available.",
            "Trigger evidence should show the event mention in context, such as 'she had a broken jaw', 'they met up', or '22 000 ended up in prison'.",
            "Argument evidence should show the argument's local relation to the event when possible, not only the argument token.",
            "Do not use a nearby context clause as evidence if that clause does not contain the corresponding text field.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor; keep the wider phrase only in trigger.evidence.",
        ],
        "target_style_calibration_rules": [
            "Candidate enumeration: scan all plausible event mentions in the text before deciding final events.",
            "Minimal trigger: prefer the shortest event-evoking lexical anchor, not the full verb phrase or clause.",
            "Close-type contrast: when a mention can map to several event types, explain the target schema choice using annotation-style cues.",
            "Local-role gate: an argument must be locally tied to the event mention, not merely a known participant in the broader story.",
            "No-extra-event gate: a semantically plausible situation is not enough; the text must contain an annotation-style event mention for the candidate schema.",
            "Final consistency: final surface strings must exactly match target events, and evidence must only help locate those strings.",
        ],
        "constraints": (
            [
                "The first non-whitespace character of your answer must be `{` and the last non-whitespace character must be `}`.",
                "Return valid JSON only: no markdown fences, no headings, no XML tags, no prose before or after the JSON object.",
                "The top-level object must have exactly two keys: `thinking` and `final`.",
                "`thinking` must be a JSON string, not a list and not an object.",
                "`final` must be a JSON object following final_json_schema.",
                "No numeric offsets, token indices, character positions, markdown, or text outside the JSON object.",
                f"Keep thinking natural, concise, and general; aim for {variant['word_range']} English words.",
                "Avoid explicitly saying that you are copying gold labels or that only one event is labeled; phrase decisions as target extraction style, schema alignment, and evidence grounding.",
            ]
            if json_wrapper
            else [
                "The first characters of your answer must be <thinking>.",
                "The last characters of your answer must be </final>.",
                "No numeric offsets, token indices, character positions, markdown, or text outside the two lowercase tags.",
                f"Keep thinking natural, concise, and general; aim for {variant['word_range']} English words.",
                "Avoid explicitly saying that you are copying gold labels or that only one event is labeled; phrase decisions as target extraction style, schema alignment, and evidence grounding.",
            ]
        ),
        "input": {
            "text": e40.extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "target_surface_events_to_copy": e40.surface_gold_json(row),
        },
    }
    if prompt_profile in {"strict_evidence", "json_acceptance_v2"}:
        payload["strict_evidence_protocol"] = [
            "Before writing the final JSON, silently verify every evidence value with a substring check against Text.",
            "If an evidence phrase is not a contiguous substring of Text, replace it before output.",
            "If the evidence does not contain the exact trigger.text or argument.text, replace it before output.",
            "Do not use ellipses, bracket insertions, normalized names, paraphrases, corrected grammar, or compressed wording in evidence.",
            "When unsure about the shortest valid evidence, copy the shortest source sentence fragment that contains the exact surface text and remains under the evidence length limit.",
            "For trigger.evidence, prefer a short fragment containing the trigger and its local event cue.",
            "For argument.evidence, prefer a short fragment containing the argument and the local relation to the trigger.",
            "Never let a broader evidence phrase become trigger.text; trigger.text must remain exactly copied from the target.",
        ]
        payload["final_self_check_before_output"] = [
            (
                "The answer is strict JSON only with exactly top-level keys `thinking` and `final`."
                if json_wrapper
                else "The answer begins with <thinking> and ends with </final>."
            ),
            "The final JSON has exactly the target events and no extra events.",
            "Every text field is copied from the target.",
            "Every evidence field is copied exactly from Text and contains its text field.",
            "No evidence field is a paraphrase or a non-contiguous quote.",
        ]
    if prompt_profile == "json_acceptance_v2":
        payload["single_json_object_contract"] = [
            "Return exactly one JSON object. Do not output a second JSON object, a corrected copy, a draft, or explanatory text.",
            "The top-level object must have exactly these two keys and no others: `thinking`, `final`.",
            "Do not put `events` at the top level; `events` must appear only inside `final`.",
            "Use normal JSON double quotes and escape internal quotes inside the thinking string.",
            "If you are tempted to add a note, self-check, or alternative answer, do it silently and do not output it.",
        ]
        payload["thinking_acceptance_requirements"] = [
            "Write 6 to 9 complete natural-language sentences, normally 120 to 240 English words.",
            "Mention every retained event type and trigger text.",
            "For each retained trigger, explicitly state why the final trigger.text is the minimal lexical anchor and why broader context belongs only in evidence.",
            "For each retained argument role, state the local textual relation to the trigger; do not justify roles from world knowledge or document-level context.",
            "When a plausible candidate is rejected, state a general schema/evidence reason in one sentence.",
            "Do not use bullet points, numbering, or a checklist style inside thinking.",
        ]
        payload["evidence_acceptance_requirements"] = [
            "Use the shortest contiguous source phrase or clause that contains the exact trigger/argument text and shows the local relation.",
            "Keep trigger.evidence at or under 18 whitespace-separated words unless the source punctuation makes a shorter exact quote impossible.",
            "Keep argument.evidence at or under 20 whitespace-separated words unless the source punctuation makes a shorter exact quote impossible.",
            "Never combine non-adjacent text with ellipses; if two useful clues are separated, choose the local fragment containing the required surface text.",
            "For repeated names or pronouns, choose the occurrence closest to the relevant trigger that still contains the exact argument text.",
            "For multi-event examples, choose separate local evidence for each event and role; do not reuse a broad sentence fragment when a local clause is available.",
        ]
        payload["final_preoutput_gate"] = [
            "Silently parse your own output as JSON before returning it.",
            "Silently check that final.events exactly matches target_surface_events_to_copy except for added evidence fields.",
            "Silently check that every evidence string is copied exactly from Text and contains its corresponding text field.",
            "Silently check that thinking is long enough and includes minimal-trigger separation and local argument-role grounding.",
        ]
    if prompt_profile == "e70_candidate_audit_v2":
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["thinking_strategy"] = VARIANTS["e70"]["thinking_strategy"]
        payload["thinking_quality_requirements"] = VARIANTS["e70"]["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Candidate inventory: scan all plausible event mentions in text order before deciding final events.",
            "Kept/rejected audit: explicitly say why each important candidate is retained or rejected.",
            "Generic-type suppression: do not choose a broad event type when local wording supports a more specific target schema type.",
            "Fine-grained type arbitration: when close schema types are possible, decide using annotation-style boundary cues, not world knowledge alone.",
            "Argument minimality: include only roles locally anchored to the retained trigger and required by the target event frame.",
            "Role abstention: reject participants that are only background, causal context, document topic, or generally associated with the event.",
            "Minimal trigger: keep trigger.text as the shortest copied event-evoking anchor; use evidence for the wider local phrase.",
            "Final inventory check: verify that the final JSON has no missing target event, no duplicate frame, and no extra plausible-but-unannotated event.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence should be a local phrase or short clause that helps locate the event or role relation.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the five moves: candidate audit, generic-type suppression, fine-grained type arbitration, argument minimality, and final consistency.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e72_e57_backbone_subtype_minarg":
        e72_variant = VARIANTS["e72"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e72_variant["name"]
        payload["task"] = e72_variant["task"]
        payload["goal"] = (
            e72_variant["goal"]
            + " The strategy should be generic across models, not tailored to a specific backbone."
        )
        payload["thinking_strategy"] = e72_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e72_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Preserve the E57 candidate-audit backbone: cover plausible event mentions in text order and decide kept versus rejected candidates.",
            "Subtype arbitration first: for communication/contact candidates, distinguish public broadcast, directed correspondence, meeting, and generic contact using local wording and schema cards.",
            "Generic-type suppression: use Contact:Contact or another broad type only when no finer candidate type is supported.",
            "Annotation-minimal argument policy: include only explicit local roles that target-style annotation would keep; do not add participants from world knowledge, broad story context, or semantic plausibility alone.",
            "Extra-role gate: before final output, ask whether each role would change the exact event tuple without clear local evidence; omit weak roles.",
            "Minimal trigger: explicitly say in thinking that trigger.text is the shortest copied event-evoking anchor and evidence is the wider local phrase.",
            "Final exact-event check: verify no missing target event, duplicate frame, wrong subtype, or extra argument remains.",
        ]
        payload["contact_subtype_arbitration_guide"] = [
            "Contact:Broadcast: public or one-to-many communication such as announcing, publishing, posting, reporting, or broadcasting.",
            "Contact:Correspondence: directed message or exchange between specific parties, including saying, telling, suggesting, emailing, calling, or writing when it is not a meeting.",
            "Contact:Meet: in-person or scheduled encounter where entities come together.",
            "Contact:Contact: use only as a fallback when the text supports contact but none of the finer candidate contact subtypes is locally supported.",
        ]
        payload["annotation_minimal_argument_guide"] = [
            "A role is kept only when the local text explicitly ties that participant to the retained trigger.",
            "Do not infer Agent, Communicator, Defendant, Victim, Entity, or Place only because they are plausible in the broader situation.",
            "Prefer a sparse gold-style argument set over a semantically richer set that adds weak roles.",
            "If a participant is merely background, document topic, causal context, undercover context, or a quoted hypothetical participant, reject it in thinking and omit it from final.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "For one-word arguments, include the nearest local words that show the relation to the trigger, while still copying one contiguous substring from Text.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking preserves candidate audit and adds subtype arbitration plus annotation-minimal argument checks.",
            "The thinking explicitly separates minimal trigger.text from broader local evidence.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No evidence field is an isolated token when a short local phrase containing that token exists in Text.",
            "No generic Contact type is chosen when a finer target Contact subtype is locally supported.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e73_e57_recall_first_exactness_last":
        e73_variant = VARIANTS["e73"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e73_variant["name"]
        payload["task"] = e73_variant["task"]
        payload["goal"] = (
            e73_variant["goal"]
            + " The strategy should be generic across models, not tailored to a specific backbone."
        )
        payload["thinking_strategy"] = e73_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e73_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate inventory: scan all plausible event mentions in text order and retain locally supported target-style frames before any subtype or argument pruning.",
            "Frame preservation: never remove a retained trigger/type frame only because its argument set is sparse, uncertain, or later needs cleanup.",
            "Subtype after retention: once a Contact or communication frame is retained, choose Broadcast, Correspondence, Meet, or generic Contact using local wording and schema cards.",
            "Generic suppression without recall suppression: prefer a finer subtype when locally supported, but do not use subtype uncertainty to drop a supported event mention.",
            "Exactness-last argument policy: after the frame inventory is fixed, omit only weak roles that are not explicitly and locally tied to the retained trigger.",
            "Minimal trigger: explicitly say that trigger.text is the shortest copied event-evoking anchor and evidence is a wider contiguous local phrase or clause.",
            "Final inventory check: first verify no supported target-style frame is missing, then verify no duplicate frame, extra frame, wrong subtype, or weak extra argument remains.",
        ]
        payload["contact_subtype_arbitration_guide"] = [
            "Contact:Broadcast: public or one-to-many communication such as announcing, publishing, posting, reporting, or broadcasting.",
            "Contact:Correspondence: directed message or exchange between specific parties, including saying, telling, suggesting, emailing, calling, or writing when it is not a meeting.",
            "Contact:Meet: in-person or scheduled encounter where entities come together.",
            "Contact:Contact: use only as a fallback after retaining the contact frame when no finer candidate contact subtype is locally supported.",
        ]
        payload["annotation_minimal_argument_final_check"] = [
            "Apply this check only after retained event frames are fixed.",
            "A role is kept only when local text explicitly ties that participant to the retained trigger.",
            "Omit participants that are merely plausible from world knowledge, broad story context, document topic, or causal background.",
            "Prefer a sparse target-style argument set over a semantically richer set that adds weak roles.",
            "Do not delete a retained event frame just because all optional roles are weak or absent.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "For one-word arguments, include the nearest local words that show the relation to the trigger, while still copying one contiguous substring from Text.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows recall-first order: candidate inventory, frame preservation, subtype after retention, exactness-last arguments, and final inventory check.",
            "The thinking explicitly separates minimal trigger.text from broader local evidence.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No supported event frame is deleted because arguments are sparse or because subtype arbitration is difficult.",
            "No generic Contact type is chosen when a finer retained Contact subtype is locally supported.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e76_contrastive_exactness":
        e76_variant = VARIANTS["e76"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e76_variant["name"]
        payload["task"] = e76_variant["task"]
        payload["goal"] = (
            e76_variant["goal"]
            + " The strategy should be generic across models, not tailored to a specific backbone."
        )
        payload["thinking_strategy"] = e76_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e76_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate frame audit: scan all plausible event mentions in text order before pruning roles or resolving difficult type boundaries.",
            "Contrastive type arbitration: for every close type choice, name the plausible rejected type and give a local wording/schema reason for rejecting it.",
            "Contact contrast is mandatory when Contact appears in candidates or targets: compare Contact:Contact, Contact:Broadcast, Contact:Correspondence, and Contact:Meet; use Contact:Contact only as a fallback when no finer subtype is locally supported.",
            "Trigger anchor exactness: choose the target trigger as the shortest event-evoking lexical anchor, not a nearby reporting, confirmation, motion, sentiment, or contextual word.",
            "Exactness-last argument pruning: after frames are fixed, keep only arguments explicitly tied to the retained trigger by local wording; omit plausible but non-annotated participants, places, artifacts, and discourse entities.",
            "No-extra-frame gate: reject semantically plausible events when the text does not contain an annotation-style event mention for that schema.",
            "Final event-tuple check: verify no missing target-style frame, no generic type replacing a finer supported type, no wrong trigger anchor, and no extra argument that would change exact Event F1.",
        ]
        payload["contact_subtype_contrast_guide"] = [
            "Contact:Broadcast: public or one-to-many communication, such as announcing, posting, publishing, reporting, broadcasting, or public statements.",
            "Contact:Correspondence: directed communication or exchange between specific parties, including email, message, call, letter, writing, telling, or replying when it is not an in-person meeting.",
            "Contact:Meet: in-person, scheduled, or physical encounter where entities come together.",
            "Contact:Contact: generic fallback only when the text supports communication/contact but no finer candidate subtype is locally supported.",
            "If the target is a finer Contact subtype, explicitly state why generic Contact:Contact would be too broad.",
            "If the target is Contact:Contact, explicitly state why Broadcast, Correspondence, and Meet are not locally supported.",
        ]
        payload["argument_exactness_contrast_guide"] = [
            "For each retained event, distinguish annotation-valid roles from plausible background participants.",
            "Reject entities that are only document topic, location of unrelated context, causal background, object of another event, quoted hypothetical participant, or world-knowledge participant.",
            "Do not add a Place, Agent, Entity, Artifact, Victim, Defendant, or Adjudicator unless local wording ties it to the retained trigger.",
            "Prefer a sparse exact event tuple over a semantically richer tuple with weak extra roles.",
            "When the target argument set is sparse, explain why additional plausible roles are omitted rather than inventing them.",
        ]
        payload["trigger_anchor_contrast_guide"] = [
            "Prefer the lexical event anchor over nearby support verbs such as said, confirmed, reported, went, took, had, made, or happened unless that support verb is the target trigger.",
            "If a nearby word describes evidence or context rather than the event itself, mention that it belongs in evidence or reasoning, not trigger.text.",
            "For nominal or adjectival event mentions, keep the target lexical anchor and do not shift to a nearby auxiliary or reporting word.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "For one-word arguments, include the nearest local words that show the relation to the trigger, while still copying one contiguous substring from Text.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the required order: candidate frame recall, contrastive type arbitration, Contact subtype contrast when relevant, trigger anchor exactness, exactness-last argument pruning, final event-tuple check.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No generic Contact type replaces a finer target Contact subtype when local wording supports the finer type.",
            "No extra plausible role is added when it is not explicitly and locally tied to the retained trigger.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile in {"e80a_no_type_arbitration", "e80b_no_argument_pruning"}:
        e80_key = "e80a" if prompt_profile == "e80a_no_type_arbitration" else "e80b"
        e80_variant = VARIANTS[e80_key]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e80_variant["name"]
        payload["task"] = e80_variant["task"]
        payload["goal"] = (
            e80_variant["goal"]
            + " This is a controlled ablation of the E76 strategy, so only the named strategy component should be removed."
        )
        payload["thinking_strategy"] = e80_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e80_variant["quality_requirements"]
        if e80_key == "e80a":
            payload["target_style_calibration_rules"] = [
                "Recall-first candidate frame audit: scan plausible event mentions in text order before pruning roles.",
                "Direct type assignment: assign each retained frame to the target schema type using local wording and schema fit, without explicitly naming rejected competing types.",
                "No explicit type arbitration: do not write phrases like 'not X but Y', do not list competing Contact subtypes, and do not force Contact:Contact versus finer subtype comparison.",
                "Trigger anchor exactness: choose the target trigger as the shortest event-evoking lexical anchor, not a nearby reporting, confirmation, motion, sentiment, or contextual word.",
                "Exactness-last argument pruning: after frames are fixed, keep only arguments explicitly tied to the retained trigger by local wording; omit plausible but non-annotated participants, places, artifacts, and discourse entities.",
                "No-extra-frame gate: reject semantically plausible events when the text does not contain an annotation-style event mention for that schema.",
                "Final event-tuple check: verify no missing target-style frame, no wrong trigger anchor, and no extra argument that would change exact Event F1.",
            ]
            payload["ablation_do_not_include"] = [
                "Do not explicitly compare Contact:Contact, Contact:Broadcast, Contact:Correspondence, and Contact:Meet.",
                "Do not name a plausible rejected event type as a contrastive alternative.",
                "Do not explain why a generic Contact type is rejected in favor of a finer Contact subtype.",
            ]
            payload["argument_exactness_contrast_guide"] = [
                "For each retained event, distinguish annotation-valid roles from plausible background participants.",
                "Reject entities that are only document topic, location of unrelated context, causal background, object of another event, quoted hypothetical participant, or world-knowledge participant.",
                "Do not add a Place, Agent, Entity, Artifact, Victim, Defendant, or Adjudicator unless local wording ties it to the retained trigger.",
                "Prefer a sparse exact event tuple over a semantically richer tuple with weak extra roles.",
            ]
            final_order = "candidate frame recall, direct type assignment, trigger anchor exactness, exactness-last argument pruning, final event-tuple check"
        else:
            payload["target_style_calibration_rules"] = [
                "Recall-first candidate frame audit: scan all plausible event mentions in text order before resolving difficult type boundaries.",
                "Contrastive type arbitration: for every close type choice, name the plausible rejected type and give a local wording/schema reason for rejecting it.",
                "Contact contrast is mandatory when Contact appears in candidates or targets: compare Contact:Contact, Contact:Broadcast, Contact:Correspondence, and Contact:Meet; use Contact:Contact only as a fallback when no finer subtype is locally supported.",
                "Trigger anchor exactness: choose the target trigger as the shortest event-evoking lexical anchor, not a nearby reporting, confirmation, motion, sentiment, or contextual word.",
                "Local argument attachment: attach target arguments from local evidence, but do not add a final pass that prunes weak plausible roles.",
                "No-extra-frame gate: reject semantically plausible events when the text does not contain an annotation-style event mention for that schema.",
                "Final frame/type/trigger check: verify no missing target-style frame, no generic type replacing a finer supported type, and no wrong trigger anchor.",
            ]
            payload["contact_subtype_contrast_guide"] = [
                "Contact:Broadcast: public or one-to-many communication, such as announcing, posting, publishing, reporting, broadcasting, or public statements.",
                "Contact:Correspondence: directed communication or exchange between specific parties, including email, message, call, letter, writing, telling, or replying when it is not an in-person meeting.",
                "Contact:Meet: in-person, scheduled, or physical encounter where entities come together.",
                "Contact:Contact: generic fallback only when the text supports communication/contact but no finer candidate subtype is locally supported.",
                "If the target is a finer Contact subtype, explicitly state why generic Contact:Contact would be too broad.",
                "If the target is Contact:Contact, explicitly state why Broadcast, Correspondence, and Meet are not locally supported.",
            ]
            payload["ablation_do_not_include"] = [
                "Do not add a final argument-pruning step.",
                "Do not say weak extra roles are removed in a final pass.",
                "Do not make the final self-check focus on extra arguments.",
            ]
            final_order = "candidate frame recall, contrastive type arbitration, Contact subtype contrast when relevant, trigger anchor exactness, local argument attachment, final frame/type/trigger check"
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "For one-word arguments, include the nearest local words that show the relation to the trigger, while still copying one contiguous substring from Text.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            f"The thinking follows the required ablation order: {final_order}.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e84_trigger_locked_no_arbitration":
        e84_variant = VARIANTS["e84"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e84_variant["name"]
        payload["task"] = e84_variant["task"]
        payload["goal"] = (
            e84_variant["goal"]
            + " This is a controlled ablation, so only the named strategy component (type arbitration) is removed."
        )
        payload["thinking_strategy"] = e84_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e84_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate frame audit: scan all plausible event mentions in text order before pruning.",
            "Trigger anchor lock: for each retained frame, fix the exact minimal event-evoking lexical anchor as the trigger before any type decision; this locked anchor must not change later.",
            "Direct type assignment: assign the target event type to each locked frame from local wording and schema fit, without explicitly comparing against or naming rejected competing candidate types.",
            "Local argument attachment: attach target arguments from local evidence, but do not add a final pass that prunes weak plausible roles.",
            "No-extra-frame gate: reject semantically plausible events when the text does not contain an annotation-style event mention for that schema.",
            "Final frame/trigger/type check: verify every locked trigger anchor is preserved with its exact minimal span, no missing target-style frame, and no wrong trigger anchor.",
        ]
        payload["ablation_do_not_include"] = [
            "Do not name a plausible rejected event type as a contrastive alternative.",
            "Do not write phrases like 'not X but Y' that contrast competing candidate types.",
            "Do not derive or discuss near-neighbor competing types from the candidate set.",
        ]
        payload["trigger_lock_guide"] = [
            "Lock the trigger as the shortest event-evoking lexical anchor; do not include nearby reporting, confirmation, motion, sentiment, or contextual words in the locked span.",
            "Once locked, the trigger span is final: do not move, shrink, extend, or drop the locked anchor.",
            "Do not drop a locally supported retained frame because its type is hard to decide.",
            "For nominal or adjectival event mentions, lock the target lexical anchor and do not shift to a nearby auxiliary or reporting word.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "For one-word arguments, include the nearest local words that show the relation to the trigger, while still copying one contiguous substring from Text.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the required ablation order: candidate frame recall, trigger anchor lock, direct type assignment, local argument attachment, final frame/trigger/type check.",
            "No competing candidate type is named or contrasted (type arbitration is ablated).",
            "Every locked trigger anchor is preserved with its exact minimal span.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e83_trigger_locked_schema_driven":
        e83_variant = VARIANTS["e83"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e83_variant["name"]
        payload["task"] = e83_variant["task"]
        payload["goal"] = (
            e83_variant["goal"]
            + " The strategy should be generic across models and datasets, not tailored to a specific backbone or ontology."
        )
        payload["thinking_strategy"] = e83_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e83_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate frame audit: scan all plausible event mentions in text order before resolving difficult type boundaries.",
            "Trigger anchor lock: for each retained frame, fix the exact minimal event-evoking lexical anchor as the trigger before any type decision; this locked anchor must not change later.",
            "Schema-driven near-neighbor arbitration over locked frames: for every close type choice, pick the most confusable competing types from the candidate event types provided for this input, name them, and give a local wording or schema reason for rejecting each in favor of the target type, only relabeling the type of the already-locked trigger and never moving, shrinking, extending, or dropping the locked anchor or its frame.",
            "Granularity control: when the candidate types include both a coarse type and a finer subtype of the same family, state why the target granularity is correct.",
            "Do not rely on a fixed hard-coded list of event types or subtypes; derive every contrast from the candidate schema given for this input so the reasoning transfers across datasets.",
            "Local argument attachment: attach target arguments from local evidence, but do not add a final pass that prunes weak plausible roles.",
            "No-extra-frame gate: reject semantically plausible events when the text does not contain an annotation-style event mention for that schema.",
            "Final frame/trigger/type check: verify every locked trigger anchor is preserved with its exact minimal span, no missing target-style frame, no generic type replacing a finer supported type, and no wrong trigger anchor.",
        ]
        payload["trigger_lock_guide"] = [
            "Lock the trigger as the shortest event-evoking lexical anchor; do not include nearby reporting, confirmation, motion, sentiment, or contextual words in the locked span.",
            "Once locked, the trigger span is final: type arbitration changes only the event_type label, not the trigger text or its boundaries.",
            "Do not drop a locally supported retained frame because its type is hard to decide; resolve the type while keeping the locked trigger.",
            "For nominal or adjectival event mentions, lock the target lexical anchor and do not shift to a nearby auxiliary or reporting word.",
        ]
        payload["schema_arbitration_guide"] = [
            "Identify near-neighbor competing types by shared coarse family or prefix (the part before ':') and by semantic adjacency among the provided candidates.",
            "For each retained frame, explicitly contrast the target type against at least one such near-neighbor candidate when one exists.",
            "Justify the choice with local wording from the Text, not with world knowledge or schema description text.",
            "If only one plausible candidate type fits, state briefly why the other candidate types do not match the local event mention.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "For one-word arguments, include the nearest local words that show the relation to the trigger, while still copying one contiguous substring from Text.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the required order: candidate frame recall, trigger anchor lock, schema-driven near-neighbor arbitration over locked frames, granularity control, local argument attachment, final frame/trigger/type check.",
            "Every locked trigger anchor is preserved with its exact minimal span and was not moved or dropped during type arbitration.",
            "Every type contrast is drawn from the candidate event types provided for this input, not from a fixed hard-coded type list.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No generic type replaces a finer target subtype when local wording supports the finer type.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e82_schema_driven_arbitration":
        e82_variant = VARIANTS["e82"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e82_variant["name"]
        payload["task"] = e82_variant["task"]
        payload["goal"] = (
            e82_variant["goal"]
            + " The strategy should be generic across models and datasets, not tailored to a specific backbone or ontology."
        )
        payload["thinking_strategy"] = e82_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e82_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate frame audit: scan all plausible event mentions in text order before resolving difficult type boundaries.",
            "Schema-driven near-neighbor arbitration: for every close type choice, pick the most confusable competing types from the candidate event types provided for this input, name them, and give a local wording or schema reason for rejecting each in favor of the target type.",
            "Granularity control: when the candidate types include both a coarse type and a finer subtype of the same family, state why the target granularity is correct (no finer candidate subtype is locally supported, or the coarser candidate is too broad).",
            "Do not rely on a fixed hard-coded list of event types or subtypes; derive every contrast from the candidate schema given for this input so the reasoning transfers across datasets.",
            "Trigger anchor exactness: choose the target trigger as the shortest event-evoking lexical anchor, not a nearby reporting, confirmation, motion, sentiment, or contextual word.",
            "Local argument attachment: attach target arguments from local evidence, but do not add a final pass that prunes weak plausible roles.",
            "No-extra-frame gate: reject semantically plausible events when the text does not contain an annotation-style event mention for that schema.",
            "Final frame/type/trigger check: verify no missing target-style frame, no generic type replacing a finer supported type, and no wrong trigger anchor.",
        ]
        payload["schema_arbitration_guide"] = [
            "Identify near-neighbor competing types by shared coarse family or prefix (the part before ':') and by semantic adjacency among the provided candidates.",
            "For each retained frame, explicitly contrast the target type against at least one such near-neighbor candidate when one exists.",
            "Justify the choice with local wording from the Text, not with world knowledge or schema description text.",
            "If only one plausible candidate type fits, state briefly why the other candidate types do not match the local event mention.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "For one-word arguments, include the nearest local words that show the relation to the trigger, while still copying one contiguous substring from Text.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the required order: candidate frame recall, schema-driven near-neighbor arbitration, granularity control, trigger anchor exactness, local argument attachment, final frame/type/trigger check.",
            "Every type contrast is drawn from the candidate event types provided for this input, not from a fixed hard-coded type list.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No generic type replaces a finer target subtype when local wording supports the finer type.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e81_trigger_locked_arbitration":
        e81_variant = VARIANTS["e81"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e81_variant["name"]
        payload["task"] = e81_variant["task"]
        payload["goal"] = (
            e81_variant["goal"]
            + " The strategy should be generic across models, not tailored to a specific backbone."
        )
        payload["thinking_strategy"] = e81_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e81_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate frame audit: scan all plausible event mentions in text order before resolving difficult type boundaries.",
            "Trigger anchor lock: for each retained frame, fix the exact minimal event-evoking lexical anchor as the trigger before any type decision; this locked anchor must not change later.",
            "Contrastive type arbitration over locked frames: for every close type choice, name the plausible rejected type and give a local wording/schema reason for rejecting it, but only relabel the type of an already-locked trigger and never move, shrink, extend, or drop the locked anchor or its frame.",
            "Contact contrast is mandatory when Contact appears in candidates or targets: compare Contact:Contact, Contact:Broadcast, Contact:Correspondence, and Contact:Meet; use Contact:Contact only as a fallback when no finer subtype is locally supported.",
            "Local argument attachment: attach target arguments from local evidence, but do not add a final pass that prunes weak plausible roles.",
            "No-extra-frame gate: reject semantically plausible events when the text does not contain an annotation-style event mention for that schema.",
            "Final frame/trigger/type check: verify every locked trigger anchor is preserved with its exact minimal span, no missing target-style frame, no generic type replacing a finer supported type, and no wrong trigger anchor.",
        ]
        payload["trigger_lock_guide"] = [
            "Lock the trigger as the shortest event-evoking lexical anchor; do not include nearby reporting, confirmation, motion, sentiment, or contextual words in the locked span.",
            "Once locked, the trigger span is final: type arbitration changes only the event_type label, not the trigger text or its boundaries.",
            "Do not drop a locally supported retained frame because its type is hard to decide; resolve the type while keeping the locked trigger.",
            "For nominal or adjectival event mentions, lock the target lexical anchor and do not shift to a nearby auxiliary or reporting word.",
        ]
        payload["contact_subtype_contrast_guide"] = [
            "Contact:Broadcast: public or one-to-many communication, such as announcing, posting, publishing, reporting, broadcasting, or public statements.",
            "Contact:Correspondence: directed communication or exchange between specific parties, including email, message, call, letter, writing, telling, or replying when it is not an in-person meeting.",
            "Contact:Meet: in-person, scheduled, or physical encounter where entities come together.",
            "Contact:Contact: generic fallback only when the text supports communication/contact but no finer candidate subtype is locally supported.",
            "If the target is a finer Contact subtype, explicitly state why generic Contact:Contact would be too broad.",
            "If the target is Contact:Contact, explicitly state why Broadcast, Correspondence, and Meet are not locally supported.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "For one-word arguments, include the nearest local words that show the relation to the trigger, while still copying one contiguous substring from Text.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the required order: candidate frame recall, trigger anchor lock, contrastive type arbitration over locked frames, Contact subtype contrast when relevant, local argument attachment, final frame/trigger/type check.",
            "Every locked trigger anchor is preserved with its exact minimal span and was not moved or dropped during type arbitration.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No generic Contact type replaces a finer target Contact subtype when local wording supports the finer type.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e130_retrieved_abstention":
        target_mode = row.get("meta", {}).get("e130_target_mode", "gold_present")
        missing_types = row.get("meta", {}).get("e130_missing_gold_types", [])
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = "retrieved_candidate_abstention"
        payload["task"] = VARIANTS["e81"]["task"]
        payload["goal"] = (
            "Apply the trigger-locked SG-CoT strategy under retrieved candidate schemas. "
            "Never emit a type absent from the candidate set and never force a locally expressed "
            "event into the nearest wrong candidate. Preserve candidate-supported target events; "
            "when retrieval omitted the needed schema, explicitly abstain from that event."
        )
        payload["retrieved_support_contract"] = {
            "target_mode": target_mode,
            "omitted_gold_types_for_audit_only": missing_types,
            "rules": [
                "Use the omitted-type list only to explain why no supplied candidate supports that local mention; do not output those types in final.",
                "Final must contain exactly target_surface_events_to_copy and therefore only candidate-supported types.",
                "For abstain mode, explain why every supplied schema is unsupported and output an empty event list.",
                "For partial_supported mode, preserve supported frames and explicitly abstain from unsupported frames.",
                "For gold_present mode, perform ordinary trigger-locked SG-CoT without inventing an abstention.",
            ],
        }
        payload["thinking_strategy"] = [
            "Audit plausible local event mentions against only the supplied candidate schemas.",
            "For each supported frame, lock the minimal trigger anchor before type arbitration.",
            "Contrast close supplied types using local wording; arbitration may only relabel a locked frame.",
            "For an unsupported mention, state that no supplied schema licenses it and abstain rather than substitute a nearby type.",
            "Attach locally supported arguments only to retained candidate-supported frames.",
            "Check that final contains no type outside candidates and no unsupported substitute frame.",
        ]
        payload["thinking_quality_requirements"] = [
            "Write 6 to 11 complete natural sentences grounded in this input.",
            "Mention retained event types and locked triggers when target events exist.",
            "For partial_supported or abstain mode, explicitly explain schema omission and abstention.",
            "Never justify a nearest-type substitution for an omitted schema.",
            "Remain exactly consistent with target_surface_events_to_copy.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be an exact contiguous substring from Text.",
            "trigger.evidence must contain trigger.text and argument.evidence must contain argument.text.",
            "Use no evidence fields when final events is empty.",
            "Do not output numeric offsets or token indices.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "Every final event type occurs in the supplied candidate set.",
            "Final exactly matches target_surface_events_to_copy plus evidence fields.",
            "No unsupported mention was mapped to a nearby candidate type.",
            "No numeric offsets, markdown, headings, or text outside the required tags appear.",
        ]
    if prompt_profile == "e93_trigger_locked_genericsuppress":
        # e81 recipe with a hardened type-granularity discipline: the diagnosed dominant error on
        # unseen types is collapsing a specific event onto a familiar high-frequency generic type
        # (Contact:Broadcast/Correspondence -> Contact:Contact). Strengthen generic-suppression into
        # a hard, cue-grounded, anti-frequency rule so the prefer-finer disposition transfers to
        # subtypes never seen as training outputs.
        e81_variant = VARIANTS["e81"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e81_variant["name"]
        payload["task"] = e81_variant["task"]
        payload["goal"] = (
            e81_variant["goal"]
            + " Additionally, enforce a strict type-granularity discipline: a coarse or generic type is a last resort, "
            "chosen only when no finer candidate subtype's cues are locally present, never as a familiar or frequent default. "
            "The strategy should be generic across models, not tailored to a specific backbone."
        )
        payload["thinking_strategy"] = [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, trigger anchor lock: fix each retained frame's exact minimal event-evoking lexical anchor as the trigger; this locked anchor is final.",
            "Step 3, granularity-first type arbitration over locked frames: before naming a type, enumerate the candidate types that share the trigger's coarse family (same prefix before ':') and check each finer subtype's schema-card cues against the local wording; pick the FINEST subtype whose cues are locally present and only relabel the locked trigger's type.",
            "Step 4, generic last-resort gate: choose a coarse or generic type ONLY when no finer candidate subtype's cues are locally present; never select a type because it is more frequent, more familiar, or a safe default, and explicitly say which finer subtypes you rejected and why their cues are absent.",
            "Step 5, local argument attachment: attach target arguments from local evidence without a final weak-role pruning pass.",
            "Step 6, final check: verify each locked anchor is preserved, no supported frame is missing, and crucially that no generic/coarse type was used where a finer subtype's cue is locally present.",
        ]
        payload["thinking_quality_requirements"] = e81_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate frame audit: scan all plausible event mentions in text order before resolving difficult type boundaries.",
            "Trigger anchor lock: fix the exact minimal event-evoking lexical anchor before any type decision; the locked anchor must not change later.",
            "Granularity-first arbitration: for every locked trigger, enumerate same-family candidate subtypes and prefer the finest whose schema-card cues match the local wording; relabel only the type, never the locked anchor.",
            "Generic suppression is a hard rule, not a preference: a coarse/generic type (e.g. a bare family-level type) is admissible ONLY when none of the finer candidate subtypes' cues appear in the local wording.",
            "Anti-familiarity bias: never choose a type because it is common, familiar, or a safe fallback; choose strictly by local schema-card cue match, even when the supported finer subtype is one you would rarely produce.",
            "Local argument attachment without a final weak-role pruning pass.",
            "No-extra-frame gate: reject semantically plausible events when the text contains no annotation-style event mention for that schema.",
            "Final frame/trigger/type check: confirm no generic type replaced a finer supported subtype and no locked anchor was moved or dropped.",
        ]
        payload["trigger_lock_guide"] = e81_variant.get("trigger_lock_guide", [
            "Lock the trigger as the shortest event-evoking lexical anchor; exclude nearby reporting, confirmation, motion, sentiment, or contextual words.",
            "Once locked, the trigger span is final: type arbitration changes only the event_type label, not the trigger boundaries.",
            "Do not drop a locally supported frame because its type is hard to decide; resolve the type while keeping the locked trigger.",
        ])
        payload["type_granularity_discipline"] = [
            "The single most common error to prevent is collapsing a specific event onto a familiar generic or coarse type.",
            "For each locked trigger, list the candidate types sharing its coarse family and test each finer subtype's defining cue against the local wording.",
            "If any finer subtype's defining cue is locally present, you MUST select that finer subtype and explicitly reject the generic, even if the finer subtype is unusual or one you rarely emit.",
            "Use the generic/coarse type only as a documented last resort, stating that none of the finer subtypes' cues are present.",
            "Frequency, prior familiarity, and 'safe default' are never valid reasons to choose a type.",
        ]
        payload["contact_subtype_contrast_guide"] = [
            "Contact:Broadcast: public or one-to-many communication (announce, post, publish, report, broadcast, public statement).",
            "Contact:Correspondence: directed communication/exchange between specific parties (email, message, call, letter, write, tell, reply) that is not an in-person meeting.",
            "Contact:Meet: in-person, scheduled, or physical encounter where entities come together.",
            "Contact:Contact: generic fallback ONLY when communication is supported but no Broadcast/Correspondence/Meet cue is locally present.",
            "Hard rule: if a Broadcast, Correspondence, or Meet cue is present, you must choose that finer subtype and reject generic Contact:Contact; do not default to Contact:Contact because it is common.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 words for trigger.evidence and 3 to 20 for argument.evidence; longer only when punctuation or repetition requires it.",
            "Do not paraphrase, normalize, fix grammar, insert ellipses, or combine separated fragments; do not use schema text as evidence.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the required order: candidate frame recall, trigger anchor lock, granularity-first type arbitration, generic last-resort gate, local argument attachment, final check.",
            "Every locked trigger anchor is preserved with its exact minimal span and was not moved or dropped during type arbitration.",
            "No generic or coarse type was chosen where a finer candidate subtype's cue is locally present; any generic choice is explicitly justified by the absence of all finer-subtype cues.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e94_trigger_locked_arg_recall":
        # e81 recipe with an exhaustive argument-enumeration step. Diagnosis: on SEEN, E81's main
        # argument weakness is RECALL (it under-attaches ~31% of gold arguments on correctly-typed
        # triggers, more than Audit-CoT). Strengthen ONLY the argument-attachment step; leave the
        # trigger-lock + type arbitration machinery identical to e81.
        e81_variant = VARIANTS["e81"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e81_variant["name"]
        payload["task"] = e81_variant["task"]
        payload["goal"] = (
            e81_variant["goal"]
            + " Additionally, attach arguments exhaustively: for every retained frame, go through all roles in its schema card and attach every locally supported argument, since under-attachment is the dominant argument error. The strategy should be generic across models, not tailored to a specific backbone."
        )
        payload["thinking_strategy"] = [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, trigger anchor lock: fix each retained frame's exact minimal event-evoking lexical anchor as the trigger; this locked anchor is final.",
            "Step 3, contrastive type arbitration over locked frames: explicitly compare plausible competing types and choose the schema type supported by local wording, only relabeling the locked trigger's type.",
            "Step 4, Contact subtype contrast when relevant: compare Contact:Contact, Contact:Broadcast, Contact:Correspondence, Contact:Meet, using the generic only as a fallback.",
            "Step 5, exhaustive local argument attachment: for each locked frame, go through EVERY role defined in its schema card and attach every argument locally supported by the text; do not stop at the most salient one or two arguments and do not omit a clearly supported role; no final pruning pass that deletes weak plausible roles.",
            "Step 6, argument-completeness and final check: re-scan each frame's local context for any schema role not yet filled but locally supported, then verify every locked trigger anchor is preserved, no supported frame is missing, and no generic type replaces a finer supported type.",
        ]
        payload["thinking_quality_requirements"] = e81_variant["quality_requirements"] + [
            "For each retained frame it must consider all schema-card roles and attach every locally supported argument, not only the most obvious one or two.",
        ]
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate frame audit: scan all plausible event mentions in text order before resolving difficult type boundaries.",
            "Trigger anchor lock: fix the exact minimal event-evoking lexical anchor before any type decision; the locked anchor must not change later.",
            "Contrastive type arbitration over locked frames: name the plausible rejected type and give a local reason, relabeling only the locked trigger's type.",
            "Contact contrast when Contact appears: compare Contact:Contact, Broadcast, Correspondence, Meet; use generic only as a fallback.",
            "Exhaustive argument attachment: for each retained frame enumerate ALL schema-card roles and attach every locally supported argument; under-attachment (omitting a locally supported role) is the main argument error to avoid; still no final weak-role pruning pass.",
            "No-extra-frame gate: reject semantically plausible events when the text has no annotation-style event mention for that schema.",
            "Final frame/trigger/type/argument check: every locked anchor preserved, no missing frame, no generic-over-finer type, and no locally supported argument omitted.",
        ]
        payload["trigger_lock_guide"] = e81_variant.get("trigger_lock_guide", [
            "Lock the trigger as the shortest event-evoking lexical anchor; exclude nearby reporting, confirmation, motion, sentiment, or contextual words.",
            "Once locked, the trigger span is final: type arbitration changes only the event_type label, not the trigger boundaries.",
        ])
        payload["contact_subtype_contrast_guide"] = e81_variant.get("contact_subtype_contrast_guide", [])
        payload["argument_attachment_guide"] = [
            "Treat the schema card's role list as a checklist: for each role, look in the local context of the locked trigger for a supported filler before moving on.",
            "Attach every argument the text locally supports, including secondary participants, places, and times when present.",
            "Do not omit a role merely because it is less salient; omit only when the text gives no local support for it.",
            "Keep argument spans minimal and copied exactly from the text; do not invent or normalize fillers.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 words for trigger.evidence and 3 to 20 for argument.evidence; longer only when punctuation or repetition requires it.",
            "Do not paraphrase, normalize, fix grammar, insert ellipses, or combine separated fragments; do not use schema text as evidence.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the required order: candidate frame recall, trigger anchor lock, contrastive type arbitration, Contact contrast when relevant, exhaustive local argument attachment, argument-completeness and final check.",
            "Every locked trigger anchor is preserved with its exact minimal span and was not moved or dropped during type arbitration.",
            "For every retained frame, each schema-card role with local support has been attached; no clearly supported argument is omitted.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "e95_trigger_locked_autocluster":
        # Keep this profile ontology-agnostic: every compared type must come from the
        # per-input candidate schema and the schema-derived neighbor map.
        schema_variant = VARIANTS["e83"]
        global AUTO_CLUSTER_MAP_CACHE
        try:
            AUTO_CLUSTER_MAP_CACHE
        except NameError:
            AUTO_CLUSTER_MAP_CACHE = None
        if AUTO_CLUSTER_MAP_CACHE is None:
            import json as _json
            from pathlib import Path as _Path
            AUTO_CLUSTER_MAP_CACHE = _json.loads(_Path(AUTO_CLUSTER_MAP_PATH).read_text())["clusters"]
        _cands = list(candidates)
        _fam = lambda t: t.split(":", 1)[0]
        neighbor_lines = []
        for _c in _cands:
            _nbrs = [n for n in AUTO_CLUSTER_MAP_CACHE.get(_c, []) if n in _cands]
            if not _nbrs:
                _nbrs = [o for o in _cands if o != _c and _fam(o) == _fam(_c)][:3]
            if _nbrs:
                neighbor_lines.append(f"{_c}: contrast against {', '.join(_nbrs)}")
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = "trigger_locked_autocluster_arbitration"
        payload["task"] = schema_variant["task"]
        payload["goal"] = (
            "Produce ontology-agnostic event-extraction reasoning using only the candidate event types, "
            "schema cards, and target events supplied for this input. Recall plausible frames and lock "
            "their minimal trigger anchors before deciding types. Then contrast each retained frame only "
            "against the auto-derived confusable neighbors listed for candidates in this input, using local "
            "wording, trigger cues, role overlap, and schema granularity. Type arbitration may relabel a "
            "locked frame but must never move or drop its trigger anchor. Attach locally supported arguments "
            "and finish with a missing/extra frame, type, and trigger check. The procedure must remain generic "
            "across ontologies and model backbones."
        )
        payload["thinking_strategy"] = [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, candidate frame recall: scan plausible event mentions in text order and retain locally supported target-style frames before pruning.",
            "Step 2, trigger anchor lock: fix each retained frame's exact minimal event-evoking lexical anchor as the trigger; this locked anchor is final.",
            "Step 3, contrastive type arbitration over locked frames using the provided auto-derived confusable-neighbor sets: for each retained frame, contrast the target type against its listed neighbors present among the candidates, give a local wording/schema reason for rejecting each neighbor, and only relabel the type of the already-locked trigger.",
            "Step 4, granularity control: when a coarse type and a finer subtype of the same family are both candidates, state why the chosen granularity is locally supported; use the coarse type only as a fallback.",
            "Step 5, local argument attachment: attach target arguments using local evidence, but do not add a final pruning pass that deletes weak plausible roles.",
            "Step 6, final frame/trigger/type check: verify every locked trigger anchor is preserved with its exact minimal span, no supported frame is missing, no generic type replaces a finer supported type, and no wrong trigger anchor is used.",
        ]
        payload["thinking_quality_requirements"] = schema_variant["quality_requirements"]
        payload["auto_derived_confusable_neighbors"] = neighbor_lines
        payload["target_style_calibration_rules"] = [
            "Recall-first candidate frame audit: scan all plausible event mentions in text order before resolving difficult type boundaries.",
            "Trigger anchor lock: fix the exact minimal event-evoking lexical anchor before any type decision; the locked anchor must not change later.",
            "Contrastive type arbitration over locked frames: contrast the target type against its auto-derived confusable neighbors listed above, naming the rejected neighbors and giving a local wording/schema reason; only relabel the locked trigger's type and never move, shrink, extend, or drop the locked anchor or its frame.",
            "Granularity control: choose a coarse or family-generic type only when no finer neighbor is locally supported.",
            "Local argument attachment: attach target arguments from local evidence, but do not add a final pass that prunes weak plausible roles.",
            "No-extra-frame gate: reject semantically plausible events when the text does not contain an annotation-style event mention for that schema.",
            "Final frame/trigger/type check: verify every locked trigger anchor is preserved with its exact minimal span, no missing target-style frame, no generic type replacing a finer supported type, and no wrong trigger anchor.",
        ]
        payload["trigger_lock_guide"] = [
            "Lock the trigger as the shortest event-evoking lexical anchor; do not include nearby reporting, confirmation, motion, sentiment, or contextual words in the locked span.",
            "Once locked, the trigger span is final: type arbitration changes only the event_type label, not the trigger text or its boundaries.",
            "Do not drop a locally supported retained frame because its type is hard to decide; resolve the type while keeping the locked trigger.",
            "For nominal or adjectival event mentions, lock the target lexical anchor and do not shift to a nearby auxiliary or reporting word.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence must be a local phrase or short clause that helps locate the minimal trigger or the explicit role relation, not an isolated single token when nearby context is available.",
            "Prefer 3 to 18 whitespace-separated words for trigger.evidence and 3 to 20 words for argument.evidence; use a longer contiguous quote only when punctuation or repeated mentions require it.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not use schema text as evidence.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking follows the required order: candidate frame recall, trigger anchor lock, contrastive type arbitration over auto-derived neighbors, granularity control, local argument attachment, final frame/trigger/type check.",
            "Every locked trigger anchor is preserved with its exact minimal span and was not moved or dropped during type arbitration.",
            "Every type contrast names at least one auto-derived confusable neighbor when one is listed for the target type.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile in {"e71_event_frame_first_light", "e71_event_frame_first_argument"}:
        e71_key = "e71a" if prompt_profile == "e71_event_frame_first_light" else "e71b"
        e71_variant = VARIANTS[e71_key]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = e71_variant["name"]
        payload["task"] = e71_variant["task"]
        payload["goal"] = (
            e71_variant["goal"]
            + " The strategy should be generic across models, not tailored to a specific backbone."
        )
        payload["thinking_strategy"] = e71_variant["thinking_strategy"]
        payload["thinking_quality_requirements"] = e71_variant["quality_requirements"]
        payload["target_style_calibration_rules"] = [
            "Event-frame-first order: decide retained/rejected event frames before filling arguments.",
            "Candidate inventory: scan plausible event mentions in text order, including later or less salient cues.",
            "Frame lock: for each retained event, fix event_type and minimal trigger.text before discussing roles.",
            "Close-type control: use schema boundary cues to choose the event type; do not use argument sparsity as a reason to delete a supported frame.",
            "Argument grounding after frame lock: attach only locally tied arguments and abstain from background participants.",
            "Evidence support: use exact local evidence quotes to locate triggers and arguments, while keeping trigger.text minimal.",
            "Final inventory check: first check missing/duplicate/extra event frames, then check argument minimality and evidence containment.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "trigger.evidence must contain trigger.text exactly.",
            "argument.evidence must contain argument.text exactly.",
            "Evidence should be a local phrase or short clause that helps locate the locked event frame or role relation.",
            "Do not paraphrase, normalize wording, fix grammar, insert ellipses, or combine separated fragments.",
            "Do not copy the whole evidence phrase into trigger.text when the target trigger is a shorter lexical anchor.",
            "Do not let evidence selection or argument uncertainty change the target event inventory.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking locks event frames before argument grounding.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "xml_lean_v3":
        payload["required_output"] = "Return exactly two blocks: <thinking>natural language</thinking><final>{JSON}</final>. No text outside the two blocks."
        payload["thinking_strategy"] = [
            "Use one compact natural-language paragraph with three moves in this order.",
            "Move 1, candidate audit: cover plausible event mentions in text order and say which target-style mentions are kept or rejected.",
            "Move 2, copied evidence selection: for each kept trigger and important argument, refer to the exact local source phrase that will be copied as evidence.",
            "Move 3, final consistency: confirm minimal trigger text, local argument-role grounding, no extra events, and exact target surface strings.",
        ]
        payload["thinking_quality_requirements"] = [
            "Write 6 to 8 complete sentences, usually 110 to 230 English words.",
            "Mention retained event type names and trigger texts.",
            "Mention minimal-trigger separation when the evidence phrase is broader than trigger.text.",
            "Mention argument roles only when their local relation to the trigger is supported by copied source wording.",
            "Avoid bullet points, headings, numbered lists, and meta-comments.",
        ]
        payload["constraints"] = [
            "The first characters of the answer must be <thinking>.",
            "The last characters of the answer must be </final>.",
            "Use lowercase XML tags only: exactly one <thinking> block and exactly one <final> block.",
            "Do not output <think>, markdown, headings, explanations, or text outside the two blocks.",
            "Do not output numeric offsets, token indices, character positions, or span indices.",
            "Inside <final>, output JSON only.",
        ]
        payload["evidence_rules"] = [
            "Every evidence string must be copied exactly as one contiguous substring from Text.",
            "Every trigger.evidence must contain trigger.text exactly.",
            "Every argument.evidence must contain argument.text exactly.",
            "Prefer a local phrase or clause that proves the event or role relation; if exact copying is difficult, copy the shortest sentence fragment that contains the surface text.",
            "Do not paraphrase, normalize, fix grammar, insert ellipses, combine separated fragments, or copy text from the schema.",
            "Keep trigger.text and argument.text exactly copied from target_surface_events_to_copy; evidence may be wider, but text fields may not change.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Each evidence string is an exact substring of Text and contains the corresponding text field.",
            "The thinking mentions candidate audit, copied evidence selection, and final consistency.",
        ]
    if prompt_profile == "freeform_nl":
        # Review concern #1: a fair FREE-FORM natural-language CoT baseline. Same teacher, same
        # gold, same surface+evidence final format, but the <thinking> carries genuine prose
        # reasoning with NO task-decomposition scaffolding (no audit/lock/arbitration/granularity
        # steps, no numbered moves, no stage labels). Isolates structure vs. mere NL reasoning.
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = "freeform_natural_language_reasoning"
        payload["task"] = "Generate free-form natural-language reasoning for event extraction."
        payload["goal"] = (
            "Write a fluent, genuine natural-language explanation of the event extraction for this input, "
            "grounded in the text and the target events. Reason in your own words about why each event is "
            "evoked, why its event type is the right reading, and why each argument fills its role. "
            "Do NOT follow any fixed checklist, numbered procedure, or named decision steps; write coherent prose."
        )
        payload["thinking_strategy"] = [
            "Write one coherent natural-language explanation of the extraction, in fluent prose.",
            "Explain why each target event is evoked by the text, why its event type is the correct reading among the candidates, and why each argument plays its role.",
            "Refer to the relevant words in the text as you reason, but do not impose a fixed step order, numbered moves, or section labels.",
            "Do not name or enumerate decision stages (no 'candidate audit', 'trigger lock', 'type arbitration', 'granularity check', etc.); just reason naturally.",
        ]
        payload["thinking_quality_requirements"] = [
            "Write 6 to 12 complete natural sentences of genuine reasoning.",
            "Mention the target event type names, trigger texts, and argument roles.",
            "Reason in fluent prose, not as a template, checklist, or numbered list.",
            "Be fully consistent with the target events; do not add, drop, or relabel any event, trigger, argument, or role.",
            "Avoid headings, bullet points, numbered steps, and stage labels.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking is free-form prose without numbered steps or named decision stages.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if prompt_profile == "norecall_first":
        # Review concern #4: ablate ONLY the recall-first candidate audit from the main SG-CoT (e81).
        # Keep trigger-anchor lock + contrastive type arbitration + Contact contrast + local argument
        # attachment; remove the upfront recall-first scan (commit to confident events directly).
        e81v = VARIANTS["e81"]
        payload["required_output"] = "Return exactly <thinking>...</thinking> followed by <final>{JSON}</final>. Use lowercase tags only."
        payload["variant"] = "no_recall_first_trigger_locked_arbitration"
        payload["task"] = e81v["task"]
        payload["goal"] = (
            "Same trigger-locked contrastive type arbitration as the main strategy, but WITHOUT a recall-first "
            "candidate audit: do not scan for and retain all plausible event mentions before pruning. Instead, "
            "commit directly to the events you are confident about and resolve their type and arguments. "
            "The strategy should be generic across models, not tailored to a specific backbone."
        )
        payload["thinking_strategy"] = [
            "Write a substantive but concise reasoning paragraph, not a generic template.",
            "Step 1, trigger anchor exactness: for each event you commit to, fix the exact minimal event-evoking lexical anchor as the trigger before any type decision; this locked anchor must not change later.",
            "Step 2, contrastive type arbitration over locked frames: for each anchored frame, name the most confusable competing candidate type(s) and give a local-wording reason for choosing the target type over them; only relabel the locked trigger's type, never move or drop the anchor.",
            "Step 3, local argument attachment: attach target arguments using local evidence, without a final pruning pass.",
            "Step 4, final type/trigger check: verify no generic type replaces a finer supported type and every locked trigger anchor is preserved.",
            "Do NOT perform an upfront recall-first scan that enumerates and retains all plausible mentions before pruning; commit to events directly.",
        ]
        payload["thinking_quality_requirements"] = [
            "The thinking should usually contain 7 to 11 natural sentences.",
            "It must NOT include an upfront recall-first candidate-coverage audit; it commits to events directly.",
            "It must lock the minimal trigger anchor before deciding the type.",
            "It must include at least one explicit contrast between the kept event type and a plausible rejected near-neighbor type.",
            "It must mention retained event type names and trigger texts.",
            "It must attach target arguments from local evidence without a separate final pruning step.",
            "It must end with a final check over wrong type and wrong trigger.",
        ]
        payload["trigger_lock_guide"] = [
            "Lock the trigger as the shortest event-evoking lexical anchor; do not include nearby reporting, confirmation, motion, sentiment, or contextual words in the locked span.",
            "Once locked, the trigger span is final: type arbitration changes only the event_type label, not the trigger text or its boundaries.",
            "For nominal or adjectival event mentions, lock the target lexical anchor and do not shift to a nearby auxiliary or reporting word.",
        ]
        payload["contact_subtype_contrast_guide"] = [
            "Contact:Broadcast: public or one-to-many communication, such as announcing, posting, publishing, reporting, broadcasting, or public statements.",
            "Contact:Correspondence: directed communication or exchange between specific parties, including email, message, call, letter, writing, telling, or replying when it is not an in-person meeting.",
            "Contact:Meet: in-person, scheduled, or physical encounter where entities come together.",
            "Contact:Contact: generic fallback only when the text supports communication/contact but no finer candidate subtype is locally supported.",
        ]
        payload["final_self_check_before_output"] = [
            "The answer starts with <thinking> and ends with </final>.",
            "The thinking does NOT include an upfront recall-first candidate-coverage audit.",
            "Every locked trigger anchor is preserved with its exact minimal span and was not moved or dropped during type arbitration.",
            "The final JSON has exactly target_surface_events_to_copy plus evidence fields, with no added or removed events or arguments.",
            "Every evidence string is an exact substring of Text and contains its corresponding text field.",
            "No numeric offsets, token indices, markdown, headings, or text outside the two required lowercase tags appear in the answer.",
        ]
    if payload.get("json_wrapper_schema") is None:
        payload.pop("json_wrapper_schema", None)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def verifier_prompt(
    row: dict,
    thinking: str,
    final_obj: dict,
    verifier_profile: str = "strict_schema_labels",
) -> str:
    if verifier_profile not in {
        "strict_schema_labels",
        "target_role_alias_v1",
        "target_role_alias_core_reasoning_v1",
    }:
        raise ValueError(f"unknown verifier profile: {verifier_profile}")
    candidates, schema_cards = e40.extract_schema(row["input"])
    variant = variant_from_row(row)
    core_reasoning = verifier_profile == "target_role_alias_core_reasoning_v1"
    scores = {
        "type_discrimination": "integer 1-5",
        "trigger_boundary_control": "integer 1-5",
        "argument_role_grounding": "integer 1-5",
        "extraction_style_control": "integer 1-5",
        "candidate_coverage": "integer 1-5",
        "minimal_trigger_separation": "integer 1-5",
        "role_abstention": "integer 1-5",
        "no_extra_event_gate": "integer 1-5",
        "final_structure_consistency": "integer 1-5",
    }
    if not core_reasoning:
        scores = {
            "trigger_evidence": "integer 1-5",
            "argument_evidence": "integer 1-5",
            "evidence_informativeness": "integer 1-5",
            **scores,
        }
    for key in variant["extra_scores"]:
        scores[key] = "integer 1-5"
    payload = {
        "task": f"Strictly verify {variant['name']} CoT/evidence training data for event extraction.",
        "instruction": "Do not repair the answer. Return strict JSON only.",
        "pass_requirements": [
            "The final surface events exactly match the target surface events.",
            "The reasoning is task-grounded rather than a generic template.",
            "The reasoning follows the requested variant strategy.",
            "The reasoning explains schema grounding, close-type contrast when relevant, trigger boundary control, argument-role grounding, and extraction-style control.",
            "The reasoning separates minimal trigger.text from wider trigger.evidence when the evidence phrase is broader than the trigger.",
            "The reasoning applies role abstention: it does not justify arguments from discourse background, causal context, document topic, or world knowledge alone.",
            "The reasoning applies an extra-event gate: it rejects semantically plausible events unless the text has an annotation-style trigger mention for that schema.",
            "Every evidence string is an exact contiguous quote from Text and contains the corresponding trigger or argument text.",
            "Evidence is locally informative: it should usually be a short phrase or clause showing the event or argument relation, not only an isolated token.",
            "The response does not use numeric offsets or token indices.",
        ],
        "return_contract": {
            "pass": "boolean",
            "scores": scores,
            "errors": ["short error labels, empty if pass"],
            "reason": "one concise sentence",
        },
        "input": {
            "text": e40.extract_text(row["input"]),
            "candidate_event_types": candidates,
            "schema_cards": schema_cards,
            "target_surface_events": e40.surface_gold_json(row),
            "generated_thinking": thinking,
            "generated_final": final_obj,
        },
    }
    if core_reasoning:
        payload["task"] = (
            f"Strictly verify the core reasoning in {variant['name']} CoT training data "
            "for event extraction."
        )
        payload["pass_requirements"] = [
            "The generated reasoning is task-grounded rather than a generic template and is consistent with the target event structure.",
            "The reasoning follows the requested variant strategy and audits plausible candidate frames before finalizing events.",
            "The reasoning explains schema grounding, close-type contrast when relevant, minimal trigger boundaries, argument-role grounding, and extraction-style control.",
            "The reasoning applies role abstention: it does not justify arguments from discourse background, causal context, document topic, or world knowledge alone.",
            "The reasoning applies an extra-event gate: it rejects semantically plausible events unless the text has an annotation-style trigger mention for that schema.",
            "The final surface structure is used only to check reasoning-final alignment; exact tuple recovery has already been deterministically verified.",
            "Do not judge or score the length or informativeness of final evidence strings. Under shortest_unique_evidence_v1 they are deterministic offset-recovery keys and may legitimately be isolated tokens.",
            "Do not reject or lower any score solely because an evidence string is short, provided the reasoning itself grounds the trigger, type, and argument relation in the source text.",
        ]
        payload["audit_scope"] = {
            "include": [
                "schema and candidate reasoning",
                "type discrimination",
                "trigger boundary control",
                "argument-role grounding",
                "role abstention",
                "extra-event control",
                "reasoning-final alignment",
            ],
            "exclude": [
                "final evidence length",
                "final evidence informativeness",
                "offset recovery already covered by deterministic validation",
            ],
        }
    if verifier_profile in {
        "target_role_alias_v1",
        "target_role_alias_core_reasoning_v1",
    }:
        payload["role_label_contract"] = [
            "Role labels in target_surface_events are authoritative dataset output labels and must remain unchanged.",
            "Schema-card role names describe semantic relations and may use a different naming vocabulary from the dataset output labels.",
            "Do not reject or lower a score solely because a target role label and a semantically corresponding schema-card role name are not identical.",
            "Still reject an argument when its local text does not support the target role relation; this contract does not relax local grounding, role abstention, evidence, type, trigger, candidate-coverage, or extra-event requirements.",
        ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def evidence_quality_errors(row: dict, thinking: str | None, final_obj: dict | None) -> list[str]:
    errors = []
    if not thinking:
        return errors
    variant = variant_from_row(row)
    wc = len(thinking.split())
    if wc < int(variant["min_words"]):
        errors.append(f"thinking_too_short_words:{wc}")
    if sentence_count(thinking) < int(variant["min_sentences"]):
        errors.append(f"thinking_too_few_sentences:{sentence_count(thinking)}")
    if final_obj is None:
        return errors
    source_n = norm_text(e40.extract_text(row["input"]))
    for event_i, event in enumerate(final_obj.get("events", []) or []):
        if not isinstance(event, dict):
            continue
        trigger = event.get("trigger") if isinstance(event.get("trigger"), dict) else {}
        trigger_text = trigger.get("text") or ""
        trigger_evidence = trigger.get("evidence") or ""
        trigger_words = norm_text(trigger_text).split()
        evidence_words = norm_text(trigger_evidence).split()
        if norm_text(trigger_evidence) not in source_n:
            errors.append(f"event_{event_i}_trigger_evidence_not_in_text")
        if norm_text(trigger_text) not in norm_text(trigger_evidence):
            errors.append(f"event_{event_i}_trigger_evidence_missing_text")
        if len(trigger_words) == 1 and len(evidence_words) <= 1:
            errors.append(f"event_{event_i}_trigger_evidence_too_short")
        if len(evidence_words) > 22:
            errors.append(f"event_{event_i}_trigger_evidence_too_long")
        for arg_i, arg in enumerate(event.get("arguments", []) or []):
            if not isinstance(arg, dict):
                continue
            arg_text = arg.get("text") or ""
            arg_evidence = arg.get("evidence") or ""
            arg_words = norm_text(arg_text).split()
            arg_ev_words = norm_text(arg_evidence).split()
            if norm_text(arg_evidence) not in source_n:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_not_in_text")
            if norm_text(arg_text) not in norm_text(arg_evidence):
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_missing_text")
            if len(arg_words) == 1 and len(arg_ev_words) <= 1:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_too_short")
            if len(arg_ev_words) > 24:
                errors.append(f"event_{event_i}_arg_{arg_i}_evidence_too_long")
    return errors


def normalize_xml_protocol(content: str) -> tuple[str, list[str]]:
    normalized = e40.strip_fence(content or "")
    fixes = []
    before = normalized
    normalized = re.sub(r"<\s*think\s*>", "<thinking>", normalized, flags=re.I)
    normalized = re.sub(r"<\s*/\s*think\s*>", "</thinking>", normalized, flags=re.I)
    if normalized != before:
        fixes.append("normalized_think_tag")
    before = normalized
    normalized = re.sub(r"<\s*thinking\s*>", "<thinking>", normalized, flags=re.I)
    normalized = re.sub(r"<\s*/\s*thinking\s*>", "</thinking>", normalized, flags=re.I)
    normalized = re.sub(r"<\s*final\s*>", "<final>", normalized, flags=re.I)
    normalized = re.sub(r"<\s*/\s*final\s*>", "</final>", normalized, flags=re.I)
    if normalized != before:
        fixes.append("normalized_tag_case_or_spacing")
    return normalized.strip(), fixes


def hard_verify(row: dict, content: str):
    normalized_content, protocol_fixes = normalize_xml_protocol(content)
    thinking, final_obj, errors = BASE_HARD_VERIFY(row, normalized_content)
    if normalized_content.strip() and not normalized_content.strip().lower().startswith("<thinking>"):
        errors.append("text_before_thinking")
    if normalized_content.strip() and not normalized_content.strip().lower().endswith("</final>"):
        errors.append("text_after_final")
    if re.search(r"<\s*/?\s*think\s*>", normalized_content or "", re.I):
        errors.append("malformed_think_tag")
    errors.extend(evidence_quality_errors(row, thinking, final_obj))
    return thinking, final_obj, errors


def json_wrapper_to_tag_content(content: str) -> tuple[str, dict | None, list[str]]:
    wrapper = e40.extract_json_obj(content or "")
    errors = []
    if not isinstance(wrapper, dict):
        return "", None, ["json_wrapper_parse_failed"]
    allowed = {"thinking", "final"}
    extra_keys = sorted(set(wrapper) - allowed)
    if extra_keys:
        errors.append("json_wrapper_extra_keys:" + ",".join(map(str, extra_keys[:5])))
    thinking = wrapper.get("thinking")
    final_obj = wrapper.get("final")
    if not isinstance(thinking, str) or not thinking.strip():
        errors.append("json_wrapper_missing_or_invalid_thinking")
    if not isinstance(final_obj, dict):
        errors.append("json_wrapper_missing_or_invalid_final")
    if errors:
        return "", wrapper, errors
    tag_content = f"<thinking>{thinking.strip()}</thinking>\n<final>{e40.compact_json(final_obj)}</final>"
    return tag_content, wrapper, []


def semantic_pass(verifier_obj: dict, semantic_profile: str = "full_v1"):
    if semantic_profile not in {"full_v1", "core_reasoning_v1"}:
        raise ValueError(f"unknown semantic profile: {semantic_profile}")
    errors = []
    if not isinstance(verifier_obj, dict):
        return False, ["verifier_not_object"]
    if verifier_obj.get("pass") is not True:
        errors.append("semantic_pass_false")
    scores = verifier_obj.get("scores") if isinstance(verifier_obj.get("scores"), dict) else {}
    score_keys = [
        "type_discrimination",
        "trigger_boundary_control",
        "argument_role_grounding",
        "extraction_style_control",
        "candidate_coverage",
        "minimal_trigger_separation",
        "role_abstention",
        "no_extra_event_gate",
    ]
    if semantic_profile == "full_v1":
        score_keys = [
            "trigger_evidence",
            "argument_evidence",
            "evidence_informativeness",
            *score_keys,
        ]
    for key in score_keys:
        try:
            val = int(scores.get(key))
        except Exception:
            val = 0
        if val < 4:
            errors.append(f"low_{key}:{val}")
    try:
        final_consistency = int(scores.get("final_structure_consistency"))
    except Exception:
        final_consistency = 0
    if final_consistency < 5:
        errors.append(f"low_final_structure_consistency:{final_consistency}")
    verifier_errors = verifier_obj.get("errors")
    if isinstance(verifier_errors, list) and verifier_errors:
        errors.append("semantic_errors:" + ",".join(map(str, verifier_errors[:5])))
    return not errors, errors


def repair_prompt(errors: list[str], failure_kind: str, repair_profile: str = "strict_full", output_protocol: str = "xml_tags") -> str:
    if output_protocol == "json_wrapper":
        return (
            "Regenerate the answer. Previous validation errors: "
            + json.dumps(errors, ensure_ascii=False)
            + "\nReturn one strict JSON object only, with exactly two top-level keys: `thinking` and `final`. "
            "`thinking` must be a natural-language string. `final` must be the surface-only event JSON object. "
            "Do not use markdown fences, XML tags, headings, or prose outside JSON. "
            "Keep the target event list unchanged. For every trigger/argument, evidence must be a short contiguous substring copied exactly from Text and must contain the corresponding text field. "
            "Do not paraphrase evidence, add extra events, or change any target type, trigger, argument text, or role."
        )
    if repair_profile == "concise":
        return (
            "Regenerate the answer. Previous validation errors: "
            + json.dumps(errors, ensure_ascii=False)
            + "\nKeep the target event list unchanged. Start exactly with <thinking> and end exactly with </final>. "
            "Use exactly two tag blocks: <thinking>...</thinking><final>...</final>. "
            "For every trigger/argument, evidence must be a short contiguous substring copied exactly from Text and must contain the corresponding text field. "
            "Do not paraphrase evidence, add extra events, or change any target type, trigger, argument text, or role."
        )
    return (
        "Regenerate the whole answer from scratch. The previous answer failed "
        f"{failure_kind} validation with these exact errors: "
        + json.dumps(errors, ensure_ascii=False)
        + "\n\nNon-negotiable repair rules:\n"
        "1. Output must start with the exact characters <thinking> and must end with the exact characters </final>.\n"
        "2. Output exactly two top-level lowercase tag blocks: <thinking>...</thinking><final>...</final>. "
        "No text before <thinking>, between </thinking> and <final> except whitespace, or after </final>.\n"
        "3. Keep the final event list semantically identical to the target surface events from the original instruction. "
        "Do not add, remove, rename, merge, or split events or arguments.\n"
        "4. Every trigger.text and argument.text must be copied exactly from the source text.\n"
        "5. Every trigger.evidence and argument.evidence must be a short contiguous substring copied exactly from the source text. "
        "Do not paraphrase evidence, normalize wording, summarize, or invent bridging text.\n"
        "6. Each evidence field must contain its corresponding text field exactly.\n"
        "7. Keep evidence local and short: prefer the smallest clause or sentence fragment that proves the trigger or role.\n"
        "8. In <thinking>, explicitly explain how unsupported candidate events and unsupported roles are rejected, "
        "but do not let the reasoning contradict the final JSON.\n\n"
        "Return only the repaired <thinking>...</thinking><final>{JSON}</final> answer."
    )


def process_one(row: dict, args, api_key: str) -> dict:
    sample_id = row["meta"]["e40_sample_id"]
    rec = {"sample_id": sample_id, "source_index": row["meta"].get("e40_source_index"), "accepted": False, "attempts": []}
    json_wrapper = args.output_protocol == "json_wrapper"
    messages = [
        {
            "role": "system",
            "content": (
                "You create faithful, general, task-grounded CoT supervision for event extraction. "
                "Return one strict JSON object only with exactly two top-level keys: `thinking` and `final`. "
                "Do not use markdown fences, XML tags, headings, or prose outside the JSON object."
                if json_wrapper
                else (
                    "You create faithful, general, task-grounded CoT supervision for event extraction. "
                    "Your full answer must start with the exact lowercase tag <thinking> and must end with </final>. "
                    "Output exactly two lowercase tags: <thinking>...</thinking> and <final>...</final>. "
                    "Do not write any text before <thinking> or after </final>."
                )
            ),
        },
        {"role": "user", "content": generator_prompt(row, args.prompt_profile, args.output_protocol)},
    ]
    for attempt in range(max(1, args.max_attempts)):
        gen = None
        ver = None
        error_stage = "generator_call"
        try:
            gen = call_model(
                args.base_url,
                api_key,
                args.model,
                messages,
                args.gen_max_tokens,
                args.timeout,
                args.reasoning_effort,
            )
            error_stage = "generator_parse"
            raw_content = gen.get("content") or ""
            wrapper_obj = None
            if json_wrapper:
                content, wrapper_obj, parse_errors = json_wrapper_to_tag_content(raw_content)
                if parse_errors:
                    thinking, final_obj, hard_errors = None, None, parse_errors
                else:
                    thinking, final_obj, hard_errors = hard_verify(row, content)
            else:
                content = raw_content
                thinking, final_obj, hard_errors = hard_verify(row, content)
            attempt_rec = {
                "attempt": attempt + 1,
                "generator": gen,
                "raw_content": raw_content if json_wrapper else None,
                "normalized_content": content if json_wrapper else None,
                "json_wrapper_obj": wrapper_obj,
                "thinking": thinking,
                "final_obj": final_obj,
                "hard_errors": hard_errors,
                "hard_ok": not hard_errors,
            }
            if hard_errors:
                rec["attempts"].append(attempt_rec)
                messages.append({"role": "assistant", "content": raw_content})
                messages.append(
                    {
                        "role": "user",
                        "content": repair_prompt(hard_errors, "hard", args.repair_profile, args.output_protocol),
                    }
                )
                continue
            error_stage = "verifier_call"
            ver = call_model(
                args.base_url,
                api_key,
                args.verifier_model,
                [
                    {"role": "system", "content": "You are a strict verifier for event-extraction CoT/evidence data. Return strict JSON only."},
                    {"role": "user", "content": verifier_prompt(row, thinking or "", final_obj or {})},
                ],
                args.verify_max_tokens,
                args.timeout,
                args.verifier_reasoning_effort,
            )
            error_stage = "verifier_parse"
            verifier_obj = e40.extract_json_obj(ver.get("content") or "")
            error_stage = "verifier_validation"
            ok, semantic_errors = semantic_pass(verifier_obj)
            attempt_rec.update(
                {
                    "verifier": ver,
                    "verifier_obj": verifier_obj,
                    "semantic_errors": semantic_errors,
                    "semantic_ok": ok,
                }
            )
            rec["attempts"].append(attempt_rec)
            if ok:
                rec.update(attempt_rec)
                rec.pop("error", None)
                rec["api_ok"] = True
                rec["verifier_api_ok"] = True
                rec["accepted"] = True
                return rec
            messages.append({"role": "assistant", "content": raw_content})
            messages.append(
                {
                    "role": "user",
                    "content": repair_prompt(semantic_errors, "semantic", args.repair_profile, args.output_protocol),
                }
            )
        except Exception as exc:  # keep batch generation resilient
            error_attempt = {
                "attempt": attempt + 1,
                "error": repr(exc),
                "error_stage": error_stage,
            }
            if gen is not None:
                error_attempt["generator"] = gen
            if ver is not None:
                error_attempt["verifier"] = ver
            rec["attempts"].append(error_attempt)
    if rec["attempts"]:
        last = rec["attempts"][-1]
        rec.update(last)
        if "error" not in last:
            rec.pop("error", None)
        rec["api_ok"] = any(bool(item.get("generator")) for item in rec["attempts"])
        rec["verifier_api_ok"] = any(
            bool(item.get("verifier")) for item in rec["attempts"]
        )
    return rec


def should_retry_existing(rec: dict, args) -> bool:
    if args.retry_error_contains:
        return args.retry_error_contains in str(rec.get("error") or "")
    return bool(args.retry_rejected and not rec.get("accepted"))


def run_generation(rows: list[dict], args) -> list[dict]:
    raw_path = args.output_dir / "e40_raw.jsonl"
    existing = {}
    if args.reuse_existing and raw_path.exists():
        for rec in e40.load_jsonl(raw_path):
            if should_retry_existing(rec, args):
                continue
            existing[rec["sample_id"]] = rec
    pending = [row for row in rows if row["meta"]["e40_sample_id"] not in existing]
    results = [existing[row["meta"]["e40_sample_id"]] for row in rows if row["meta"]["e40_sample_id"] in existing]
    if pending:
        api_key = resolve_api_key()
        if not api_key:
            names = ", ".join(API_KEY_ENV_NAMES)
            raise SystemExit(f"a LiteLLM API key is required via one of: {names}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(process_one, row, args, api_key) for row in pending]
            for fut in concurrent.futures.as_completed(futs):
                rec = fut.result()
                results.append(rec)
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(
                    json.dumps(
                        {
                            "sample_id": rec["sample_id"],
                            "accepted": rec.get("accepted"),
                            "hard": rec.get("hard_errors"),
                            "semantic": rec.get("semantic_errors"),
                            "error": rec.get("error"),
                        },
                        ensure_ascii=False,
                    )
                )
    results = sorted(results, key=lambda r: r["sample_id"])
    e40.write_jsonl(raw_path, results)
    return results


def make_evidence_row(row: dict, thinking: str, final_obj: dict, dataset_role: str, run_name: str) -> dict:
    out = BASE_MAKE_EVIDENCE_ROW(row, thinking, final_obj, dataset_role, run_name)
    variant_key = variant_key_from_name(run_name)
    variant = VARIANTS[variant_key]
    is_autocluster = run_name.startswith(
        (
            "e95",
            "e111",
            "e121",
            "e122",
            "e123",
            "e124",
            "e125",
            "e126",
            "e127",
        )
    )
    is_retrieved_abstention = run_name.startswith("e130")
    if is_retrieved_abstention:
        instruction_variant = "retrieved-candidate trigger-locked arbitration with abstention"
        instruction_focus = (
            "audit local mentions against supplied schemas, lock triggers for supported frames, "
            "and abstain when no candidate supports a mention rather than substituting a nearby type"
        )
        variant_name = "retrieved_candidate_abstention"
    elif is_autocluster:
        instruction_variant = "trigger-locked-auto-cluster-arbitration"
        instruction_focus = (
            "audit plausible frames, lock minimal trigger anchors, contrast only the auto-derived "
            "confusable neighbors present in this input's candidate set, and attach locally supported arguments"
        )
        variant_name = "trigger_locked_autocluster"
    else:
        instruction_variant = variant["name"].replace("_", "-")
        instruction_focus = variant["instruction_focus"]
        variant_name = variant["name"]
    out["instruction"] = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        f"First output `<thinking>...</thinking>` with {instruction_variant} natural-language reasoning: {instruction_focus}. "
        "Then output `<final>{...}</final>` with a surface-only JSON event list: each trigger and argument must include `text` and a short contiguous local `evidence` quote from the input text. "
        "Do not output numeric offsets, token indices, or text outside these lowercase tags."
    )
    meta = out.setdefault("meta", {})
    meta.update(
        {
            "adaptive_source": f"strategy_variant_evidence_cot_{variant_key}",
            "adaptive_target_style": f"{variant_name}_thinking_surface_evidence_cot",
            "e47_run_name": run_name,
            "e47_variant": variant_name,
            "e40_generator_model": ACTIVE_GENERATOR_MODEL,
            "e40_verifier_model": ACTIVE_VERIFIER_MODEL,
            "e47_generator_model": ACTIVE_GENERATOR_MODEL,
            "e47_verifier_model": ACTIVE_VERIFIER_MODEL,
            "e130_retrieved_abstention": is_retrieved_abstention,
        }
    )
    return out


def make_eval_evidence_row(row: dict, dataset_role: str, run_name: str) -> dict:
    placeholder = VARIANTS[variant_key_from_name(run_name)]["placeholder"]
    return make_evidence_row(row, placeholder, e40.surface_with_empty_evidence(row), dataset_role, run_name)


def write_train_config(branch: str, train_name: str, dev_name: str) -> Path:
    path = e40.CONFIG_DIR / f"{QWEN4_RUN_PREFIX}_{branch}_full_stepmatch.yaml"
    config = {
        "model_name_or_path": QWEN4_WARM_START,
        "template": "qwen",
        "dataset_dir": TRAIN_DATASET_DIR,
        "dataset": train_name,
        "eval_dataset": dev_name,
        "output_dir": f"/workspace/project/outputs/stage2_adaptive_runs_user/{QWEN4_RUN_PREFIX}_{branch}_full",
        "stage": "sft",
        "do_train": True,
        "overwrite_cache": True,
        "preprocessing_num_workers": 8,
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "logging_steps": 1,
        "report_to": "none",
        "finetuning_type": "full",
        "cutoff_len": 1536,
        "max_samples": 20000,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "packing": False,
        "learning_rate": 2.0e-6,
        "warmup_ratio": 0.05,
        "bf16": True,
        "val_size": 0.0,
        "eval_steps": 10,
        "do_eval": True,
        "save_only_model": True,
        "num_train_epochs": 3.0,
        "load_best_model_at_end": False,
        "deepspeed": "/workspace/project/configs/deepspeed/zero2_optimizer_offload_cpu.json",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="e47b_qwen4_seed1500_candidate_audit_cot")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=4747)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base_url", default=e40.DEFAULT_BASE_URL)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--verifier_model", default="deepseek-v4-pro")
    parser.add_argument("--gen_max_tokens", type=int, default=8192)
    parser.add_argument("--verify_max_tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--verifier_reasoning_effort", default=None)
    parser.add_argument(
        "--prompt_profile",
        choices=[
            "standard",
            "strict_evidence",
            "json_acceptance_v2",
            "xml_lean_v3",
            "e70_candidate_audit_v2",
            "e72_e57_backbone_subtype_minarg",
            "e73_e57_recall_first_exactness_last",
            "e76_contrastive_exactness",
            "e80a_no_type_arbitration",
            "e80b_no_argument_pruning",
            "e81_trigger_locked_arbitration",
            "e130_retrieved_abstention",
            "e82_schema_driven_arbitration",
            "e83_trigger_locked_schema_driven",
            "e84_trigger_locked_no_arbitration",
            "e93_trigger_locked_genericsuppress",
            "e94_trigger_locked_arg_recall",
            "e95_trigger_locked_autocluster",
            "freeform_nl",
            "norecall_first",
            "e71_event_frame_first_light",
            "e71_event_frame_first_argument",
        ],
        default="standard",
    )
    parser.add_argument("--repair_profile", choices=["strict_full", "concise"], default="strict_full")
    parser.add_argument("--output_protocol", choices=["xml_tags", "json_wrapper"], default="xml_tags")
    parser.add_argument("--reuse_existing", action="store_true", default=True)
    parser.add_argument("--retry_rejected", action="store_true")
    parser.add_argument("--retry_error_contains", default=None)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--sampled_rows_path", type=Path, default=None)
    parser.add_argument(
        "--sampled_rows_mode",
        choices=["prefix", "priority_sample"],
        default="prefix",
        help=(
            "How --limit is applied to --sampled_rows_path. Use priority_sample to "
            "reuse the seeded E40 priority-plus-random selection recipe."
        ),
    )
    parser.add_argument("--output_dir", type=Path)
    # Dataset-family overrides (default to RichERE; set these for ACE05 cross-dataset).
    parser.add_argument("--run_prefix", default=None,
                        help="Config/output naming prefix; default richere QWEN4_RUN_PREFIX.")
    parser.add_argument("--warm_start", default=None,
                        help="Warm-start checkpoint for the generated train config; default richere Direct ck2064.")
    parser.add_argument("--auto_cluster_map_path", default=None,
                        help="Schema-derived confusable-neighbor map for e95 (build_auto_cluster_map_*.py); default richere map.")
    parser.add_argument("--adaptive_prefix", default=None,
                        help="Generated dataset naming prefix; default e40.ADAPTIVE_PREFIX (richere).")
    parser.add_argument("--data_prefix", default=None,
                        help="Gold pool prefix for eval splits under FORMAL_DATA_DIR; default e40.DATA_PREFIX (richere).")
    parser.add_argument("--formal_data_dir", type=Path, default=None,
                        help="Override the formal gold-pool directory used with --data_prefix.")
    parser.add_argument("--adaptive_data_dir", type=Path, default=None,
                        help="Override the generated dataset directory instead of the shared adaptive directory.")
    parser.add_argument("--config_dir", type=Path, default=None,
                        help="Override the generated train-config directory.")
    parser.add_argument("--train_dataset_dir", default=None,
                        help="Container-visible dataset_dir written into the generated train config.")
    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Write accepted training rows only; do not read or materialize dev/test datasets or a train config.",
    )
    return parser.parse_args()


def main():
    global QWEN4_RUN_PREFIX, QWEN4_WARM_START, AUTO_CLUSTER_MAP_PATH, TRAIN_DATASET_DIR
    global ACTIVE_GENERATOR_MODEL, ACTIVE_VERIFIER_MODEL
    e40.generator_prompt = generator_prompt
    e40.verifier_prompt = verifier_prompt
    e40.hard_verify = hard_verify
    e40.semantic_pass = semantic_pass
    e40.process_one = process_one
    e40.run_generation = run_generation
    e40.make_evidence_row = make_evidence_row
    e40.make_eval_evidence_row = make_eval_evidence_row
    e40.write_train_config = write_train_config

    args = parse_args()
    ACTIVE_GENERATOR_MODEL = args.model
    ACTIVE_VERIFIER_MODEL = args.verifier_model
    # Dataset-family overrides (default to RichERE; set for ACE05 cross-dataset).
    if args.run_prefix:
        QWEN4_RUN_PREFIX = args.run_prefix
    if args.warm_start:
        QWEN4_WARM_START = args.warm_start
    if getattr(args, "auto_cluster_map_path", None):
        AUTO_CLUSTER_MAP_PATH = args.auto_cluster_map_path
    if args.adaptive_prefix:
        e40.ADAPTIVE_PREFIX = args.adaptive_prefix
    if args.data_prefix:
        e40.DATA_PREFIX = args.data_prefix
    if args.formal_data_dir:
        e40.FORMAL_DATA_DIR = args.formal_data_dir
    if args.adaptive_data_dir:
        e40.DATA_DIR = args.adaptive_data_dir
        e37.DATA_DIR = args.adaptive_data_dir
        ensure_dataset_registry(args.adaptive_data_dir)
    if args.config_dir:
        e40.CONFIG_DIR = args.config_dir
    if args.train_dataset_dir:
        TRAIN_DATASET_DIR = args.train_dataset_dir
    e40.RUN_PREFIX = QWEN4_RUN_PREFIX
    if args.output_dir is None:
        args.output_dir = REPO / "outputs/stage2_strategy_cot_e47" / f"{args.run_name}_20260606"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.sampled_rows_path is not None:
        source_rows = e40.load_jsonl(args.sampled_rows_path)
        if args.sampled_rows_mode == "priority_sample":
            sampled = e40.sample_rows(source_rows, args.limit, args.seed, args.run_name)
        else:
            sampled = source_rows[: args.limit] if args.limit else source_rows
            for idx, row in enumerate(sampled):
                row.setdefault("meta", {})
                row["meta"].setdefault(
                    "e40_source_index", row["meta"].get("source_index", idx)
                )
                row["meta"]["e40_sample_id"] = f"{args.run_name}_{idx:04d}"
    else:
        source_rows = e40.load_jsonl(e40.FORMAL_DATA_DIR / f"{e40.DATA_PREFIX}_train_pos.jsonl")
        sampled = e40.sample_rows(source_rows, args.limit, args.seed, args.run_name)
    e40.write_jsonl(args.output_dir / "sampled_rows.jsonl", sampled)
    results = e40.run_generation(sampled, args)
    if args.train_only:
        by_id = {row["meta"]["e40_sample_id"]: row for row in sampled}
        accepted = [record for record in results if record.get("accepted")]
        accepted_rows = []
        train_rows = []
        for record in accepted:
            row = by_id[record["sample_id"]]
            accepted_rows.append(
                {
                    **record,
                    "input": row["input"],
                    "gold_output": e40.compact_json(e40.gold_json(row)),
                    "meta": row.get("meta", {}),
                }
            )
            train_rows.append(
                make_evidence_row(
                    row, record["thinking"], record["final_obj"], "train", args.run_name
                )
            )
        e40.write_jsonl(args.output_dir / "accepted_evidence_cot.jsonl", accepted_rows)
        branch = f"{args.run_name}_thinking_evidence_cot"
        train_name = f"{e40.ADAPTIVE_PREFIX}_{branch}_train_pos"
        e40.write_dataset(train_name, train_rows)
        dataset_info = {
            "branch": branch,
            "accepted_count": len(accepted),
            "train_dataset": train_name,
            "train_rows": len(train_rows),
            "eval_datasets": [],
            "train_config": None,
            "train_only": True,
        }
    else:
        dataset_info = e40.write_datasets(sampled, results, args)
    summary = e40.summarize(sampled, results, dataset_info, args)
    variant_key = variant_key_from_name(args.run_name)
    summary["mode"] = "strategy_variant_evidence_cot_e47"
    summary["variant"] = (
        "retrieved_candidate_abstention"
        if args.prompt_profile == "e130_retrieved_abstention"
        else (
            "trigger_locked_autocluster"
            if args.prompt_profile == "e95_trigger_locked_autocluster"
            else VARIANTS[variant_key]["name"]
        )
    )
    summary["qwen4_warm_start"] = QWEN4_WARM_START
    summary["reasoning_effort"] = args.reasoning_effort
    summary["verifier_reasoning_effort"] = args.verifier_reasoning_effort
    summary["prompt_profile"] = args.prompt_profile
    summary["repair_profile"] = args.repair_profile
    summary["output_protocol"] = args.output_protocol
    e40.write_json(args.output_dir / "e47_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
