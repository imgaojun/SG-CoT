import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from src.stage2_cot.build_selective_aux_reasoning_dataset import (  # noqa: E402
    build_auxiliary_payload,
    build_decisions,
    normalize_event,
    parse_output_events,
    row_id,
)
from src.stage2_data.build_formal_stage2_dataset import (  # noqa: E402
    load_jsonl,
    load_schema_map,
    update_dataset_info,
    write_json,
)


ROUTE_FREE = "free_route"
ROUTE_FORCED_DIRECT = "forced_direct"
ROUTE_FORCED_REASON = "forced_reason"
PLAN_TARGET_STYLES = {
    "plan_lite",
    "type_plan_lite",
    "arg_plan_lite",
    "type_plan_v2",
    "type_role_hint_plan_lite",
    "calibrated_type_role_hint_plan_lite",
}


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_route_labels(path: Path | None):
    if path is None:
        return {}
    labels = {}
    for row in load_jsonl(path):
        labels[row["wnd_id"]] = row
    return labels


def adapt_input(input_text: str):
    return input_text.replace("\n\nReturn JSON only.", "\n\nReturn the tagged adaptive output only.")


def uses_plan_target(target_style: str):
    return target_style in PLAN_TARGET_STYLES


def render_instruction(route_mode: str, target_style: str):
    common = (
        "You are doing event extraction. Use only the provided candidate event types and schema cards. "
        "The final extraction must be strict JSON with top-level key `events` and token offsets. "
        "Wrap the final extraction as `<FINAL>{...}</FINAL>`. "
        "Do not add text outside the requested tags."
    )
    if uses_plan_target(target_style):
        reason_format = (
            "a compact non-JSON extraction plan inside `<PLAN>...</PLAN>`"
        )
        if target_style == "type_plan_v2":
            reason_detail = (
                "The plan contains only short schema-boundary lines such as `E1 TYPE ...`, "
                "`E1 TRIGGER \"...\"`, `E1 CONTRAST reject ...`, and `E1 CUE ...`. "
                "Do not put arguments, JSON braces, or an `events` key inside `<PLAN>`."
            )
        elif target_style in {"type_role_hint_plan_lite", "calibrated_type_role_hint_plan_lite"}:
            reason_detail = (
                "The plan contains short lines such as `E1 TYPE ...`, `E1 TRIGGER \"...\"`, "
                "`E1 CONTRAST reject ...`, `E1 ROLE Agent present = \"argument\"`, "
                "and `E1 ROLE Place absent`. Do not put JSON or an `events` key inside `<PLAN>`."
            )
        else:
            reason_detail = (
                "The plan contains short lines such as `E1 TYPE ...`, `E1 TRIGGER \"...\"`, "
                "`E1 CONTRAST reject ...`, and `E1 ARG Role = \"argument\"`. "
                "Do not put JSON or an `events` key inside `<PLAN>`."
            )
    else:
        reason_format = "compact JSON reasoning inside `<REASON>...</REASON>`"
        reason_detail = (
            "The reasoning JSON contains only compact event type decisions and role checks, not the final `events` key."
        )
    if route_mode == ROUTE_FORCED_DIRECT:
        return (
            common
            + " Use the direct route. First output `<ROUTE>direct</ROUTE>`, then output `<FINAL>{...}</FINAL>`. "
            'If no valid event is expressed by the candidate set, the final JSON is {"events": []}.'
        )
    if route_mode == ROUTE_FORCED_REASON:
        return (
            common
            + f" Use the reasoning route. First output `<ROUTE>reason</ROUTE>`, then output {reason_format}, "
            "then output `<FINAL>{...}</FINAL>`. "
            + reason_detail
            + ' If no valid event is expressed by the candidate set, use an empty plan/reasoning block and final JSON {"events": []}.'
        )
    if route_mode != ROUTE_FREE:
        raise ValueError(f"Unsupported route_mode: {route_mode}")
    if target_style == "calibrated_type_role_hint_plan_lite":
        return (
            common
            + " First decide whether this case should be solved directly or with compact reasoning. "
            "Use reason only when event type, trigger, or role assignment is ambiguous. "
            "Use direct when extraction is straightforward. "
            "Hard samples may appear in both seen and unseen schemas. "
            "If using direct, output `<ROUTE>direct</ROUTE>` followed by `<FINAL>{...}</FINAL>`. "
            f"If using reason, output `<ROUTE>reason</ROUTE>`, {reason_format}, then `<FINAL>{{...}}</FINAL>`. "
            + reason_detail
            + ' If no valid event is expressed by the candidate set, the final JSON is {"events": []}.'
        )
    return (
        common
        + " First decide whether this case should be solved directly or with compact reasoning. "
        "If it is simple, output `<ROUTE>direct</ROUTE>` followed by `<FINAL>{...}</FINAL>`. "
        f"If it is schema-confusable or role-ambiguous, output `<ROUTE>reason</ROUTE>`, {reason_format}, "
        "then `<FINAL>{...}</FINAL>`. "
        + reason_detail
        + ' If no valid event is expressed by the candidate set, the final JSON is {"events": []}.'
    )


def render_route_classifier_instruction():
    return (
        "You are doing route selection for schema-conditioned event extraction. "
        "Use only the provided text, candidate event types, and schema cards. "
        "Choose whether a downstream extractor should use direct extraction or compact reasoning. "
        "Output exactly one tag and nothing else: `<ROUTE>direct</ROUTE>` or `<ROUTE>reason</ROUTE>`. "
        "Use `<ROUTE>reason</ROUTE>` when reasoning is expected to improve event type, trigger, or role grounding. "
        "Use `<ROUTE>direct</ROUTE>` when direct extraction should be sufficient."
    )


def direct_target(final_json_text: str):
    return f"<ROUTE>direct</ROUTE>\n<FINAL>{final_json_text}</FINAL>"


def route_only_target(route_label: str):
    return f"<ROUTE>{route_label}</ROUTE>"


def quote_text(value: str):
    return json.dumps(value, ensure_ascii=False)


def plan_lines_for_event(event_index: int, event, decision, schema_by_type, max_args: int, target_style: str):
    label = f"E{event_index}"
    lines = []
    trigger = event.get("trigger", {})
    trigger_text = trigger.get("text", "")
    if target_style in {"plan_lite", "type_plan_lite", "type_plan_v2"}:
        lines.append(f"{label} TYPE {event['event_type']}")
    if target_style in {"type_role_hint_plan_lite", "calibrated_type_role_hint_plan_lite"}:
        lines.append(f"{label} TYPE {event['event_type']}")
    if target_style in {"plan_lite", "type_plan_lite", "arg_plan_lite", "type_plan_v2", "type_role_hint_plan_lite", "calibrated_type_role_hint_plan_lite"}:
        lines.append(f"{label} TRIGGER {quote_text(trigger_text)}")
    if target_style in {"plan_lite", "type_plan_lite", "type_plan_v2", "type_role_hint_plan_lite", "calibrated_type_role_hint_plan_lite"}:
        contrast = decision.get("contrast_type") if decision else ""
        if contrast:
            lines.append(f"{label} CONTRAST reject {contrast}")
    if target_style == "type_plan_v2":
        reason_code = decision.get("reason_code") if decision else ""
        if reason_code:
            lines.append(f"{label} CUE {reason_code}")
    if target_style in {"plan_lite", "arg_plan_lite"}:
        for arg in event.get("arguments", [])[:max_args]:
            lines.append(f"{label} ARG {arg['role']} = {quote_text(arg['text'])}")
    if target_style in {"type_role_hint_plan_lite", "calibrated_type_role_hint_plan_lite"} and max_args > 0:
        args_by_role = {}
        for arg in event.get("arguments", []):
            args_by_role.setdefault(arg["role"], []).append(arg)
        core_roles = list(schema_by_type[event["event_type"]].get("core_roles", []))
        emitted = 0
        for role in core_roles:
            if emitted >= max_args:
                break
            args = args_by_role.get(role, [])
            if args:
                for arg in args:
                    if emitted >= max_args:
                        break
                    lines.append(f"{label} ROLE {role} present = {quote_text(arg['text'])}")
                    emitted += 1
            else:
                lines.append(f"{label} ROLE {role} absent")
                emitted += 1
        for arg in event.get("arguments", []):
            if emitted >= max_args:
                break
            if arg["role"] in core_roles:
                continue
            lines.append(f"{label} ROLE {arg['role']} present = {quote_text(arg['text'])}")
            emitted += 1
    return lines


def build_plan_text(events, candidate_types, schema_by_type, max_role_checks: int, target_style: str):
    if not events:
        return "NO_EVENT"
    decisions = build_decisions(events, candidate_types, schema_by_type)
    lines = []
    remaining_args = max_role_checks
    for idx, event in enumerate(events, start=1):
        event_arg_budget = max(0, remaining_args)
        event_lines = plan_lines_for_event(
            idx,
            event,
            decisions[idx - 1] if idx - 1 < len(decisions) else None,
            schema_by_type,
            event_arg_budget,
            target_style,
        )
        remaining_args -= sum(1 for line in event_lines if " ARG " in line or " ROLE " in line)
        lines.extend(event_lines)
    return "\n".join(lines) if lines else "NO_EVENT"


def reasoning_target(row, schema_by_type, max_role_checks: int, target_style: str):
    events = parse_output_events(row)
    candidate_types = row.get("meta", {}).get("candidate_types", [])
    if uses_plan_target(target_style):
        normalized_events = [normalize_event(event) for event in events]
        plan_text = build_plan_text(events, candidate_types, schema_by_type, max_role_checks, target_style)
        final_json_text = json.dumps({"events": normalized_events}, ensure_ascii=False)
        return (
            "<ROUTE>reason</ROUTE>\n"
            f"<PLAN>\n{plan_text}\n</PLAN>\n"
            f"<FINAL>{final_json_text}</FINAL>"
        )
    payload = build_auxiliary_payload(events, candidate_types, schema_by_type, max_role_checks, target_style)
    reason_payload = {key: value for key, value in payload.items() if key != "events"}
    final_json_text = json.dumps({"events": payload["events"]}, ensure_ascii=False)
    return (
        "<ROUTE>reason</ROUTE>\n"
        f"<REASON>{json.dumps(reason_payload, ensure_ascii=False)}</REASON>\n"
        f"<FINAL>{final_json_text}</FINAL>"
    )


def choose_route(row, label_map, route_mode: str):
    if route_mode == ROUTE_FORCED_DIRECT:
        return "direct", None
    if route_mode == ROUTE_FORCED_REASON:
        return "reason", None
    label_row = label_map.get(row_id(row))
    if label_row is None:
        return "direct", None
    return label_row.get("route_label", "direct"), label_row


def build_adaptive_row(
    row,
    schema_by_type,
    label_map,
    route_mode: str,
    max_role_checks: int,
    target_style: str,
    dataset_role: str,
    forced_label: str | None = None,
    pair_source: str | None = None,
    route_only: bool = False,
    route_classifier_prompt: bool = False,
):
    item = copy.deepcopy(row)
    label, label_row = choose_route(row, label_map, route_mode)
    if forced_label is not None:
        label = forced_label
    final_json_text = row["output"]
    item["instruction"] = (
        render_route_classifier_instruction()
        if route_classifier_prompt
        else render_instruction(route_mode, target_style)
    )
    item["input"] = adapt_input(item["input"])
    item["gold_output"] = final_json_text
    if route_only:
        item["output"] = route_only_target(label)
    else:
        item["output"] = (
            reasoning_target(row, schema_by_type, max_role_checks, target_style)
            if label == "reason"
            else direct_target(final_json_text)
        )
    meta = dict(item.get("meta", {}))
    meta.update(
        {
            "adaptive_source": "adaptive_route_reasoning",
            "adaptive_dataset_role": dataset_role,
            "adaptive_route_mode": route_mode,
            "adaptive_route_label": label,
            "adaptive_target_style": target_style,
            "adaptive_label_source": label_row.get("label_source") if label_row else None,
            "adaptive_label_reason_rate_cap": label_row.get("reason_rate_cap") if label_row else None,
            "adaptive_pair_source": pair_source,
            "adaptive_route_only": route_only,
            "adaptive_route_classifier_prompt": route_classifier_prompt,
        }
    )
    item["meta"] = meta
    return item


def variant_dataset_name(dataset_name: str, route_mode: str, role: str):
    if route_mode == ROUTE_FREE:
        return dataset_name
    marker = f"_{role}_pos"
    if marker not in dataset_name:
        return f"{dataset_name}_{route_mode}"
    return dataset_name.replace(marker, f"_{route_mode}_{role}_pos")


def register_dataset(dataset_dir: Path, dataset_name: str, rows, meta: dict):
    file_name = f"{dataset_name}.jsonl"
    write_jsonl(dataset_dir / file_name, rows)
    update_dataset_info(dataset_dir, dataset_name, file_name)
    write_json(dataset_dir / f"{dataset_name}.meta.json", {"dataset_name": dataset_name, "file_name": file_name, **meta})


def output_has_final(row):
    output = row.get("output", "")
    return "<FINAL>" in output and "</FINAL>" in output


def audit_rows(rows):
    route_only_rows = [row for row in rows if row.get("meta", {}).get("adaptive_route_only")]
    full_rows = [row for row in rows if not row.get("meta", {}).get("adaptive_route_only")]
    route_only_classifier_rows = [
        row for row in route_only_rows if row.get("meta", {}).get("adaptive_route_classifier_prompt")
    ]
    route_only_full_prompt_rows = [
        row for row in route_only_rows if not row.get("meta", {}).get("adaptive_route_classifier_prompt")
    ]
    full_with_final_rows = [row for row in full_rows if output_has_final(row)]
    full_without_final_rows = [row for row in full_rows if not output_has_final(row)]
    payload = {
        "total_rows": len(rows),
        "full_rows": len(full_rows),
        "full_rows_with_final": len(full_with_final_rows),
        "full_rows_without_final": len(full_without_final_rows),
        "route_only_rows": len(route_only_rows),
        "route_only_classifier_prompt_rows": len(route_only_classifier_rows),
        "route_only_full_extraction_prompt_rows": len(route_only_full_prompt_rows),
        "route_only_rows_with_final": sum(1 for row in route_only_rows if output_has_final(row)),
        "route_only_direct_rows": sum(
            1 for row in route_only_rows if row.get("meta", {}).get("adaptive_route_label") == "direct"
        ),
        "route_only_reason_rows": sum(
            1 for row in route_only_rows if row.get("meta", {}).get("adaptive_route_label") == "reason"
        ),
        "direct_full_rows": sum(
            1
            for row in full_rows
            if row.get("meta", {}).get("adaptive_route_label") == "direct" and output_has_final(row)
        ),
        "reason_full_rows": sum(
            1
            for row in full_rows
            if row.get("meta", {}).get("adaptive_route_label") == "reason" and output_has_final(row)
        ),
    }
    payload.update(
        {
            "total_count": payload["total_rows"],
            "full_with_final_count": payload["full_rows_with_final"],
            "route_only_count": payload["route_only_rows"],
            "route_only_classifier_prompt_count": payload["route_only_classifier_prompt_rows"],
            "route_only_full_extraction_prompt_count": payload["route_only_full_extraction_prompt_rows"],
        }
    )
    return payload


def build_rows(
    source_jsonl: str,
    schema_by_type,
    label_jsonl: str | None,
    route_mode: str,
    max_role_checks: int,
    target_style: str,
    dataset_role: str,
    pair_selected_direct: bool = False,
    direct_dup_selected_train: bool = False,
    train_pair_all_routes: bool = False,
    route_aux_repeat: int = 0,
    route_only_train: bool = False,
    route_only_eval: bool = False,
    route_classifier_prompt: bool = False,
    route_aux_classifier_prompt: bool = False,
    route_reason_oversample: int = 1,
    route_aux_reason_balance: bool = False,
    route_aux_reason_target_rate: float | None = None,
):
    label_map = load_route_labels(Path(label_jsonl) if label_jsonl else None)
    source_rows = load_jsonl(Path(source_jsonl))
    route_aux_reason_extra_by_id = {}
    route_aux_reason_extra_source = "route_only_aux_reason_balance" if route_aux_reason_balance else "route_only_aux_reason_target"
    route_aux_reason_target = 0.5 if route_aux_reason_balance else route_aux_reason_target_rate
    if (
        route_aux_reason_target is not None
        and dataset_role == "train"
        and route_mode == ROUTE_FREE
        and route_aux_repeat > 0
        and not route_only_train
    ):
        label_counts = {"direct": 0, "reason": 0}
        reason_ids = []
        for source_row in source_rows:
            source_label, _ = choose_route(source_row, label_map, route_mode)
            label_counts[source_label] = label_counts.get(source_label, 0) + 1
            if source_label == "reason":
                reason_ids.append(row_id(source_row))
        reason_aux_base = label_counts.get("reason", 0) * route_aux_repeat
        direct_aux_base = label_counts.get("direct", 0) * route_aux_repeat
        base_total = direct_aux_base + reason_aux_base
        if base_total and reason_aux_base / base_total < route_aux_reason_target:
            if not reason_ids:
                raise ValueError("Cannot add route-aux reason rows because there are no reason-labeled source rows")
            extra_total = math.ceil(
                (route_aux_reason_target * base_total - reason_aux_base)
                / (1.0 - route_aux_reason_target)
            )
            extra_each = extra_total // len(reason_ids)
            extra_remainder = extra_total % len(reason_ids)
            for idx, source_id in enumerate(reason_ids):
                route_aux_reason_extra_by_id[source_id] = extra_each + (1 if idx < extra_remainder else 0)
    rows = []
    for row in source_rows:
        current_route_only = route_only_train if dataset_role == "train" else route_only_eval
        if train_pair_all_routes and dataset_role == "train" and route_mode == ROUTE_FREE:
            rows.append(
                build_adaptive_row(
                    row,
                    schema_by_type,
                    label_map,
                    route_mode,
                    max_role_checks,
                    target_style,
                    dataset_role,
                    forced_label="direct",
                    pair_source="pair_all_direct",
                    route_only=current_route_only,
                    route_classifier_prompt=route_classifier_prompt,
                )
            )
            rows.append(
                build_adaptive_row(
                    row,
                    schema_by_type,
                    label_map,
                    route_mode,
                    max_role_checks,
                    target_style,
                    dataset_role,
                    forced_label="reason",
                    pair_source="pair_all_reason",
                    route_only=current_route_only,
                    route_classifier_prompt=route_classifier_prompt,
                )
            )
            continue
        force_direct_control = direct_dup_selected_train and dataset_role == "train" and route_mode == ROUTE_FREE
        item = build_adaptive_row(
            row,
            schema_by_type,
            label_map,
            route_mode,
            max_role_checks,
            target_style,
            dataset_role,
            forced_label="direct" if force_direct_control else None,
            pair_source="direct_duplicate_control_base" if force_direct_control else None,
            route_only=current_route_only,
            route_classifier_prompt=route_classifier_prompt,
        )
        rows.append(item)
        if (
            dataset_role == "train"
            and route_mode == ROUTE_FREE
            and route_reason_oversample > 1
            and item.get("meta", {}).get("adaptive_route_label") == "reason"
        ):
            for dup_idx in range(route_reason_oversample - 1):
                rows.append(
                    build_adaptive_row(
                        row,
                        schema_by_type,
                        label_map,
                        route_mode,
                        max_role_checks,
                        target_style,
                        dataset_role,
                        forced_label="reason",
                        pair_source=f"route_reason_oversample_{dup_idx + 1}",
                        route_only=current_route_only,
                        route_classifier_prompt=route_classifier_prompt,
                    )
                )
        if route_aux_repeat > 0 and dataset_role == "train" and route_mode == ROUTE_FREE and not route_only_train:
            for aux_idx in range(route_aux_repeat):
                rows.append(
                    build_adaptive_row(
                        row,
                        schema_by_type,
                        label_map,
                        route_mode,
                        max_role_checks,
                        target_style,
                        dataset_role,
                        forced_label=item.get("meta", {}).get("adaptive_route_label"),
                        pair_source=f"route_only_aux_{aux_idx + 1}",
                        route_only=True,
                        route_classifier_prompt=route_classifier_prompt or route_aux_classifier_prompt,
                    )
                )
            for aux_idx in range(route_aux_reason_extra_by_id.get(row_id(row), 0)):
                rows.append(
                    build_adaptive_row(
                        row,
                        schema_by_type,
                        label_map,
                        route_mode,
                        max_role_checks,
                        target_style,
                        dataset_role,
                        forced_label="reason",
                        pair_source=f"{route_aux_reason_extra_source}_{aux_idx + 1}",
                        route_only=True,
                        route_classifier_prompt=route_classifier_prompt or route_aux_classifier_prompt,
                    )
                )
        if direct_dup_selected_train and dataset_role == "train" and route_mode == ROUTE_FREE:
            original_label, _ = choose_route(row, label_map, route_mode)
            if original_label == "reason":
                rows.append(
                    build_adaptive_row(
                        row,
                        schema_by_type,
                        label_map,
                        route_mode,
                        max_role_checks,
                        target_style,
                        dataset_role,
                        forced_label="direct",
                        pair_source="selected_direct_duplicate_control",
                        route_only=current_route_only,
                        route_classifier_prompt=route_classifier_prompt,
                    )
                )
        if pair_selected_direct and dataset_role == "train" and route_mode == ROUTE_FREE:
            if item.get("meta", {}).get("adaptive_route_label") == "reason":
                rows.append(
                    build_adaptive_row(
                        row,
                        schema_by_type,
                        label_map,
                        route_mode,
                        max_role_checks,
                        target_style,
                        dataset_role,
                        forced_label="direct",
                        pair_source="selected_direct_anchor",
                        route_only=current_route_only,
                        route_classifier_prompt=route_classifier_prompt,
                    )
                )
    return rows, label_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--direct_train_jsonl", required=True)
    parser.add_argument("--direct_dev_jsonl", required=True)
    parser.add_argument("--direct_test_jsonl", required=True)
    parser.add_argument("--direct_test_seen_jsonl", required=True)
    parser.add_argument("--direct_test_unseen_jsonl", required=True)
    parser.add_argument("--train_label_jsonl", required=True)
    parser.add_argument("--dev_label_jsonl", default=None)
    parser.add_argument("--test_label_jsonl", default=None)
    parser.add_argument("--test_seen_label_jsonl", default=None)
    parser.add_argument("--test_unseen_label_jsonl", default=None)
    parser.add_argument("--dataset_dir", default="data/stage2_adaptive_datasets")
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--dev_dataset_name", required=True)
    parser.add_argument("--test_dataset_name", required=True)
    parser.add_argument("--test_seen_dataset_name", required=True)
    parser.add_argument("--test_unseen_dataset_name", required=True)
    parser.add_argument(
        "--target_style",
        choices=[
            "type_role_lite",
            "type_lite",
            "role_lite",
            "type_role_verify_lite",
            "plan_lite",
            "type_plan_lite",
            "arg_plan_lite",
            "type_plan_v2",
            "type_role_hint_plan_lite",
            "calibrated_type_role_hint_plan_lite",
        ],
        default="type_role_lite",
    )
    parser.add_argument("--max_role_checks_per_sample", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--write_forced_eval_variants", action="store_true")
    parser.add_argument("--write_forced_train_variants", action="store_true")
    parser.add_argument("--pair_selected_direct", action="store_true")
    parser.add_argument("--direct_dup_selected_train", action="store_true")
    parser.add_argument("--train_pair_all_routes", action="store_true")
    parser.add_argument("--route_aux_repeat", type=int, default=0)
    parser.add_argument("--route_only_train", action="store_true")
    parser.add_argument("--route_only_eval", action="store_true")
    parser.add_argument("--route_classifier_prompt", action="store_true")
    parser.add_argument("--route_aux_classifier_prompt", action="store_true")
    parser.add_argument("--allow_route_aux_full_prompt", action="store_true")
    parser.add_argument("--route_reason_oversample", type=int, default=1)
    parser.add_argument(
        "--route_aux_reason_balance",
        action="store_true",
        help=(
            "Add extra classifier-prompt route-only reason rows until route-only reason and direct "
            "counts are balanced. This affects only train route-aux rows."
        ),
    )
    parser.add_argument(
        "--route_aux_reason_target_rate",
        type=float,
        default=None,
        help=(
            "Add extra classifier-prompt route-only reason rows until the train route-only reason "
            "rate reaches this target. This affects only train route-aux rows."
        ),
    )
    args = parser.parse_args()
    if args.route_aux_repeat < 0:
        raise ValueError("--route_aux_repeat must be >= 0")
    if args.route_reason_oversample < 1:
        raise ValueError("--route_reason_oversample must be >= 1")
    if args.route_aux_reason_balance and args.route_aux_reason_target_rate is not None:
        raise ValueError("--route_aux_reason_balance and --route_aux_reason_target_rate are mutually exclusive")
    if args.route_aux_reason_target_rate is not None:
        if not 0.0 < args.route_aux_reason_target_rate < 1.0:
            raise ValueError("--route_aux_reason_target_rate must be in (0, 1)")
        if args.route_aux_repeat <= 0:
            raise ValueError("--route_aux_reason_target_rate requires --route_aux_repeat > 0")
        if args.route_only_train:
            raise ValueError("--route_aux_reason_target_rate is incompatible with --route_only_train")
    if sum([args.pair_selected_direct, args.direct_dup_selected_train, args.train_pair_all_routes]) > 1:
        raise ValueError("--pair_selected_direct, --direct_dup_selected_train, and --train_pair_all_routes are mutually exclusive")

    schema_by_type = load_schema_map(Path(args.schema_path))
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_labels = build_rows(
        args.direct_train_jsonl,
        schema_by_type,
        args.train_label_jsonl,
        ROUTE_FREE,
        args.max_role_checks_per_sample,
        args.target_style,
        "train",
        args.pair_selected_direct,
        args.direct_dup_selected_train,
        args.train_pair_all_routes,
        args.route_aux_repeat,
        args.route_only_train,
        args.route_only_eval,
        args.route_classifier_prompt,
        args.route_aux_classifier_prompt,
        args.route_reason_oversample,
        args.route_aux_reason_balance,
        args.route_aux_reason_target_rate,
    )
    random.Random(args.seed).shuffle(train_rows)
    train_audit = audit_rows(train_rows)
    if train_audit["full_rows_without_final"]:
        raise ValueError(f"Full training rows without <FINAL>: {train_audit['full_rows_without_final']}")
    if train_audit["route_only_rows_with_final"]:
        raise ValueError(f"Route-only training rows unexpectedly contain <FINAL>: {train_audit['route_only_rows_with_final']}")
    if train_audit["route_only_full_extraction_prompt_rows"] and not args.allow_route_aux_full_prompt:
        raise ValueError(
            "Route-only training rows use the full extraction prompt. "
            "Use --route_aux_classifier_prompt for route_aux rows, or pass --allow_route_aux_full_prompt to override."
        )

    shared_meta = {
        "schema_path": args.schema_path,
        "target_style": args.target_style,
        "max_role_checks_per_sample": args.max_role_checks_per_sample,
        "seed": args.seed,
        "direct_train_jsonl": args.direct_train_jsonl,
        "direct_dev_jsonl": args.direct_dev_jsonl,
        "direct_test_jsonl": args.direct_test_jsonl,
        "direct_test_seen_jsonl": args.direct_test_seen_jsonl,
        "direct_test_unseen_jsonl": args.direct_test_unseen_jsonl,
        "train_label_jsonl": args.train_label_jsonl,
        "dev_label_jsonl": args.dev_label_jsonl,
        "test_label_jsonl": args.test_label_jsonl,
        "test_seen_label_jsonl": args.test_seen_label_jsonl,
        "test_unseen_label_jsonl": args.test_unseen_label_jsonl,
        "format": "<ROUTE>{direct|reason}</ROUTE> optional <REASON>{...}</REASON> or <PLAN>...</PLAN> <FINAL>{\"events\": ...}</FINAL>",
        "pair_selected_direct": args.pair_selected_direct,
        "direct_dup_selected_train": args.direct_dup_selected_train,
        "train_pair_all_routes": args.train_pair_all_routes,
        "route_aux_repeat": args.route_aux_repeat,
        "route_only_train": args.route_only_train,
        "route_only_eval": args.route_only_eval,
        "route_classifier_prompt": args.route_classifier_prompt,
        "route_aux_classifier_prompt": args.route_aux_classifier_prompt,
        "allow_route_aux_full_prompt": args.allow_route_aux_full_prompt,
        "route_reason_oversample": args.route_reason_oversample,
        "route_aux_reason_balance": args.route_aux_reason_balance,
        "route_aux_reason_target_rate": args.route_aux_reason_target_rate,
        "write_forced_train_variants": args.write_forced_train_variants,
    }
    register_dataset(
        dataset_dir,
        args.train_dataset_name,
        train_rows,
        {
            **shared_meta,
            "dataset_role": "train",
            "route_mode": ROUTE_FREE,
            "num_examples": len(train_rows),
            "audit": train_audit,
            "reason_count": sum(1 for row in train_rows if row.get("meta", {}).get("adaptive_route_label") == "reason"),
            "paired_direct_anchor_count": sum(1 for row in train_rows if row.get("meta", {}).get("adaptive_pair_source") == "selected_direct_anchor"),
            "direct_duplicate_control_count": sum(1 for row in train_rows if row.get("meta", {}).get("adaptive_pair_source") == "selected_direct_duplicate_control"),
            "pair_all_direct_count": sum(1 for row in train_rows if row.get("meta", {}).get("adaptive_pair_source") == "pair_all_direct"),
            "pair_all_reason_count": sum(1 for row in train_rows if row.get("meta", {}).get("adaptive_pair_source") == "pair_all_reason"),
            "route_only_count": sum(1 for row in train_rows if row.get("meta", {}).get("adaptive_route_only")),
            "route_aux_count": sum(1 for row in train_rows if str(row.get("meta", {}).get("adaptive_pair_source", "")).startswith("route_only_aux_")),
            "route_aux_reason_balance_count": sum(1 for row in train_rows if str(row.get("meta", {}).get("adaptive_pair_source", "")).startswith("route_only_aux_reason_balance_")),
            "route_aux_reason_target_count": sum(1 for row in train_rows if str(row.get("meta", {}).get("adaptive_pair_source", "")).startswith("route_only_aux_reason_target_")),
            "route_reason_oversample_count": sum(1 for row in train_rows if str(row.get("meta", {}).get("adaptive_pair_source", "")).startswith("route_reason_oversample_")),
            "train_label_count": len(train_labels),
        },
    )
    if args.write_forced_train_variants:
        for train_route_mode in [ROUTE_FORCED_DIRECT, ROUTE_FORCED_REASON]:
            forced_train_rows, forced_train_labels = build_rows(
                args.direct_train_jsonl,
                schema_by_type,
                args.train_label_jsonl,
                train_route_mode,
                args.max_role_checks_per_sample,
                args.target_style,
                "train",
                False,
                False,
                False,
                0,
                args.route_only_train,
                args.route_only_eval,
                args.route_classifier_prompt,
                False,
                1,
            )
            register_dataset(
                dataset_dir,
                variant_dataset_name(args.train_dataset_name, train_route_mode, "train"),
                forced_train_rows,
                {
                    **shared_meta,
                    "dataset_role": "train",
                    "route_mode": train_route_mode,
                    "source_jsonl": args.direct_train_jsonl,
                    "label_jsonl": args.train_label_jsonl,
                    "num_examples": len(forced_train_rows),
                    "audit": audit_rows(forced_train_rows),
                    "reason_count": sum(1 for row in forced_train_rows if row.get("meta", {}).get("adaptive_route_label") == "reason"),
                    "label_count": len(forced_train_labels),
                },
            )

    eval_specs = [
        (args.direct_dev_jsonl, args.dev_dataset_name, args.dev_label_jsonl, "dev_seen"),
        (args.direct_test_jsonl, args.test_dataset_name, args.test_label_jsonl, "test"),
        (args.direct_test_seen_jsonl, args.test_seen_dataset_name, args.test_seen_label_jsonl, "test_seen"),
        (args.direct_test_unseen_jsonl, args.test_unseen_dataset_name, args.test_unseen_label_jsonl, "test_unseen"),
    ]
    route_modes = [ROUTE_FREE]
    if args.write_forced_eval_variants:
        route_modes.extend([ROUTE_FORCED_DIRECT, ROUTE_FORCED_REASON])

    for source_jsonl, dataset_name, label_jsonl, role in eval_specs:
        for route_mode in route_modes:
            rows, label_map = build_rows(
                source_jsonl,
                schema_by_type,
                label_jsonl,
                route_mode,
                args.max_role_checks_per_sample,
                args.target_style,
                role,
                False,
                False,
                False,
                0,
                False,
                args.route_only_eval,
                args.route_classifier_prompt,
                False,
                1,
            )
            eval_audit = audit_rows(rows)
            if eval_audit["full_rows_without_final"] and not args.route_only_eval:
                raise ValueError(f"{dataset_name} {route_mode} rows without <FINAL>: {eval_audit['full_rows_without_final']}")
            register_dataset(
                dataset_dir,
                variant_dataset_name(dataset_name, route_mode, role),
                rows,
                {
                    **shared_meta,
                    "dataset_role": role,
                    "route_mode": route_mode,
                    "source_jsonl": source_jsonl,
                    "label_jsonl": label_jsonl,
                    "num_examples": len(rows),
                    "audit": eval_audit,
                    "reason_count": sum(1 for row in rows if row.get("meta", {}).get("adaptive_route_label") == "reason"),
                    "label_count": len(label_map),
                },
            )

    print(
        json.dumps(
            {
                "train_dataset_name": args.train_dataset_name,
                "train_count": len(train_rows),
                "train_reason_count": sum(1 for row in train_rows if row.get("meta", {}).get("adaptive_route_label") == "reason"),
                "train_audit": train_audit,
                "dataset_dir": dataset_dir.as_posix(),
                "write_forced_eval_variants": args.write_forced_eval_variants,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
