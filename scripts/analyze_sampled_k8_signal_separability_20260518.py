#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA_PREFIX = "richere_balanced_split1_oracle_mixed_noise_top10_shuffle_adaptive"
LABEL_SOURCE = "sampled_counterfactual_utility_k8_checkpoint-258"
SOURCE_BRANCH = "sampled_reason_expert_forcedreason_from_noaux_20260517_forced_direct"
SCHEMA_PATH = REPO / "data/schema/richere-en.event_schema.json"
METRIC_KEYS = ["argument_f1", "event_f1", "trigger_f1", "score"]


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def load_schema(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["event_type"]: row for row in rows}


def row_key(row):
    meta = row.get("meta") or {}
    return meta.get("wnd_id") or row.get("wnd_id") or meta.get("doc_id")


def event_family(event_type: str):
    return event_type.split(":", 1)[0] if ":" in event_type else event_type


def normalize_cue_tokens(cues):
    tokens = set()
    for cue in cues or []:
        for piece in str(cue).lower().replace("-", " ").replace("/", " ").split():
            if piece:
                tokens.add(piece)
    return tokens


def jaccard(left, right):
    union = set(left) | set(right)
    if not union:
        return 0.0
    return len(set(left) & set(right)) / len(union)


def schema_roles(schema_by_type, event_type):
    return set(schema_by_type.get(event_type, {}).get("core_roles") or [])


def schema_cues(schema_by_type, event_type):
    return normalize_cue_tokens(schema_by_type.get(event_type, {}).get("trigger_cues") or [])


def type_confusion_score(left_type, right_type, schema_by_type):
    if left_type not in schema_by_type or right_type not in schema_by_type:
        return 0.0
    same_family = 1.0 if event_family(left_type) == event_family(right_type) else 0.0
    role_overlap = jaccard(schema_roles(schema_by_type, left_type), schema_roles(schema_by_type, right_type))
    cue_overlap = jaccard(schema_cues(schema_by_type, left_type), schema_cues(schema_by_type, right_type))
    return same_family * 10.0 + role_overlap * 3.0 + cue_overlap


def parse_prompt_text(input_text: str):
    text = ""
    token_line = ""
    if "Text:\n" in input_text:
        text = input_text.split("Text:\n", 1)[1].split("\n\nTokens:", 1)[0]
    if "Tokens:\n" in input_text:
        token_line = input_text.split("Tokens:\n", 1)[1].split("\n\nCandidate event types:", 1)[0]
    tokens = [tok for tok in token_line.split() if tok]
    if not tokens and text:
        tokens = text.split()
    return text, tokens


def pairwise(values):
    for i, left in enumerate(values):
        for right in values[i + 1:]:
            yield left, right


def mean(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else 0.0


def safe_ratio(num, den):
    return num / den if den else 0.0


def compact_events(payload):
    if isinstance(payload, str):
        payload = json.loads(payload)
    events = payload.get("events") or []
    return [event for event in events if isinstance(event, dict)]


def event_role_signature(event):
    roles = sorted({arg.get("role") for arg in event.get("arguments", []) if isinstance(arg, dict) and arg.get("role")})
    role_text = "|".join(roles) if roles else "NO_ROLES"
    return f"{event.get('event_type')}::{role_text}"


def build_gold_stats(source_rows):
    type_counts = Counter()
    role_signature_counts = Counter()
    for row in source_rows:
        for event in compact_events(row["gold_output"]):
            event_type = event.get("event_type")
            if event_type:
                type_counts[event_type] += 1
                role_signature_counts[event_role_signature(event)] += 1
    return {
        "type_counts": type_counts,
        "role_signature_counts": role_signature_counts,
        "max_type_freq": max(type_counts.values(), default=0),
        "max_signature_freq": max(role_signature_counts.values(), default=0),
    }


def log_frequency_rarity(freq, max_freq):
    if freq <= 0 or max_freq <= 0:
        return 0.0
    return 1.0 - math.log1p(freq) / math.log1p(max_freq)


def deployable_features(source_row, schema_by_type):
    meta = source_row.get("meta") or {}
    candidate_types = [typ for typ in meta.get("candidate_types", []) if typ in schema_by_type]
    text, tokens = parse_prompt_text(source_row.get("input") or "")
    text_lower = text.lower()
    token_set = {tok.lower().strip(".,;:!?\"'()[]{}") for tok in tokens}

    families = [event_family(typ) for typ in candidate_types]
    family_counts = Counter(families)
    same_family_pairs = sum(count * (count - 1) / 2 for count in family_counts.values())
    total_pairs = len(candidate_types) * (len(candidate_types) - 1) / 2

    confusion_scores = []
    role_overlaps = []
    cue_overlaps = []
    for left, right in pairwise(candidate_types):
        confusion_scores.append(type_confusion_score(left, right, schema_by_type))
        role_overlaps.append(jaccard(schema_roles(schema_by_type, left), schema_roles(schema_by_type, right)))
        cue_overlaps.append(jaccard(schema_cues(schema_by_type, left), schema_cues(schema_by_type, right)))

    cue_matched_types = 0
    cue_match_total = 0
    cue_token_overlap_total = 0
    for event_type in candidate_types:
        schema = schema_by_type[event_type]
        type_matched = False
        for cue in schema.get("trigger_cues") or []:
            cue_lower = str(cue).lower()
            cue_tokens = normalize_cue_tokens([cue])
            phrase_hit = cue_lower and cue_lower in text_lower
            token_hit = bool(cue_tokens & token_set)
            if phrase_hit or token_hit:
                type_matched = True
                cue_match_total += 1
            cue_token_overlap_total += len(cue_tokens & token_set)
        if type_matched:
            cue_matched_types += 1

    core_role_counts = [len(schema_roles(schema_by_type, typ)) for typ in candidate_types]
    return {
        "text_token_count": float(len(tokens)),
        "candidate_count": float(len(candidate_types)),
        "candidate_family_count": float(len(family_counts)),
        "same_family_pair_count": float(same_family_pairs),
        "same_family_pair_ratio": safe_ratio(same_family_pairs, total_pairs),
        "max_family_concentration": safe_ratio(max(family_counts.values(), default=0), len(candidate_types)),
        "candidate_pair_confusion_max": max(confusion_scores, default=0.0),
        "candidate_pair_confusion_mean": mean(confusion_scores),
        "candidate_role_overlap_max": max(role_overlaps, default=0.0),
        "candidate_role_overlap_mean": mean(role_overlaps),
        "candidate_cue_overlap_max": max(cue_overlaps, default=0.0),
        "candidate_cue_overlap_mean": mean(cue_overlaps),
        "cue_matched_type_count": float(cue_matched_types),
        "cue_matched_type_ratio": safe_ratio(cue_matched_types, len(candidate_types)),
        "cue_match_total": float(cue_match_total),
        "cue_token_overlap_total": float(cue_token_overlap_total),
        "core_role_count_max": max(core_role_counts, default=0.0),
        "core_role_count_mean": mean(core_role_counts),
    }


def gold_oracle_features(source_row, schema_by_type, gold_stats):
    events = compact_events(source_row["gold_output"])
    meta = source_row.get("meta") or {}
    candidate_types = [typ for typ in meta.get("candidate_types", []) if typ in schema_by_type]
    event_types = [event.get("event_type") for event in events if event.get("event_type")]
    argument_count = sum(len(event.get("arguments") or []) for event in events)
    trigger_spans = {
        (event.get("event_type"), (event.get("trigger") or {}).get("start"), (event.get("trigger") or {}).get("end"))
        for event in events
        if isinstance(event.get("trigger"), dict)
    }
    confusion_scores = [
        type_confusion_score(event_type, candidate_type, schema_by_type)
        for event_type in event_types
        for candidate_type in candidate_types
        if candidate_type != event_type
    ]
    role_sig_rarity = max(
        [
            log_frequency_rarity(
                gold_stats["role_signature_counts"].get(event_role_signature(event), 0),
                gold_stats["max_signature_freq"],
            )
            for event in events
        ],
        default=0.0,
    )
    missing = 0
    possible = 0
    observed_core = 0
    for event in events:
        event_type = event.get("event_type")
        core_roles = schema_roles(schema_by_type, event_type)
        observed = {arg.get("role") for arg in event.get("arguments", []) if isinstance(arg, dict)}
        possible += len(core_roles)
        observed_core += len(core_roles & observed)
        missing += len(core_roles - observed)
    core_absence = safe_ratio(missing, possible)
    core_density = safe_ratio(observed_core, possible)
    role_density_norm = min(argument_count / 6.0, 1.0)
    multi_event_or_multi_trigger = 1.0 if len(events) > 1 or len(trigger_spans) > 1 else 0.0
    confusion_norm = max(confusion_scores, default=0.0) / 14.0
    hardconf_score = (
        0.35 * confusion_norm
        + 0.20 * role_sig_rarity
        + 0.20 * role_density_norm
        + 0.15 * multi_event_or_multi_trigger
        + 0.10 * core_absence
    )
    return {
        "gold_event_count": float(len(events)),
        "gold_trigger_count": float(len(trigger_spans)),
        "gold_argument_count": float(argument_count),
        "gold_role_density_norm": role_density_norm,
        "gold_core_density": core_density,
        "gold_core_absence_risk": core_absence,
        "gold_multi_event_or_multi_trigger": multi_event_or_multi_trigger,
        "gold_confusion_norm": confusion_norm,
        "gold_role_signature_rarity": role_sig_rarity,
        "gold_hardconf_score": hardconf_score,
    }


def metric_from_label(label, route):
    prefix = f"{route}_mean"
    return {
        "trigger_f1": float(label[f"{prefix}_trigger_f1"]),
        "argument_f1": float(label[f"{prefix}_argument_f1"]),
        "event_f1": float(label[f"{prefix}_event_f1"]),
        "score": float(label[f"{route}_mean_score"]),
    }


def build_rows(split, source_rows, labels, schema_by_type, gold_stats):
    source_by_key = {row_key(row): row for row in source_rows}
    rows = []
    for label in labels:
        key = label["wnd_id"]
        source_row = source_by_key.get(key)
        if source_row is None:
            raise KeyError(f"missing source row for {split} label {key}")
        utility_label = label["utility_label"]
        sampling = {
            "mean_gain": float(label["mean_gain"]),
            "p_win": float(label["p_win"]),
            "p_trigger_noharm": float(label["p_trigger_noharm"]),
            "reason_valid_rate": float(label["reason_valid_rate"]),
            "reason_mean_score": float(label["reason_mean_score"]),
            "direct_mean_score": float(label["direct_mean_score"]),
            "direct_mean_score_neg": -float(label["direct_mean_score"]),
            "stable_reason_indicator": 1.0 if utility_label == "stable_reason" else 0.0,
        }
        rows.append(
            {
                "key": key,
                "split": split,
                "utility_label": utility_label,
                "is_stable_reason": utility_label == "stable_reason",
                "is_stable_direct": utility_label == "stable_direct",
                "direct_metric": metric_from_label(label, "direct"),
                "reason_metric": metric_from_label(label, "reason"),
                "sampled_gain": float(label["mean_gain"]),
                "features": {
                    "deployable": deployable_features(source_row, schema_by_type),
                    "oracle_gold": gold_oracle_features(source_row, schema_by_type, gold_stats),
                    "sampling_oracle": sampling,
                },
            }
        )
    return rows


def normalize(values):
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [0.0 for _ in values]
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) if math.isfinite(value) else 0.0 for value in values]


def selector_specs():
    deployable_keys = [
        "text_token_count",
        "candidate_family_count",
        "same_family_pair_count",
        "same_family_pair_ratio",
        "max_family_concentration",
        "candidate_pair_confusion_max",
        "candidate_pair_confusion_mean",
        "candidate_role_overlap_max",
        "candidate_role_overlap_mean",
        "candidate_cue_overlap_max",
        "candidate_cue_overlap_mean",
        "cue_matched_type_count",
        "cue_matched_type_ratio",
        "cue_match_total",
        "cue_token_overlap_total",
        "core_role_count_max",
        "core_role_count_mean",
    ]
    gold_keys = [
        "gold_event_count",
        "gold_trigger_count",
        "gold_argument_count",
        "gold_role_density_norm",
        "gold_core_density",
        "gold_core_absence_risk",
        "gold_multi_event_or_multi_trigger",
        "gold_confusion_norm",
        "gold_role_signature_rarity",
        "gold_hardconf_score",
    ]
    sampling_keys = [
        "mean_gain",
        "p_win",
        "p_trigger_noharm",
        "reason_valid_rate",
        "reason_mean_score",
        "direct_mean_score",
        "direct_mean_score_neg",
        "stable_reason_indicator",
    ]
    specs = []
    for family, keys in [
        ("deployable", deployable_keys),
        ("oracle_gold", gold_keys),
        ("sampling_oracle", sampling_keys),
    ]:
        for key in keys:
            specs.append({"name": f"{family}:+{key}", "family": family, "terms": [(family, key, 1.0)], "normalize": False})
            specs.append({"name": f"{family}:-{key}", "family": family, "terms": [(family, key, -1.0)], "normalize": False})

    specs.extend(
        [
            {
                "name": "deployable:static_schema_hardness",
                "family": "deployable",
                "terms": [
                    ("deployable", "candidate_pair_confusion_max", 1.0),
                    ("deployable", "same_family_pair_ratio", 0.8),
                    ("deployable", "candidate_role_overlap_max", 0.5),
                    ("deployable", "cue_matched_type_ratio", 0.3),
                ],
                "normalize": True,
            },
            {
                "name": "deployable:hard_but_low_cue",
                "family": "deployable",
                "terms": [
                    ("deployable", "candidate_pair_confusion_max", 1.0),
                    ("deployable", "same_family_pair_ratio", 0.8),
                    ("deployable", "cue_matched_type_ratio", -0.6),
                    ("deployable", "cue_match_total", -0.3),
                ],
                "normalize": True,
            },
            {
                "name": "deployable:long_schema_ambiguous",
                "family": "deployable",
                "terms": [
                    ("deployable", "text_token_count", 0.4),
                    ("deployable", "candidate_pair_confusion_mean", 0.8),
                    ("deployable", "candidate_family_count", 0.4),
                    ("deployable", "core_role_count_mean", 0.4),
                ],
                "normalize": True,
            },
            {
                "name": "oracle_gold:hardconf_plus_density",
                "family": "oracle_gold",
                "terms": [
                    ("oracle_gold", "gold_hardconf_score", 1.0),
                    ("oracle_gold", "gold_argument_count", 0.4),
                    ("oracle_gold", "gold_multi_event_or_multi_trigger", 0.4),
                ],
                "normalize": True,
            },
            {
                "name": "sampling_oracle:stable_reason_label",
                "family": "sampling_oracle",
                "terms": [("sampling_oracle", "stable_reason_indicator", 1.0)],
                "normalize": False,
            },
            {
                "name": "sampling_oracle:stability_rule_score",
                "family": "sampling_oracle",
                "terms": [
                    ("sampling_oracle", "mean_gain", 1.0),
                    ("sampling_oracle", "p_win", 0.8),
                    ("sampling_oracle", "p_trigger_noharm", 0.5),
                    ("sampling_oracle", "direct_mean_score", -0.4),
                ],
                "normalize": True,
            },
        ]
    )
    return specs


def score_selector(rows, spec):
    if spec.get("normalize"):
        normalized_by_term = []
        for family, key, _weight in spec["terms"]:
            normalized_by_term.append(normalize([float(row["features"][family].get(key, 0.0)) for row in rows]))
        scores = {}
        for idx, row in enumerate(rows):
            scores[row["key"]] = sum(weight * normalized_by_term[tidx][idx] for tidx, (_family, _key, weight) in enumerate(spec["terms"]))
        return scores
    scores = {}
    for row in rows:
        value = 0.0
        for family, key, weight in spec["terms"]:
            raw = float(row["features"][family].get(key, 0.0))
            value += weight * raw
        scores[row["key"]] = value
    return scores


def auc(rows, scores, negative_mode):
    positives = [scores[row["key"]] for row in rows if row["is_stable_reason"]]
    if negative_mode == "stable_direct":
        negatives = [scores[row["key"]] for row in rows if row["is_stable_direct"]]
    else:
        negatives = [scores[row["key"]] for row in rows if not row["is_stable_reason"]]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total if total else None


def average_precision(rows, scores):
    positives = {row["key"] for row in rows if row["is_stable_reason"]}
    if not positives:
        return None
    ranked = sorted(rows, key=lambda row: (scores[row["key"]], row["key"]), reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, row in enumerate(ranked, start=1):
        if row["key"] in positives:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(positives)


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_metric(metrics):
    return {key: mean([metric[key] for metric in metrics]) for key in METRIC_KEYS}


def evaluate_selection(rows, selected_keys):
    direct = aggregate_metric([row["direct_metric"] for row in rows])
    routed = aggregate_metric([row["reason_metric"] if row["key"] in selected_keys else row["direct_metric"] for row in rows])
    labels = Counter(row["utility_label"] for row in rows if row["key"] in selected_keys)
    stable_reason = {row["key"] for row in rows if row["is_stable_reason"]}
    tp = len(selected_keys & stable_reason)
    fp = len(selected_keys - stable_reason)
    fn = len(stable_reason - selected_keys)
    selected_rows = [row for row in rows if row["key"] in selected_keys]
    return {
        "selected_count": len(selected_keys),
        "selected_reason_rate": len(selected_keys) / len(rows) if rows else 0.0,
        "selected_labels": dict(labels),
        "route_vs_stable_reason": prf(tp, fp, fn),
        "selected_avg_sampled_gain": mean([row["sampled_gain"] for row in selected_rows]),
        "sampled_expected_direct": direct,
        "sampled_expected_routed": routed,
        "sampled_expected_routed_minus_direct": {key: routed[key] - direct[key] for key in METRIC_KEYS},
    }


def evaluate_selector(rows, spec, budget_name, budget_count):
    scores = score_selector(rows, spec)
    ranked = sorted(rows, key=lambda row: (scores[row["key"]], row["key"]), reverse=True)
    selected_keys = {row["key"] for row in ranked[:budget_count]}
    row = evaluate_selection(rows, selected_keys)
    row.update(
        {
            "selector": spec["name"],
            "family": spec["family"],
            "budget": budget_name,
            "budget_count": budget_count,
            "auc_stable_reason_vs_stable_direct": auc(rows, scores, "stable_direct"),
            "auc_stable_reason_vs_all_other": auc(rows, scores, "all_other"),
            "average_precision_all": average_precision(rows, scores),
        }
    )
    return row


def label_summary(rows):
    counts = Counter(row["utility_label"] for row in rows)
    total = len(rows)
    return {
        "num_examples": total,
        "stable_reason": counts["stable_reason"],
        "stable_direct": counts["stable_direct"],
        "ambiguous": counts["ambiguous"],
        "stable_reason_rate": counts["stable_reason"] / total if total else 0.0,
        "stable_direct_rate": counts["stable_direct"] / total if total else 0.0,
        "ambiguous_rate": counts["ambiguous"] / total if total else 0.0,
    }


def fmt_float(value, digits=3):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def fmt_delta(delta):
    return (
        f"{delta['argument_f1']:+.4f}/"
        f"{delta['event_f1']:+.4f}/"
        f"{delta['trigger_f1']:+.4f}/"
        f"{delta['score']:+.4f}"
    )


def labels_cell(labels):
    return (
        f"R={labels.get('stable_reason', 0)}, "
        f"D={labels.get('stable_direct', 0)}, "
        f"A={labels.get('ambiguous', 0)}"
    )


def best_rows(evals, split, budget, family=None, key="sampled_expected_routed_minus_direct"):
    rows = [row for row in evals[split][budget] if family is None or row["family"] == family]
    return sorted(
        rows,
        key=lambda row: (
            row[key]["score"],
            row["route_vs_stable_reason"]["f1"],
            row["average_precision_all"] or 0.0,
        ),
        reverse=True,
    )


def render_report(payload):
    train_summary = payload["label_summary"]["train"]
    dev_summary = payload["label_summary"]["dev_seen"]
    evals = payload["evaluations"]
    stable_budget = "exact_stable_reason_count"
    dev_best_deployable = best_rows(evals, "dev_seen", stable_budget, "deployable")[:1][0]
    dev_best_gold = best_rows(evals, "dev_seen", stable_budget, "oracle_gold")[:1][0]
    dev_best_sampling = best_rows(evals, "dev_seen", stable_budget, "sampling_oracle")[:1][0]

    lines = [
        "# Sampled K8 Signal Separability Diagnostic",
        "",
        "This report asks whether K=8 stable Reason labels are separable from information visible before execution. It compares deployable raw-prompt/schema features, gold-only oracle features, and sampled-outcome oracle features.",
        "",
        "## Label Summary",
        "",
        "| split | n | stable_reason | stable_direct | ambiguous | stable_reason rate |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| train | {train_summary['num_examples']} | {train_summary['stable_reason']} | "
            f"{train_summary['stable_direct']} | {train_summary['ambiguous']} | {train_summary['stable_reason_rate']:.1%} |"
        ),
        (
            f"| dev_seen | {dev_summary['num_examples']} | {dev_summary['stable_reason']} | "
            f"{dev_summary['stable_direct']} | {dev_summary['ambiguous']} | {dev_summary['stable_reason_rate']:.1%} |"
        ),
        "",
        "## Main Reading",
        "",
        (
            f"- Best deployable dev selector at the stable-reason budget: `{dev_best_deployable['selector']}`; "
            f"selected {labels_cell(dev_best_deployable['selected_labels'])}; "
            f"P/R/F1 `{dev_best_deployable['route_vs_stable_reason']['precision']:.3f}/"
            f"{dev_best_deployable['route_vs_stable_reason']['recall']:.3f}/"
            f"{dev_best_deployable['route_vs_stable_reason']['f1']:.3f}`; "
            f"sampled routed-minus-direct A/E/T/Score `{fmt_delta(dev_best_deployable['sampled_expected_routed_minus_direct'])}`."
        ),
        (
            f"- Best gold-oracle dev selector: `{dev_best_gold['selector']}`; "
            f"selected {labels_cell(dev_best_gold['selected_labels'])}; "
            f"P/R/F1 `{dev_best_gold['route_vs_stable_reason']['precision']:.3f}/"
            f"{dev_best_gold['route_vs_stable_reason']['recall']:.3f}/"
            f"{dev_best_gold['route_vs_stable_reason']['f1']:.3f}`; "
            f"score delta `{dev_best_gold['sampled_expected_routed_minus_direct']['score']:+.4f}`."
        ),
        (
            f"- Sampled-outcome oracle sanity: `{dev_best_sampling['selector']}`; "
            f"selected {labels_cell(dev_best_sampling['selected_labels'])}; "
            f"P/R/F1 `{dev_best_sampling['route_vs_stable_reason']['precision']:.3f}/"
            f"{dev_best_sampling['route_vs_stable_reason']['recall']:.3f}/"
            f"{dev_best_sampling['route_vs_stable_reason']['f1']:.3f}`; "
            f"score delta `{dev_best_sampling['sampled_expected_routed_minus_direct']['score']:+.4f}`."
        ),
        "",
        "Interpretation: deployable features are a lower bound on what a raw route prompt can learn, while sampled-outcome features are the label oracle. If the gap between them is large, the next selector should expose stronger intermediate evidence instead of only asking the model to emit a route tag from the raw prompt.",
        "",
        "## Dev Selectors",
        "",
        "Budget here is exactly the dev stable_reason count, so `15/197` examples are routed to Reason.",
        "",
        "| family | selector | AUC vs stable_direct | AP | P/R/F1 | selected labels | delta A/E/T/Score | avg gain |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    top_rows = []
    for family in ["deployable", "oracle_gold", "sampling_oracle"]:
        top_rows.extend(best_rows(evals, "dev_seen", stable_budget, family)[:8])
    for row in top_rows:
        prf_row = row["route_vs_stable_reason"]
        lines.append(
            f"| {row['family']} | `{row['selector']}` | "
            f"{fmt_float(row['auc_stable_reason_vs_stable_direct'])} | "
            f"{fmt_float(row['average_precision_all'])} | "
            f"{prf_row['precision']:.3f}/{prf_row['recall']:.3f}/{prf_row['f1']:.3f} | "
            f"{labels_cell(row['selected_labels'])} | "
            f"{fmt_delta(row['sampled_expected_routed_minus_direct'])} | "
            f"{row['selected_avg_sampled_gain']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Train-Selected Transfer",
            "",
            "Each row chooses the best selector on train at the exact stable-reason budget, then reports that same selector on dev_seen.",
            "",
            "| family | train-selected selector | train P/R/F1 | dev P/R/F1 | dev selected labels | dev delta A/E/T/Score |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for family in ["deployable", "oracle_gold", "sampling_oracle"]:
        train_best = best_rows(evals, "train", stable_budget, family)[:1][0]
        dev_match = [
            row for row in evals["dev_seen"][stable_budget]
            if row["selector"] == train_best["selector"]
        ][0]
        train_prf = train_best["route_vs_stable_reason"]
        dev_prf = dev_match["route_vs_stable_reason"]
        lines.append(
            f"| {family} | `{train_best['selector']}` | "
            f"{train_prf['precision']:.3f}/{train_prf['recall']:.3f}/{train_prf['f1']:.3f} | "
            f"{dev_prf['precision']:.3f}/{dev_prf['recall']:.3f}/{dev_prf['f1']:.3f} | "
            f"{labels_cell(dev_match['selected_labels'])} | "
            f"{fmt_delta(dev_match['sampled_expected_routed_minus_direct'])} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{payload['output_json']}`",
            f"- Train labels: `{payload['inputs']['train_labels']}`",
            f"- Dev labels: `{payload['inputs']['dev_seen_labels']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args):
    schema_by_type = load_schema(SCHEMA_PATH)
    source = {}
    labels = {}
    for split in ["train", "dev_seen"]:
        source_path = REPO / f"data/stage2_adaptive_datasets/{DATA_PREFIX}_{SOURCE_BRANCH}_{split}_pos.jsonl"
        label_path = REPO / f"data/stage2_adaptive_datasets/labels/{DATA_PREFIX}_{LABEL_SOURCE}_{split}_labels.jsonl"
        source[split] = load_jsonl(source_path)
        labels[split] = load_jsonl(label_path)

    gold_stats = build_gold_stats(source["train"])
    rows = {
        split: build_rows(split, source[split], labels[split], schema_by_type, gold_stats)
        for split in ["train", "dev_seen"]
    }

    specs = selector_specs()
    budgets = {
        split: {
            "exact_stable_reason_count": sum(1 for row in split_rows if row["is_stable_reason"]),
            "cap05": round(len(split_rows) * 0.05),
            "cap10": round(len(split_rows) * 0.10),
            "cap15": round(len(split_rows) * 0.15),
            "cap20": round(len(split_rows) * 0.20),
        }
        for split, split_rows in rows.items()
    }
    evaluations = defaultdict(dict)
    for split, split_rows in rows.items():
        for budget_name, budget_count in budgets[split].items():
            evaluations[split][budget_name] = [
                evaluate_selector(split_rows, spec, budget_name, budget_count)
                for spec in specs
            ]

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    payload = {
        "inputs": {
            "train_labels": (REPO / f"data/stage2_adaptive_datasets/labels/{DATA_PREFIX}_{LABEL_SOURCE}_train_labels.jsonl").as_posix(),
            "dev_seen_labels": (REPO / f"data/stage2_adaptive_datasets/labels/{DATA_PREFIX}_{LABEL_SOURCE}_dev_seen_labels.jsonl").as_posix(),
            "schema": SCHEMA_PATH.as_posix(),
        },
        "label_summary": {split: label_summary(split_rows) for split, split_rows in rows.items()},
        "budgets": budgets,
        "evaluations": evaluations,
        "output_json": output_json.as_posix(),
        "output_md": output_md.as_posix(),
    }
    write_json(output_json, payload)
    write_text(output_md, render_report(payload))
    print(json.dumps({"output_json": output_json.as_posix(), "output_md": output_md.as_posix()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_json",
        default="reports/artifacts/2026-05-18_stage2_sampled_k8_signal_separability_checkpoint258.json",
    )
    parser.add_argument(
        "--output_md",
        default="reports/2026-05-18_stage2_sampled_k8_signal_separability_checkpoint258.md",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
