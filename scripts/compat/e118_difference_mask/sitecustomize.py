"""Install E118 difference masking into LLaMAFactory's pairwise collator."""

from __future__ import annotations

import json
import os

from llamafactory.data.collator import PairwiseDataCollatorWithPadding

from src.stage2_preference.difference_masking import mask_pair_labels


if not getattr(PairwiseDataCollatorWithPadding, "_e118_difference_masked", False):
    _original_call = PairwiseDataCollatorWithPadding.__call__
    _runtime_log_written = False

    def _difference_masked_call(self, features):
        global _runtime_log_written
        context_tokens = int(os.environ.get("E118_DIFF_CONTEXT_TOKENS", "1"))
        masked_features = []
        batch_statistics = []
        for feature in features:
            masked_feature = dict(feature)
            chosen, rejected, statistics = mask_pair_labels(
                list(feature["chosen_labels"]),
                list(feature["rejected_labels"]),
                ignore_index=-100,
                context_tokens=context_tokens,
            )
            masked_feature["chosen_labels"] = chosen
            masked_feature["rejected_labels"] = rejected
            masked_features.append(masked_feature)
            batch_statistics.append(statistics)
        if not _runtime_log_written:
            summary = {
                "event": "e118_difference_mask_active",
                "batch_pairs": len(batch_statistics),
                "context_tokens": context_tokens,
                "chosen_kept_tokens": sum(
                    item["chosen_kept_tokens"] for item in batch_statistics
                ),
                "chosen_response_tokens": sum(
                    item["chosen_response_tokens"] for item in batch_statistics
                ),
                "rejected_kept_tokens": sum(
                    item["rejected_kept_tokens"] for item in batch_statistics
                ),
                "rejected_response_tokens": sum(
                    item["rejected_response_tokens"] for item in batch_statistics
                ),
            }
            print(json.dumps(summary, sort_keys=True), flush=True)
            _runtime_log_written = True
        return _original_call(self, masked_features)

    PairwiseDataCollatorWithPadding.__call__ = _difference_masked_call
    PairwiseDataCollatorWithPadding._e118_difference_masked = True
