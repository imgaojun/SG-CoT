"""Training-only transfer balancing for category-isolated preference updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linprog


CATEGORIES = (
    "argument_omission",
    "event_omission",
    "extra_frame",
    "trigger_drift",
    "wrong_type",
)


def _as_matrix(
    values: Mapping[str, Mapping[str, float]],
    experts: Sequence[str],
    categories: Sequence[str],
) -> np.ndarray:
    if set(values) != set(experts):
        raise ValueError("transfer rows do not match experts")
    matrix = []
    for expert in experts:
        row = values[expert]
        if set(row) != set(categories):
            raise ValueError(f"transfer columns do not match categories: {expert}")
        matrix.append([float(row[category]) for category in categories])
    result = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("transfer matrix contains a non-finite value")
    return result


def solve_maximin_weights(
    masked_values: Mapping[str, Mapping[str, float]],
    full_values: Mapping[str, Mapping[str, float]],
    *,
    experts: Sequence[str] = CATEGORIES,
    categories: Sequence[str] = CATEGORIES,
    full_floor: float = 0.0,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Maximize the worst masked transfer while keeping full transfer nonnegative."""

    experts = tuple(experts)
    categories = tuple(categories)
    if not experts or not categories:
        raise ValueError("experts and categories must be nonempty")
    masked = _as_matrix(masked_values, experts, categories)
    full = _as_matrix(full_values, experts, categories)
    count = len(experts)

    # Variables are category weights followed by the minimum masked transfer t.
    objective = np.zeros(count + 1, dtype=np.float64)
    objective[-1] = -1.0
    masked_constraints = np.hstack((-masked.T, np.ones((len(categories), 1))))
    full_constraints = np.hstack((-full.T, np.zeros((len(categories), 1))))
    upper = np.concatenate(
        (masked_constraints, full_constraints), axis=0
    )
    bounds = [(0.0, 1.0)] * count + [(None, None)]
    first = linprog(
        objective,
        A_ub=upper,
        b_ub=np.concatenate(
            (np.zeros(len(categories)), -np.full(len(categories), full_floor))
        ),
        A_eq=np.asarray([[1.0] * count + [0.0]], dtype=np.float64),
        b_eq=np.asarray([1.0], dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if not first.success:
        raise ValueError(f"maximin transfer problem is infeasible: {first.message}")

    optimum = float(first.x[-1])
    # Resolve equivalent optima deterministically without materially relaxing t.
    secondary_objective = np.asarray(
        [float(index + 1) for index in range(count)] + [0.0], dtype=np.float64
    )
    secondary_upper = np.vstack(
        (
            upper,
            np.asarray([0.0] * count + [-1.0], dtype=np.float64),
        )
    )
    secondary_b = np.concatenate(
        (
            np.zeros(len(categories)),
            -np.full(len(categories), full_floor),
            np.asarray([-(optimum - tolerance)]),
        )
    )
    second = linprog(
        secondary_objective,
        A_ub=secondary_upper,
        b_ub=secondary_b,
        A_eq=np.asarray([[1.0] * count + [0.0]], dtype=np.float64),
        b_eq=np.asarray([1.0], dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if not second.success:
        raise ValueError(f"deterministic tie-break failed: {second.message}")

    weights_array = second.x[:count]
    weights_array[np.abs(weights_array) < tolerance] = 0.0
    weights_array /= weights_array.sum()
    predicted_masked = weights_array @ masked
    predicted_full = weights_array @ full
    if predicted_masked.min() < optimum - 10 * tolerance:
        raise AssertionError("tie-break degraded the maximin optimum")
    if predicted_full.min() < full_floor - 10 * tolerance:
        raise AssertionError("tie-break violated the full-response floor")
    return {
        "weights": {
            expert: float(weight)
            for expert, weight in zip(experts, weights_array, strict=True)
        },
        "maximin_masked_margin_delta": float(predicted_masked.min()),
        "predicted_masked_margin_deltas": {
            category: float(value)
            for category, value in zip(categories, predicted_masked, strict=True)
        },
        "predicted_full_response_margin_deltas": {
            category: float(value)
            for category, value in zip(categories, predicted_full, strict=True)
        },
        "solver": "scipy.optimize.linprog(method=highs), lexicographic tie-break",
        "full_floor": float(full_floor),
    }


def combine_tensor(base: Any, experts: Sequence[Any], weights: Sequence[float], scale: float) -> Any:
    """Combine floating tensors as a scaled convex sum of expert deltas."""

    if len(experts) != len(weights) or not experts:
        raise ValueError("expert tensors and weights must have equal nonzero length")
    if abs(sum(float(weight) for weight in weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to one")
    if any(float(weight) < 0.0 for weight in weights):
        raise ValueError("weights must be nonnegative")
    if not getattr(base.dtype, "is_floating_point", False):
        if any(not expert.equal(base) for expert in experts):
            raise ValueError("non-floating expert tensor differs from base")
        return base.clone()
    base_float = base.float()
    merged = base_float.clone()
    for expert, weight in zip(experts, weights, strict=True):
        if expert.shape != base.shape:
            raise ValueError("expert tensor shape differs from base")
        merged.add_(expert.float() - base_float, alpha=float(scale) * float(weight))
    return merged.to(dtype=base.dtype)
