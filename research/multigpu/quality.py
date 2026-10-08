"""Independent prediction comparison for research artifacts (NumPy only)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# Screening thresholds, not promises about acceptable application degradation.
TOLERANCES = {
    "float64": (1e-9, 1e-7),
    "float32": (1e-5, 1e-4),
    "float16": (2e-3, 1e-2),
    "bfloat16": (1e-2, 5e-2),
}


def numerical_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    compute_dtype: str,
) -> dict[str, Any]:
    """Compare arrays after the caller aligns observation identities."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or not a.size:
        raise ValueError("Predictions must have the same nonempty shape")
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    result: dict[str, Any] = {
        "shape": list(a.shape),
        "reference_dtype": str(a.dtype),
        "candidate_dtype": str(b.dtype),
        "finite": finite,
        "bitwise_equal": bool(
            a.dtype == b.dtype and a.tobytes() == b.tobytes()
        ),
    }
    atol, rtol = TOLERANCES[compute_dtype]
    result.update(atol=atol, rtol=rtol, compute_dtype=compute_dtype)
    if not finite:
        result["within_tolerance"] = False
        return result
    a, b = a.astype(np.float64), b.astype(np.float64)
    error = np.abs(a - b)
    failed = error > atol + rtol * np.abs(a)
    result.update(
        max_abs=float(error.max()),
        mean_abs=float(error.mean()),
        rms=float(np.sqrt(np.square(error).mean())),
        p99_abs=float(np.quantile(error, 0.99)),
        relative_l2=float(
            np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)
        ),
        within_tolerance=bool(not failed.any()),
        failing_entries=int(failed.sum()),
        rows_with_failures=int(
            failed.reshape(a.shape[0] if a.ndim else 1, -1).any(axis=1).sum()
        ),
    )
    return result


def compare_prediction_repeats(
    reference: np.ndarray,
    candidates: Sequence[np.ndarray],
    *,
    compute_dtype: str,
) -> dict[str, Any]:
    """Compare every saved repeat without substituting a favorable repeat.

    The caller must first validate row identities and output-column semantics.
    This function never aligns, normalizes, sorts or overwrites predictions.
    """
    if not candidates:
        raise ValueError(
            "At least one candidate prediction repeat is required"
        )
    against_reference = [
        numerical_comparison(reference, value, compute_dtype=compute_dtype)
        for value in candidates
    ]
    against_first = [
        numerical_comparison(candidates[0], value, compute_dtype=compute_dtype)
        for value in candidates
    ]
    return {
        "repeat_count": len(candidates),
        "all_repeats_within_tolerance": all(
            value["within_tolerance"] for value in against_reference
        ),
        "all_repeats_bitwise_equal_reference": all(
            value["bitwise_equal"] for value in against_reference
        ),
        "bitwise_repeatable": all(
            value["bitwise_equal"] for value in against_first
        ),
        "against_reference": against_reference,
        "against_first_candidate": against_first,
    }


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Mann-Whitney AUROC with average ranks for tied scores."""
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    values = scores[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.repeat((starts + ends + 1) / 2, ends - starts)
    return float(
        (ranks[labels[order]].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: Sequence[Any],
) -> dict[str, Any]:
    """Score validation probabilities with explicit column labels."""
    y, p = (
        np.asarray(labels).reshape(-1),
        np.asarray(probabilities, dtype=np.float64),
    )
    if p.shape != (len(y), len(classes)) or len(set(classes)) != len(classes):
        raise ValueError(
            "Probability shape/class labels must match observations"
        )
    index = {value: i for i, value in enumerate(classes)}
    try:
        target = np.asarray([index[value] for value in y])
    except KeyError as exc:
        raise ValueError(
            "A validation label is absent from the fitted classes"
        ) from exc
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Expected finite probabilities in [0, 1]")
    if not np.allclose(p.sum(-1), 1, atol=1e-4, rtol=1e-4):
        raise ValueError("Probability columns must sum to one")
    one_hot = np.eye(len(classes))[target]
    aucs = [binary_auroc(target == i, p[:, i]) for i in range(len(classes))]
    present_aucs = [auc for auc in aucs if auc is not None]
    return {
        "rows": len(y),
        "accuracy": float(np.mean(p.argmax(-1) == target)),
        "log_loss": float(
            -np.log(np.clip(p[np.arange(len(y)), target], 1e-15, 1)).mean()
        ),
        "brier_multiclass": float(np.square(p - one_hot).sum(-1).mean()),
        "auroc_by_class": aucs,
        "auroc_macro_present_classes": float(np.mean(present_aucs))
        if present_aucs
        else None,
        "max_probability_sum_error": float(np.abs(p.sum(-1) - 1).max()),
    }


def regression_metrics(
    labels: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """Score predictions in original target units after inverse transforms."""
    y, p = np.asarray(labels).reshape(-1), np.asarray(prediction).reshape(-1)
    if (
        y.shape != p.shape
        or not y.size
        or not np.isfinite(p).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Expected aligned nonempty finite regression arrays")
    error = y.astype(np.float64) - p.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mae": float(np.abs(error).mean()),
    }


def quantile_metrics(
    labels: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    """Audit q001..q999 outputs without sorting away quantile crossings."""
    p = np.asarray(prediction)
    y = np.asarray(labels).reshape(-1)
    if p.shape != (len(y), 999) or not np.isfinite(p).all():
        raise ValueError("Expected finite [rows, 999] q001..q999 predictions")
    gaps = np.diff(p.astype(np.float64), axis=1)
    levels = np.arange(1, 1000) / 1000
    errors = y[:, None] - p
    return {
        **regression_metrics(y, p[:, 499]),
        "crossing_pairs": int((gaps < 0).sum()),
        "rows_with_crossings": int((gaps < 0).any(axis=1).sum()),
        "max_crossing": float(max(0, -gaps.min())),
        "mean_pinball_loss": float(
            np.maximum(levels * errors, (levels - 1) * errors).mean()
        ),
        "coverage_q050_q950": float(
            ((y >= p[:, 49]) & (y <= p[:, 949])).mean()
        ),
    }


def paired_loss_interval(
    reference_loss: np.ndarray,
    candidate_loss: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    repetitions: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Paired bootstrap of candidate-minus-reference mean loss.

    Use entity IDs as groups for repeated relational observations; this samples
    entities with replacement and includes all their rows on each draw.
    """
    a, b = (
        np.asarray(reference_loss).reshape(-1),
        np.asarray(candidate_loss).reshape(-1),
    )
    if (
        a.shape != b.shape
        or not a.size
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Expected aligned finite loss arrays")
    groups = (
        np.arange(len(a)) if groups is None else np.asarray(groups).reshape(-1)
    )
    if len(groups) != len(a):
        raise ValueError("One group is required per observation")
    _, inverse = np.unique(groups, return_inverse=True)
    totals = np.bincount(inverse, weights=b - a)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for i in range(repetitions):
        sample = rng.integers(len(counts), size=len(counts))
        draws[i] = totals[sample].sum() / counts[sample].sum()
    return {
        "candidate_minus_reference": float((b - a).mean()),
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "groups": len(counts),
        "repetitions": repetitions,
        "seed": seed,
    }
