"""Minimal error-correcting output codes for benchmark models."""

import math

import numpy as np


def many_class_estimator_count(
    n_classes: int,
    alphabet_size: int,
    requested: int | None = None,
    redundancy: int = 4,
) -> int:
    """Resolve the number of error-correcting code rows."""
    minimum = math.ceil(n_classes / (alphabet_size - 1))
    if requested is not None:
        count = requested
    else:
        if redundancy < 1:
            raise ValueError("redundancy must be positive")
        log_cover = math.ceil(math.log(max(n_classes, 2), alphabet_size))
        base = max(minimum, log_cover)
        count = max(
            base,
            min(base * redundancy, max(base, 4 * log_cover)),
        )
    if count < minimum:
        raise ValueError(
            f"n_estimators must be at least {minimum} to cover "
            f"{n_classes} classes with alphabet size {alphabet_size}"
        )
    return count


def many_class_codebook(
    n_classes: int,
    alphabet_size: int,
    n_estimators: int,
    random_state: int | None,
) -> np.ndarray:
    """Build a rest-symbol codebook with balanced class coverage."""
    rng = np.random.RandomState(random_state)
    rest = alphabet_size - 1
    best: np.ndarray | None = None
    best_score = (-1, -1.0)

    for _ in range(50):
        codebook = np.full(
            (n_estimators, n_classes),
            rest,
            dtype=np.int64,
        )
        coverage = np.zeros(n_classes, dtype=np.int64)
        for row in range(n_estimators):
            priority = coverage + rng.uniform(0.0, 0.1, size=n_classes)
            chosen = np.argsort(priority)[: min(rest, n_classes)]
            codebook[row, chosen] = rng.permutation(rest)[: len(chosen)]
            coverage[chosen] += 1
        if np.any(coverage == 0):
            continue

        distances = (codebook.T[:, None] != codebook.T[None, :]).sum(axis=2)
        pairwise = distances[np.triu_indices(n_classes, k=1)]
        score = (int(pairwise.min()), float(pairwise.mean()))
        if score > best_score:
            best = codebook
            best_score = score

    if best is None:
        raise RuntimeError("Failed to generate a many-class codebook")
    return best


def align_symbol_probabilities(
    probabilities: np.ndarray,
    symbols: np.ndarray,
    alphabet_size: int,
) -> np.ndarray:
    """Insert zero probability columns for symbols that are absent."""
    aligned = np.zeros(
        (probabilities.shape[0], alphabet_size),
        dtype=np.float64,
    )
    aligned[:, symbols] = probabilities
    totals = aligned.sum(axis=1, keepdims=True)
    if (
        not np.isfinite(aligned).all()
        or np.any(aligned < 0)
        or not np.isfinite(totals).all()
        or np.any(totals <= 0)
    ):
        raise RuntimeError("ECOC symbol probabilities are invalid")
    aligned /= totals
    return aligned


def decode_many_class_probabilities(
    probabilities: np.ndarray,
    codebook: np.ndarray,
) -> np.ndarray:
    """Decode code-row probabilities into the original class space."""
    rest = probabilities.shape[2] - 1
    selected = np.take_along_axis(
        probabilities,
        codebook[:, None, :],
        axis=2,
    )
    active = codebook != rest
    coverage = active.sum(axis=0)
    if np.any(coverage == 0):
        raise RuntimeError("The many-class codebook leaves a class uncovered")

    log_probabilities = np.log(np.clip(selected, 1e-12, 1.0))
    scores = np.where(
        active[:, None, :],
        log_probabilities,
        0.0,
    ).sum(axis=0)
    scores /= coverage[None, :]
    scores -= scores.max(axis=1, keepdims=True)
    decoded = np.exp(scores)
    return decoded / decoded.sum(axis=1, keepdims=True)
