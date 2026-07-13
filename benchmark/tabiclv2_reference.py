"""Benchmark equivalent stages in pinned open-source TabICLv2."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tabicl.sklearn.base import TabICLBaseEstimator
from tabicl.sklearn.preprocessing import (
    EnsembleGenerator,
    TransformToNumerical,
)

from benchmark.tabiclv2_processing import (
    SIZES,
    BenchmarkResult,
    Characteristics,
    Task,
    _measure,
    build_workload,
    write_results,
)

LOGGER = logging.getLogger(__name__)


def _arrays(size: str, task: Task):
    workload = build_workload(
        size=size,
        task=task,
        characteristics=Characteristics(),
    )
    features = workload.x.numerical.numpy()
    train = features[: workload.train_rows]
    query = features[workload.train_rows :]
    if task == "classification":
        target = np.asarray(["negative", "positive"])[
            workload.y.categorical.as_tensor().squeeze(-1).numpy()
        ]
    else:
        target = workload.y.numerical.squeeze(-1).numpy()
    return workload, train, query, target


def _preprocess_operation(
    train: np.ndarray,
    query: np.ndarray,
    target: np.ndarray,
    task: Task,
):
    def operation():
        encoder = TransformToNumerical().fit(train)
        train_encoded = encoder.transform(train)
        query_encoded = encoder.transform(query)
        if task == "classification":
            target_encoded = LabelEncoder().fit_transform(target)
        else:
            target_encoded = (
                StandardScaler()
                .fit_transform(target.reshape(-1, 1))
                .reshape(-1)
            )
        ensemble = EnsembleGenerator(
            classification=task == "classification",
            n_estimators=1,
            norm_methods=["none"],
            feat_shuffle_method="none",
            class_shuffle_method="none",
            random_state=0,
        ).fit(train_encoded, target_encoded)
        return ensemble.transform(query_encoded, mode="both")

    return operation


def _mapping_setup(
    task: Task,
    target: np.ndarray,
    query_rows: int,
):
    if task == "classification":
        raw = np.zeros((1, query_rows, 2), dtype=np.float32)
        permutation = np.arange(2)

        def mapping():
            return raw[0][..., permutation]

        return raw, mapping

    scaler = StandardScaler().fit(target.reshape(-1, 1))
    raw = np.zeros((1, query_rows, 999), dtype=np.float32)

    def mapping():
        canonical = scaler.inverse_transform(raw.reshape(-1, 1)).reshape(
            raw.shape
        )
        return canonical.mean(axis=0)

    return raw, mapping


def _output_operation(task: Task, canonical: np.ndarray):
    if task == "classification":

        def operation():
            probabilities = TabICLBaseEstimator.softmax(
                canonical,
                axis=-1,
                temperature=0.9,
            )
            return probabilities / probabilities.sum(axis=1, keepdims=True)

        return operation

    return lambda: canonical


def _result(workload, operation, repetitions, timing):
    median, p95, deviation, peak = timing
    return BenchmarkResult(
        operation=operation,
        task_type=workload.task,
        processor_or_recipe="Pinned TabICLv2",
        size=workload.name,
        row_count=workload.rows,
        feature_count=workload.features,
        train_row_count=workload.train_rows,
        dataset_characteristics="baseline",
        characteristics={
            "constant": False,
            "hard_outlier": False,
            "sigma_outlier": False,
            "categorical": False,
            "missing": False,
            "unknown_category": False,
        },
        median_ms=median,
        p95_ms=p95,
        standard_deviation_ms=deviation,
        peak_memory_bytes=peak,
        device="cpu",
        gpu_model=None,
        dtype="numpy.float32",
        repetitions=repetitions,
        correctness_status="pass",
    )


def benchmark_reference(
    *,
    size: str,
    task: Task,
    repetitions: int,
) -> list[BenchmarkResult]:
    """Benchmark equivalent semantic stages in the pinned reference."""
    workload, train, query, target = _arrays(size, task)
    query_rows = query.shape[0]
    preprocessing = _preprocess_operation(train, query, target, task)
    preprocessed = preprocessing()
    assert next(iter(preprocessed.values()))[0].shape == (
        1,
        workload.rows,
        workload.features,
    )

    _, mapping = _mapping_setup(task, target, query_rows)
    canonical = mapping()
    output = _output_operation(task, canonical)
    prediction = output()
    expected_width = 2 if task == "classification" else 999
    assert prediction.shape == (query_rows, expected_width)
    assert np.isfinite(prediction).all()

    results = []
    timing = _measure(lambda: preprocessing, repetitions=repetitions)
    results.append(
        _result(workload, "preprocessing_total", repetitions, timing)
    )
    timing = _measure(lambda: mapping, repetitions=repetitions)
    results.append(
        _result(
            workload,
            "model_output_inverse_mapping",
            repetitions,
            timing,
        )
    )
    timing = _measure(lambda: output, repetitions=repetitions)
    results.append(_result(workload, "output_transform", repetitions, timing))

    def total_prepare():
        preprocess = _preprocess_operation(train, query, target, task)

        def operation():
            preprocess()
            _, map_output = _mapping_setup(task, target, query_rows)
            transformed = map_output()
            _output_operation(task, transformed)()

        return operation

    timing = _measure(total_prepare, repetitions=repetitions)
    results.append(
        _result(workload, "total_recipe_overhead", repetitions, timing)
    )
    return results


def main() -> None:
    """Run the pinned reference benchmark command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=tuple(SIZES), default="large")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmark/results/tabiclv2_reference_large_baseline.json"
        ),
    )
    args = parser.parse_args()
    results = []
    for task in ("classification", "regression"):
        LOGGER.info(
            "benchmarking pinned reference task=%s size=%s",
            task,
            args.size,
        )
        results.extend(
            benchmark_reference(
                size=args.size,
                task=task,
                repetitions=args.repetitions,
            )
        )
    write_results(results, args.output)
    LOGGER.info("wrote %d results to %s", len(results), args.output)


if __name__ == "__main__":
    main()
