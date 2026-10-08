# ruff: noqa: D103, TID253
import numpy as np
import pytest
from research.multigpu.quality import (
    binary_auroc,
    classification_metrics,
    compare_prediction_repeats,
    numerical_comparison,
    paired_loss_interval,
    quantile_metrics,
    regression_metrics,
)


def test_auc_ties_and_degenerate_labels() -> None:
    assert binary_auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    assert binary_auroc(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1])) == 1
    assert binary_auroc(np.array([0, 1]), np.array([1, 0])) == 0
    assert binary_auroc(np.array([1, 1]), np.array([0, 1])) is None


def test_class_order_is_explicit() -> None:
    metrics = classification_metrics(
        np.array([4, 9]), np.array([[0.1, 0.9], [0.8, 0.2]]), classes=[9, 4]
    )
    assert metrics["accuracy"] == 1
    assert metrics["auroc_by_class"] == [1, 1]
    assert metrics["log_loss"] == pytest.approx(-np.log([0.9, 0.8]).mean())
    with pytest.raises(ValueError, match="absent"):
        classification_metrics(
            np.array([4, 8]),
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            classes=[9, 4],
        )


def test_probabilities_are_not_silently_normalized() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        classification_metrics(
            np.array([0]), np.array([[0.1, 0.2]]), classes=[0, 1]
        )


def test_numerical_nonfinite_and_bitwise() -> None:
    a = np.array([0, 1], dtype=np.float32)
    assert numerical_comparison(a, a.copy(), compute_dtype="float32")[
        "bitwise_equal"
    ]
    assert not numerical_comparison(
        a, np.array([np.nan, 1]), compute_dtype="float32"
    )["within_tolerance"]
    b = a.copy()
    b[0] = -0.0
    result = numerical_comparison(a, b, compute_dtype="float32")
    assert result["within_tolerance"]
    assert not result["bitwise_equal"]


def test_regression_original_units_and_bootstrap_pairing() -> None:
    assert regression_metrics(np.array([1, 3]), np.array([2, 2])) == {
        "rmse": 1,
        "mae": 1,
    }
    result = paired_loss_interval(
        np.array([1, 3, 5]),
        np.array([2, 4, 6]),
        groups=np.array([0, 0, 1]),
        repetitions=30,
    )
    assert result["candidate_minus_reference"] == 1
    assert result["ci95"] == [1, 1]
    assert result["groups"] == 2


def test_quantiles_preserve_crossings_and_choose_q500() -> None:
    p = np.tile(np.arange(1, 1000), (2, 1)).astype(float)
    result = quantile_metrics(np.array([500, 500]), p)
    assert result["rmse"] == 0
    assert result["crossing_pairs"] == 0
    p[0, 500] = 498
    result = quantile_metrics(np.array([500, 500]), p)
    assert result["crossing_pairs"] == 1
    assert result["max_crossing"] == 2


def test_repeat_audit_does_not_hide_later_quantile_failure() -> None:
    reference = np.zeros((2, 999), dtype=np.float32)
    later = reference.copy()
    later[1, 0] = 0.02
    report = compare_prediction_repeats(
        reference, [reference.copy(), later], compute_dtype="bfloat16"
    )
    assert not report["all_repeats_within_tolerance"]
    assert not report["bitwise_repeatable"]
    assert report["against_reference"][0]["bitwise_equal"]
    assert report["against_reference"][1]["failing_entries"] == 1
    assert report["against_reference"][1]["rows_with_failures"] == 1
    assert np.count_nonzero(reference) == 0
    assert later[1, 0] == np.float32(0.02)


def test_repeat_audit_rejects_empty_and_retains_nonfinite() -> None:
    reference = np.zeros((2, 7))
    with pytest.raises(ValueError, match="At least one"):
        compare_prediction_repeats(reference, [], compute_dtype="float32")
    bad = reference.copy()
    bad[0, 0] = np.nan
    report = compare_prediction_repeats(
        reference, [bad], compute_dtype="float32"
    )
    assert not report["all_repeats_within_tolerance"]
    assert not report["against_reference"][0]["finite"]
