# ruff: noqa: D103, TID253
"""Research metrics retain model classes absent from validation cohorts."""

import numpy as np
import pytest
import torch
from research.multigpu.relational_bench import score

import sdm


def tables(probabilities, columns, targets):
    return (
        sdm.TableTensor.from_tensor(
            torch.tensor(probabilities, dtype=torch.float32), columns=columns
        ),
        sdm.TableTensor.from_columns(
            {"target": targets}, stypes={"target": "categorical"}
        ),
    )


@pytest.mark.parametrize("observed", [[4, 4], [4, 2]])
@pytest.mark.parametrize("string_labels", [False, True])
def test_seven_class_support_survives_small_cohorts(observed, string_labels):
    classes = [7, 1, 4, 2, 6, 3, 5]
    names = [
        f"class-{value}" if string_labels else str(value) for value in classes
    ]
    targets = (
        [f"class-{value}" for value in observed] if string_labels else observed
    )
    values = np.full((2, 7), 0.1, dtype=np.float32)
    values[:, 0] = (
        0.4  # Unobserved class wins; dropping it would inflate accuracy.
    )
    pred, target = tables(values, names, targets)
    metrics = score(pred, target, "classification")
    assert metrics["accuracy"] == 0
    assert metrics["log_loss"] == pytest.approx(-np.log(0.1))
    assert "auroc" not in metrics
    assert "positive_class" not in metrics


@pytest.mark.parametrize("observed", [0, 1])
def test_single_class_binary_auc_is_explicitly_undefined(observed):
    pred, target = tables([[0.8, 0.2]], ["1", "0"], [observed])
    metrics = score(pred, target, "classification")
    assert metrics["positive_class"] == 1
    assert metrics["auroc"] is None
    assert metrics["auroc_undefined_reason"] == "validation_contains_one_class"
    assert metrics["log_loss"] == pytest.approx(
        -np.log(0.8 if observed == 1 else 0.2)
    )


def test_named_binary_classes_require_explicit_positive_semantics():
    pred, target = tables(
        [[0.8, 0.2], [0.1, 0.9]], ["yes", "no"], ["yes", "no"]
    )
    metrics = score(pred, target, "classification", positive_class="yes")
    assert metrics["accuracy"] == metrics["auroc"] == 1
    assert metrics["positive_class"] == "yes"
    unspecified = score(pred, target, "classification")
    assert unspecified["auroc"] is None
    assert (
        unspecified["auroc_undefined_reason"] == "positive_class_not_in_model"
    )


def test_unknown_target_class_is_rejected():
    pred, target = tables([[0.8, 0.2]], ["yes", "no"], ["unknown"])
    with pytest.raises(ValueError, match="classes missing from prediction"):
        score(pred, target, "classification")


@pytest.mark.parametrize(
    "values", [[[float("nan"), 0.2]], [[0.7, 0.2]], [[1.1, -0.1]]]
)
def test_malformed_probabilities_are_not_silently_renormalized(values):
    pred, target = tables(values, ["1", "0"], [1])
    with pytest.raises(ValueError, match="finite normalized"):
        score(pred, target, "classification")
