from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("tabicl")

from sdm import TableTensor
from sdm.models.tabiclv2.recipe import default_recipe
from sdm.processing import Choice, Power

from .datasets import classification_cases, regression_cases
from .harness import (
    AGGREGATED_OUTPUT,
    ENCODED_NUMERICAL_FEATURES,
    FEATURE_PERMUTATION,
    FEATURE_PIPELINE_OUTPUT,
    FINAL_MODEL_INPUT,
    FINAL_OUTPUT,
    RAW_MODEL_OUTPUT,
    TARGET_ENCODING,
    TARGET_INVERSE_OUTPUT,
    TARGET_TRANSFORMATION,
    USER_PREDICTION,
    EnsembleMemberPlan,
    StageSnapshot,
    StageTrace,
    Task,
    first_divergence,
    record_output_stages,
    reference_ensemble_plan,
    sdm_ensemble_plan,
    trace_reference_member,
    trace_sdm_member,
)


def _features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stypes: dict[str, str] | None = None,
) -> TableTensor:
    frame = pd.concat([train, test], ignore_index=True)
    if stypes is None:
        stypes = dict.fromkeys(frame.columns, "numerical")
    return TableTensor.from_pandas(frame, stypes)


def _target(values: np.ndarray, task: str) -> TableTensor:
    return TableTensor.from_pandas(
        pd.DataFrame({"target": values}),
        {"target": "categorical" if task == "classification" else "numerical"},
    )


def _member(
    *,
    task: Task,
    feature_permutation: tuple[int, ...],
    class_permutation: tuple[int, ...] | None = None,
) -> EnsembleMemberPlan:
    return EnsembleMemberPlan(
        member_index=0,
        normalization="none",
        feature_permutation=feature_permutation,
        class_permutation=class_permutation,
        target_transformation=(
            "label_encode+class_permute"
            if task == "classification"
            else "standard_scale"
        ),
        processor_seeds=(("random_state", 42),),
        model_output_type=(
            "logits" if task == "classification" else "raw_quantiles"
        ),
        aggregation_space=(
            "canonical_logits"
            if task == "classification"
            else "original_target"
        ),
    )


@pytest.mark.parametrize(
    ("task", "n_features", "n_classes", "expected_members"),
    [
        ("classification", 1, 2, 4),
        ("classification", 1, 3, 6),
        ("regression", 1, None, 2),
        ("classification", 4, 3, 8),
        ("regression", 4, None, 8),
    ],
)
def test_reference_plan_cardinality_is_configuration_limited(
    task: Task,
    n_features: int,
    n_classes: int | None,
    expected_members: int,
) -> None:
    plan = reference_ensemble_plan(
        n_features=n_features,
        n_classes=n_classes,
        n_estimators=8,
        task=task,
        seed=42,
    )

    assert len(plan.members) == expected_members
    assert all(
        member.aggregation_space
        == (
            "canonical_logits"
            if task == "classification"
            else "original_target"
        )
        for member in plan.members
    )


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_reference_single_estimator_is_identity(task: Task) -> None:
    plan = reference_ensemble_plan(
        n_features=4,
        n_classes=3 if task == "classification" else None,
        n_estimators=1,
        task=task,
        seed=42,
    )

    assert len(plan.members) == 1
    member = plan.members[0]
    assert member.normalization == "none"
    assert member.feature_permutation == (0, 1, 2, 3)
    if task == "classification":
        assert member.class_permutation == (0, 1, 2)


def test_reference_plan_is_seed_deterministic() -> None:
    first = reference_ensemble_plan(
        n_features=7,
        n_classes=4,
        n_estimators=8,
        task="classification",
        seed=42,
    )
    repeated = reference_ensemble_plan(
        n_features=7,
        n_classes=4,
        n_estimators=8,
        task="classification",
        seed=42,
    )
    changed = reference_ensemble_plan(
        n_features=7,
        n_classes=4,
        n_estimators=8,
        task="classification",
        seed=7,
    )

    assert first == repeated
    assert first.canonical_members() != changed.canonical_members()


def test_sdm_plan_seed_and_intentional_difference() -> None:
    case = classification_cases()[1]
    features = _features(
        case.train_features,
        case.test_features,
        {"number": "numerical", "category": "categorical"},
    )
    target = _target(case.target, case.task)
    kwargs = {
        "recipe": default_recipe(),
        "features": features,
        "target": target,
        "n_estimators": 8,
    }

    first = sdm_ensemble_plan(**kwargs, seed=42)
    repeated = sdm_ensemble_plan(**kwargs, seed=42)
    changed = sdm_ensemble_plan(**kwargs, seed=7)

    assert first == repeated
    assert first.canonical_members() != changed.canonical_members()
    assert len(first.members) == 8
    assert {member.normalization for member in first.members} <= {
        "none",
        "power",
    }
    reference = reference_ensemble_plan(
        n_features=2,
        n_classes=3,
        n_estimators=8,
        task="classification",
        seed=42,
    )
    assert {member.normalization for member in reference.members} == {
        "none",
        "power",
    }


def test_injected_numeric_member_reaches_equivalent_model_input() -> None:
    train = pd.DataFrame(
        {"a": [1.0, 2.0, 4.0, 8.0], "b": [8.0, 4.0, 2.0, 1.0]}
    )
    test = pd.DataFrame({"a": [3.0, 10.0], "b": [3.0, 0.0]})
    target_values = np.array([1.0, 2.0, 4.0, 8.0])
    member = _member(task="regression", feature_permutation=(1, 0))

    reference = trace_reference_member(
        train_features=train,
        test_features=test,
        target=target_values,
        member=member,
        task="regression",
        seed=42,
    )
    candidate, _ = trace_sdm_member(
        recipe=default_recipe(),
        features=_features(train, test),
        target=_target(target_values, "regression"),
        member=member,
    )

    difference = first_divergence(
        reference,
        candidate,
        stages=[
            ENCODED_NUMERICAL_FEATURES,
            FEATURE_PIPELINE_OUTPUT,
            FEATURE_PERMUTATION,
            TARGET_ENCODING,
            TARGET_TRANSFORMATION,
            FINAL_MODEL_INPUT,
        ],
    )
    assert difference is not None
    assert difference.stage == ENCODED_NUMERICAL_FEATURES
    assert difference.reason == "dtype float64 != torch.float32"

    expected = reference.snapshots[FINAL_MODEL_INPUT]
    actual = candidate.snapshots[FINAL_MODEL_INPUT]
    assert expected.dtype == actual.dtype == "torch.float32"
    assert expected.columns == actual.columns == ("b", "a")
    torch.testing.assert_close(
        expected.values,
        actual.values,
        atol=2e-7,
        rtol=1e-7,
    )


def test_explicit_power_member_is_structurally_equivalent() -> None:
    train = pd.DataFrame(
        {
            "a": [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0],
            "b": [8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 1.0],
        }
    )
    test = pd.DataFrame({"a": [2.5, 9.0], "b": [2.5, 9.0]})
    target_values = np.arange(8, dtype=np.float64)
    recipe = default_recipe()
    choice = next(
        module
        for module in recipe.features.modules()
        if isinstance(module, Choice)
    )
    choice.options[1] = Power()
    member = _member(task="regression", feature_permutation=(0, 1))
    member = replace(member, normalization="power")

    reference = trace_reference_member(
        train_features=train,
        test_features=test,
        target=target_values,
        member=member,
        task="regression",
        seed=42,
    )
    candidate, _ = trace_sdm_member(
        recipe=recipe,
        features=_features(train, test),
        target=_target(target_values, "regression"),
        member=member,
    )
    expected = reference.snapshots[FINAL_MODEL_INPUT].values
    actual = candidate.snapshots[FINAL_MODEL_INPUT].values
    assert expected is not None
    assert actual is not None

    absolute = (expected - actual).abs()
    relative = absolute / expected.abs().clamp_min(
        torch.finfo(expected.dtype).tiny
    )
    assert absolute.max().item() <= 2e-4
    assert relative.max().item() <= 4e-4


def test_mixed_categories_preserve_values_across_route_order() -> None:
    case = classification_cases()[1]
    member = _member(
        task="classification",
        feature_permutation=(0, 1),
        class_permutation=(1, 2, 0),
    )
    reference = trace_reference_member(
        train_features=case.train_features,
        test_features=case.test_features,
        target=case.target,
        member=member,
        task="classification",
        seed=42,
    )
    candidate, _ = trace_sdm_member(
        recipe=default_recipe(),
        features=_features(
            case.train_features,
            case.test_features,
            {"number": "numerical", "category": "categorical"},
        ),
        target=_target(case.target, case.task),
        member=member,
    )

    reference_encoded = reference.snapshots[ENCODED_NUMERICAL_FEATURES]
    candidate_encoded = candidate.snapshots[ENCODED_NUMERICAL_FEATURES]
    assert reference_encoded.columns == ("category", "number")
    assert candidate_encoded.columns == ("number", "category")
    assert reference_encoded.values is not None
    assert candidate_encoded.values is not None
    reference_category = reference_encoded.values[
        :, reference_encoded.columns.index("category")
    ]
    candidate_category = candidate_encoded.values[
        :, candidate_encoded.columns.index("category")
    ]
    torch.testing.assert_close(
        candidate_category,
        reference_category.to(candidate_category.dtype),
        atol=0,
        rtol=0,
    )
    reference_number = reference_encoded.values[
        :, reference_encoded.columns.index("number")
    ]
    candidate_number = candidate_encoded.values[
        :, candidate_encoded.columns.index("number")
    ]
    assert torch.isnan(candidate_number[1])
    assert not torch.isnan(reference_number[1])
    assert candidate_category[-2:].tolist() == [-1.0, -1.0]

    expected_target = reference.snapshots[TARGET_TRANSFORMATION]
    actual_target = candidate.snapshots[TARGET_TRANSFORMATION]
    assert expected_target.values is not None
    assert actual_target.values is not None
    assert expected_target.values.tolist() == [0, 1, 2, 0, 1, 2]
    assert actual_target.values.tolist() == expected_target.values.tolist()

    expected_snapshot = reference.snapshots[FINAL_MODEL_INPUT]
    actual_snapshot = candidate.snapshots[FINAL_MODEL_INPUT]
    expected_model = expected_snapshot.values
    actual_model = actual_snapshot.values
    assert expected_model is not None
    assert actual_model is not None
    assert set(actual_snapshot.columns) == set(expected_snapshot.columns)
    # SDM intentionally retains StypeDispatch's default route order. Route
    # order changes positions, not the transformed value for each feature.
    actual_model = actual_model[
        :,
        [
            actual_snapshot.columns.index(column)
            for column in expected_snapshot.columns
        ],
    ]
    torch.testing.assert_close(
        actual_model,
        expected_model,
        atol=2e-7,
        rtol=1e-7,
    )


def test_output_stage_recorder_preserves_contract_metadata() -> None:
    member = _member(
        task="classification",
        feature_permutation=(0,),
        class_permutation=(1, 0),
    )
    trace = StageTrace("analytic", member)
    raw = torch.tensor([[2.0, -1.0]])
    canonical = torch.tensor([[-1.0, 2.0]])
    probabilities = canonical.softmax(dim=-1)
    record_output_stages(
        trace,
        raw_model_output=raw,
        target_inverse_output=canonical,
        aggregated_output=canonical,
        final_output=probabilities,
        user_prediction=torch.tensor([1]),
    )

    assert tuple(trace.snapshots) == (
        RAW_MODEL_OUTPUT,
        TARGET_INVERSE_OUTPUT,
        AGGREGATED_OUTPUT,
        FINAL_OUTPUT,
        USER_PREDICTION,
    )
    assert trace.snapshots[AGGREGATED_OUTPUT].metadata == (
        ("space", "canonical_logits"),
    )


def test_dataset_catalog_covers_required_shapes_and_targets() -> None:
    classification = classification_cases()
    regression = regression_cases()

    assert {case.name for case in classification} == {
        "binary_one_feature",
        "three_string_classes_mixed_unknown_missing",
        "constant_outlier_more_features_than_members",
    }
    assert {case.name for case in regression} == {
        "negative_and_positive",
        "constant_target",
        "near_constant_target",
        "strongly_skewed_target",
    }
    assert any(case.train_features.shape[1] == 1 for case in classification)
    assert any(case.train_features.shape[1] > 2 for case in classification)
    assert np.ptp(regression[0].target) > 0
    assert np.ptp(regression[1].target) == 0
    assert regression[-1].target[-1] / regression[-1].target[-2] == 16


def test_stage_snapshot_compares_metadata_as_discrete_state() -> None:
    member = _member(task="regression", feature_permutation=(0,))
    reference = StageTrace("reference", member)
    candidate = StageTrace("candidate", member)
    reference.record(
        StageSnapshot.from_array(
            AGGREGATED_OUTPUT,
            np.array([1.0]),
            metadata={"space": "original_target"},
        )
    )
    candidate.record(
        StageSnapshot.from_array(
            AGGREGATED_OUTPUT,
            np.array([1.0]),
            metadata={"space": "transformed_target"},
        )
    )

    difference = first_divergence(
        reference,
        candidate,
        stages=[AGGREGATED_OUTPUT],
    )
    assert difference is not None
    assert "metadata" in difference.reason
