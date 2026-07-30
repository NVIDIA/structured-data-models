from __future__ import annotations

import pytest
import torch

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.models.tabiclv2.recipe import default_recipe


def _features() -> TableTensor:
    values = torch.tensor(
        [
            [-8.0, 0.0, 1.0, 2.0],
            [-3.0, 1.0, 1.5, 4.0],
            [-1.0, 2.0, 2.0, 8.0],
            [0.0, 4.0, 3.0, 16.0],
            [1.0, 8.0, 5.0, 32.0],
            [3.0, 16.0, 8.0, 64.0],
            [8.0, 32.0, 13.0, 128.0],
            [21.0, 64.0, 21.0, 256.0],
        ]
    )
    return TableTensor.from_tensor(
        values,
        columns=("x0", "x1", "x2", "x3"),
    )


def _target(task: str) -> TableTensor:
    if task == "regression":
        return TableTensor.from_tensor(
            torch.tensor(
                [[-4.0], [-1.0], [0.0], [1.0], [4.0], [9.0], [16.0], [25.0]]
            )
        )
    return TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [1], [2], [0], [1], [2], [0], [1]],
                dtype=torch.int64,
            ),
            categories=(torch.tensor([10, 20, 30]),),
        ),
    )


def _unsorted_target_with_unused_class() -> TableTensor:
    return TableTensor(
        columns={Stype.categorical: ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[1], [2], [0], [1], [2], [0], [1], [2]],
                dtype=torch.int64,
            ),
            categories=(torch.tensor([30, 10, 20, 40]),),
        ),
    )


@pytest.mark.parametrize(
    ("task", "feature_permutations"),
    [
        (
            "classification",
            (
                (2, 0, 3, 1),
                (0, 1, 2, 3),
                (3, 2, 1, 0),
                (2, 0, 3, 1),
            ),
        ),
        (
            "regression",
            (
                (2, 0, 3, 1),
                (0, 1, 2, 3),
                (1, 3, 0, 2),
                (3, 2, 1, 0),
            ),
        ),
    ],
)
def test_eight_member_recipe_matches_tabicl_reference_plan(
    task: str,
    feature_permutations: tuple[tuple[int, ...], ...],
) -> None:
    recipe = default_recipe()
    features = _features()
    target = _target(task)

    transformed_features, transformed_target, _ = recipe.fit_transform(
        features,
        target,
        num_members=8,
        generator=torch.Generator().manual_seed(42),
    )

    expected_columns = tuple(
        tuple(f"x{index}" for index in permutation)
        for permutation in feature_permutations
        for _ in range(2)
    )
    assert (
        tuple(
            transformed_features[member].columns[Stype.numerical]
            for member in range(8)
        )
        == expected_columns
    )

    canonicalized = tuple(
        transformed_features[member].numerical.index_select(
            -1,
            torch.tensor(feature_permutations[member // 2]).argsort(),
        )
        for member in range(8)
    )
    for member in range(2, 8, 2):
        torch.testing.assert_close(canonicalized[member], canonicalized[0])
        torch.testing.assert_close(
            canonicalized[member + 1],
            canonicalized[1],
        )
    assert not torch.allclose(canonicalized[0], canonicalized[1])

    if task == "classification":
        class_permutations = (
            (2, 0, 1),
            (1, 2, 0),
            (1, 2, 0),
            (1, 2, 0),
        )
        original_codes = target.categorical.code.squeeze(-1)
        for member in range(8):
            permutation = torch.tensor(class_permutations[member // 2])
            torch.testing.assert_close(
                transformed_target[member].categorical.code.squeeze(-1),
                permutation[original_codes],
            )


def test_reference_recipe_uses_sorted_observed_classes() -> None:
    recipe = default_recipe()
    _, transformed_target, _ = recipe.fit_transform(
        _features(),
        _unsorted_target_with_unused_class(),
        num_members=8,
        generator=torch.Generator().manual_seed(42),
    )
    outputs = tuple(
        TableTensor.from_tensor(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        for _ in range(8)
    )

    decoded = recipe.transform_output(outputs)

    assert transformed_target[0].categorical.categories[0].numel() == 3
    assert decoded.columns[Stype.numerical] == ("10", "20", "30")
