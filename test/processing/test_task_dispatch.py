from textwrap import dedent
from typing import cast

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import (
    CategoryShuffle,
    Identity,
    SoftmaxTemperature,
    StandardScale,
    TaskDispatch,
)


def _categorical_target() -> TableTensor:
    return TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def _numerical_table(
    columns: tuple[str, ...] = ("prediction",),
) -> TableTensor:
    return TableTensor.from_tensor(
        torch.arange(2 * len(columns), dtype=torch.float).reshape(
            2, len(columns)
        ),
        columns=columns,
    )


def test_task_dispatch_routes_output_and_has_stable_repr() -> None:
    dispatch = TaskDispatch(
        classification=SoftmaxTemperature(),
        regression=[Identity()],
    )
    output = _numerical_table(("a", "b"))
    description = dedent("""\
        TaskDispatch(
          classification: SoftmaxTemperature(),
          regression: Sequential(
              Identity(),
            ),
        )""")
    assert repr(dispatch) == description

    dispatch._resolve(_numerical_table())
    assert dispatch.transform(output) is output
    assert repr(dispatch) == description

    dispatch._resolve(_categorical_target())
    transformed = dispatch.transform(output)

    assert torch.allclose(
        transformed.numerical.sum(dim=-1),
        torch.ones(2),
    )
    assert repr(dispatch) == description

    restored = TaskDispatch(
        classification=SoftmaxTemperature(),
        regression=[Identity()],
    )
    restored.load_state_dict(dispatch.state_dict())

    torch.testing.assert_close(
        restored.transform(output).numerical,
        transformed.numerical,
    )


def test_task_dispatch_rejects_invalid_routes_and_targets() -> None:
    with pytest.raises(ValueError, match="at least one route"):
        TaskDispatch()

    dispatch = TaskDispatch(regression=Identity())
    output = _numerical_table()

    with pytest.raises(RuntimeError, match=r"recipe\.target\.fit"):
        dispatch.transform(output)
    with pytest.raises(RuntimeError, match=r"recipe\.target\.fit"):
        dispatch.inverse_transform(output)
    with pytest.raises(ValueError, match="no 'classification' route"):
        dispatch._resolve(_categorical_target())
    with pytest.raises(ValueError, match=r"exactly one.*got 2"):
        dispatch._resolve(_numerical_table(("y0", "y1")))


def test_task_dispatch_fits_target_route() -> None:
    torch.manual_seed(0)
    dispatch = TaskDispatch(
        classification=CategoryShuffle(method="shift"),
        regression=StandardScale(),
    )
    assert dispatch.requires_fit

    target = _categorical_target()
    transformed = dispatch.fit_transform(target)
    assert transformed.columns == target.columns

    # The inverse receives the complete numerical model-output head and
    # restores the original class-score order:
    head = torch.randn(2, 10)
    restored = dispatch.inverse_transform(TableTensor.from_tensor(head))
    shuffle = cast(CategoryShuffle, dispatch.processors["classification"])
    torch.testing.assert_close(
        restored.numerical,
        head[:, shuffle.permutations],
    )

    numerical_target = TableTensor.from_tensor(
        torch.randn(8, 1),
        columns=("target",),
    )
    transformed = dispatch.fit_transform(numerical_target)
    restored = dispatch.inverse_transform(transformed)
    torch.testing.assert_close(
        restored.numerical,
        numerical_target.numerical,
    )

    stateless = TaskDispatch(regression=SoftmaxTemperature())
    stateless.fit(numerical_target)
    with pytest.raises(TypeError, match=r"regression.*SoftmaxTemperature"):
        stateless.inverse_transform(transformed)
