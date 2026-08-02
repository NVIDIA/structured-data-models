from textwrap import dedent

import pytest
import torch

from sdm import CategoricalTensor, EnsembleTable, StringTensor, TableTensor
from sdm.processing import (
    EnsembleProcessor,
    Identity,
    Softmax,
    Standardize,
    TaskDispatch,
)


def _categorical_target() -> TableTensor:
    return TableTensor(
        columns={"categorical": ("target",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32),
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
        classification=Softmax(),
        regression=[Identity()],
    )
    output = _numerical_table(("a", "b"))
    description = dedent("""\
        TaskDispatch(
          classification: Softmax(),
          regression: Sequential(
            Identity(),
          ),
        )""")
    assert repr(dispatch) == description

    dispatch._resolve(_numerical_table())
    assert dispatch.transform(output).equal(output)
    assert repr(dispatch) == description

    dispatch._resolve(_categorical_target())
    transformed = dispatch.transform(output)

    assert torch.allclose(
        transformed.numerical.sum(dim=-1),
        torch.ones(2),
    )
    assert repr(dispatch) == description

    restored = TaskDispatch(
        classification=Softmax(),
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

    with pytest.raises(ValueError, match=r"regression.*requires fit"):
        TaskDispatch(regression=Standardize())

    dispatch = TaskDispatch(regression=Identity())
    output = _numerical_table()

    with pytest.raises(RuntimeError, match=r"recipe\.target\.fit"):
        dispatch.transform(output)
    with pytest.raises(ValueError, match="no 'classification' route"):
        dispatch._resolve(_categorical_target())
    with pytest.raises(ValueError, match=r"exactly one.*got 2"):
        dispatch._resolve(_numerical_table(("y0", "y1")))


def test_task_dispatch_is_an_ensemble_processor() -> None:
    assert issubclass(TaskDispatch, EnsembleProcessor)


def test_task_dispatch_routes_packed_ensemble_output() -> None:
    first = _numerical_table(("a", "b"))
    second = first.replace_blocks(numerical=first.numerical.flip(-1))
    table = EnsembleTable.from_representations(
        (first, second),
        member_representation_ids=(1, 0, 1),
    )
    dispatch = TaskDispatch(classification=Softmax())
    dispatch._resolve(_categorical_target())

    output = dispatch.transform_ensemble(table)

    for member_id, source in enumerate((second, first, second)):
        torch.testing.assert_close(
            output.representation(member_id).numerical,
            source.numerical.softmax(dim=-1),
        )
