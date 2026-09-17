# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from textwrap import dedent

import pytest
import torch

import sdm.processing as sp
from sdm import EnsembleTable, Stype, TableTensor


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
    dispatch = sp.TaskDispatch(classification=sp.Softmax())
    output = _numerical_table(("a", "b"))
    description = dedent("""\
        TaskDispatch(
          classification=Softmax(),
        )""")
    assert repr(dispatch) == description

    dispatch._task = "regression"
    assert dispatch.transform(output).equal(output)
    assert repr(dispatch) == description

    restored = sp.TaskDispatch(classification=sp.Softmax())
    restored.load_state_dict(dispatch.state_dict())
    assert restored.transform(output).equal(output)

    dispatch._task = "classification"
    transformed = dispatch.transform(output)

    assert torch.allclose(
        transformed.numerical.sum(dim=-1),
        torch.ones(2),
    )
    assert repr(dispatch) == description

    restored = sp.TaskDispatch(classification=sp.Softmax())
    restored.load_state_dict(dispatch.state_dict())

    torch.testing.assert_close(
        restored.transform(output).numerical,
        transformed.numerical,
    )


def test_task_dispatch_rejects_invalid_routes_and_targets() -> None:
    assert len(sp.TaskDispatch().processors) == 0
    assert sp.TaskDispatch(regression=sp.Standardize()).requires_fit

    dispatch = sp.TaskDispatch(regression=sp.Identity())
    output = _numerical_table()

    with pytest.raises(RuntimeError, match="model execution"):
        dispatch.transform(output)
    dispatch._task = "classification"
    assert dispatch.transform(output).equal(output)


def test_task_dispatch_routes_ensemble_members() -> None:
    first = _numerical_table(("a", "b"))
    second = _numerical_table(("c", "d"))
    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )
    dispatch = sp.TaskDispatch(classification=sp.Softmax())
    dispatch._task = "classification"

    output = dispatch.transform_ensemble(ensemble_table)

    for member_id, source in enumerate((second, first, second)):
        result = output.table(member_id)
        assert result.columns == source.columns
        torch.testing.assert_close(
            result.numerical,
            source.numerical.softmax(dim=-1),
        )


def test_task_dispatch_forwards_stacked_outputs_to_reducers() -> None:
    dispatch = sp.TaskDispatch(
        classification=sp.ReduceEstimators(method="mean"),
        regression=[
            sp.ReduceQuantiles(),
            sp.ReduceEstimators(method="trimmed"),
        ],
    )
    stacked = TableTensor.from_tensor(torch.randn(8, 5, 4))

    dispatch._task = "regression"
    out = dispatch.transform(stacked)
    assert out.size() == (5, 1)
    assert out.columns[Stype.numerical] == ("mean",)

    dispatch._task = "classification"
    out = dispatch.transform(stacked)
    assert out.size() == (5, 4)
    torch.testing.assert_close(out.numerical, stacked.numerical.mean(dim=0))
