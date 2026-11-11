# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import cast

import pytest
import torch

from sdm import EnsembleTable, EnsembleTensor, StringTensor, Stype, TableTensor
from sdm.processing import (
    InvertibleMixin,
    Processor,
    Standardize,
)
from sdm.testing import withCUDA


class IdentityProcessor(Processor):
    handles_stypes = (Stype.numerical,)
    requires_fit = True

    def _fit(
        self,
        table: TableTensor | EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        pass

    def _transform(
        self,
        table: TableTensor | EnsembleTable,
    ) -> TableTensor | EnsembleTable:
        return table


class FusedProcessor(IdentityProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.used_fused_transform = False

    def _fit_transform(
        self,
        table: TableTensor | EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor | EnsembleTable:
        self.used_fused_transform = True
        return table


class InvertibleIdentityProcessor(
    IdentityProcessor,
    InvertibleMixin,
):
    def _inverse_transform(
        self,
        table: TableTensor | EnsembleTable,
    ) -> TableTensor | EnsembleTable:
        if isinstance(table, TableTensor):
            return table
        return EnsembleTable.from_table(table[0], num_members=1)


def test_ensemble_buffer_replacement_and_loading() -> None:
    processor = IdentityProcessor()
    processor.register_buffer("values", None)
    processor.values = torch.zeros(1)
    restored = IdentityProcessor()
    restored.register_buffer("values", None)
    restored.load_state_dict(processor.state_dict())
    torch.testing.assert_close(restored.values, torch.zeros(1))
    processor.register_buffer(
        "values",
        EnsembleTensor(
            groups=(torch.arange(6.0).reshape(2, 3),),
            locations=((0, 0), (0, 1)),
        ),
    )
    saved = copy.deepcopy(processor.state_dict())
    processor.values = EnsembleTensor(
        groups=(torch.ones(1, 1),), locations=((0, 0),)
    )
    processor.load_state_dict(saved)
    actual = cast(EnsembleTensor[torch.Tensor], processor.values)
    assert isinstance(actual, EnsembleTensor)
    torch.testing.assert_close(actual[1], torch.tensor([3.0, 4.0, 5.0]))
    cloned = copy.deepcopy(processor)
    cloned_values = cast(EnsembleTensor[torch.Tensor], cloned.values)
    assert isinstance(cloned_values, EnsembleTensor)
    cloned_values[0].zero_()
    torch.testing.assert_close(actual[0], torch.tensor([0.0, 1.0, 2.0]))
    processor.register_buffer("values", torch.ones(1))
    assert isinstance(processor.values, torch.Tensor)


@withCUDA
def test_nonpersistent_ensemble_buffer(device: torch.device) -> None:
    processor = IdentityProcessor()
    processor.register_buffer(
        "temporary",
        EnsembleTensor(groups=(torch.ones(1, 2),), locations=((0, 0),)),
        persistent=False,
    )
    processor.register_buffer(
        "temporary_strings",
        StringTensor.from_list(["cached"]),
        persistent=False,
    )
    processor.to(device)
    temporary = cast(EnsembleTensor[torch.Tensor], processor.temporary)
    assert isinstance(temporary, EnsembleTensor)
    assert temporary[0].device == device
    assert not any(
        key.startswith("temporary") for key in processor.state_dict()
    )
    processor.load_state_dict(processor.state_dict())


def test_ensemble_processor_preserves_member_order_and_metadata() -> None:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [2.0]]),
        columns=("first",),
    )
    second = TableTensor.from_tensor(
        torch.tensor([[3.0], [4.0]]),
        columns=("second",),
    )
    ensemble_table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )
    processor = IdentityProcessor()

    output = processor.fit_transform(ensemble_table)

    assert output is ensemble_table
    assert len(output) == 3
    assert output[0].columns == second.columns
    assert output[1].columns == first.columns
    assert output[2].columns == second.columns
    assert processor.transform(ensemble_table) is ensemble_table


def test_ensemble_invertible_mixin_requires_fit_and_delegates() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable.from_table(table, num_members=2)
    processor = InvertibleIdentityProcessor()

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform(ensemble_table)

    processor.fit_transform(ensemble_table)
    output = processor.inverse_transform(ensemble_table)

    assert len(output) == 1
    assert output[0].equal(table)
    assert processor.inverse_transform(table).equal(table)


def test_ensemble_processor_uses_fused_table_tensor_lifecycle() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    processor = FusedProcessor()

    assert processor.fit_transform(table).equal(table)
    assert processor.used_fused_transform


def test_ensemble_processor_passthrough_for_empty_supported_blocks() -> None:
    empty_ensemble_table = EnsembleTable.from_table(
        TableTensor.from_tensor(torch.empty(2, 0)),
        num_members=2,
    )
    ensemble_table = EnsembleTable.from_table(
        TableTensor.from_tensor(torch.ones(2, 1)),
        num_members=2,
    )
    processor = IdentityProcessor()

    assert processor.fit(empty_ensemble_table) is processor
    assert (
        processor.fit_transform(empty_ensemble_table) is empty_ensemble_table
    )
    assert processor.transform(empty_ensemble_table) is empty_ensemble_table
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform(ensemble_table)


def _two_group_ensemble_table() -> EnsembleTable:
    first = TableTensor.from_tensor(
        torch.tensor([[1.0], [3.0]]), columns=("first",)
    )
    second = TableTensor.from_tensor(
        torch.tensor([[2.0], [6.0]]), columns=("second",)
    )
    return EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )


def test_processor_learns_member_statistics() -> None:
    ensemble_table = _two_group_ensemble_table()
    processor = Standardize()

    processor.fit(ensemble_table)
    output = processor.transform(ensemble_table)

    assert len(output) == 3
    assert output[0].numerical.tolist() == [[-1.0], [1.0]]
    assert output[1].numerical.tolist() == [[-1.0], [1.0]]
    assert output[2].equal(output[0])
