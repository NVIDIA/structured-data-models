# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import pytest
import torch

import sdm.processing as sp
from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import (
    InvertibleMixin,
    Processor,
)

ProcessorFactory = Callable[[], Processor]


@pytest.mark.parametrize(
    "processor_factory",
    [sp.ClipQuantiles, sp.Standardize],
)
def test_processor_requires_fit_for_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    inp = TableTensor.from_tensor(torch.ones(2, 2))

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform(inp)
    with pytest.raises(RuntimeError, match="not fitted"):
        processor(inp)


@pytest.mark.parametrize(
    "processor_factory",
    [sp.Standardize],
)
def test_invertible_processor_requires_fit_for_inverse_transform(
    processor_factory: ProcessorFactory,
) -> None:
    processor = processor_factory()
    inp = TableTensor.from_tensor(torch.ones(2, 2))

    assert isinstance(processor, InvertibleMixin)
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform(inp)


class StatelessProcessor(Processor):
    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + 1)


class StatefulProcessor(Processor):
    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        pass

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + 1)


def test_stateless_processor_runs_without_fit() -> None:
    processor = StatelessProcessor()
    inp = torch.ones(2, 2)
    table = TableTensor.from_tensor(inp)

    assert torch.equal(processor.transform(table).numerical, inp + 1)
    assert torch.equal(processor(table).numerical, inp + 1)
    assert torch.equal(processor.fit_transform(table).numerical, inp + 1)


def _mixed_table() -> TableTensor:
    return TableTensor(
        numerical=torch.tensor([[1.0], [2.0]]),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def _id_table() -> TableTensor:
    return TableTensor(id=ColumnarTensor((torch.arange(2),)))


def test_processor_preserves_unhandled_stypes() -> None:
    mixed = _mixed_table()

    transformed = sp.Standardize().fit_transform(mixed)
    processor = sp.Standardize().fit(mixed)

    for output in (transformed, processor.transform(mixed), processor(mixed)):
        assert output.columns == mixed.columns
        torch.testing.assert_close(
            output.numerical.mean(dim=0),
            torch.zeros(1),
            atol=1e-6,
            rtol=1e-6,
        )
        assert torch.equal(output.categorical.code, mixed.categorical.code)


def test_processor_noops_when_only_unhandled_stypes_are_active() -> None:
    table = _id_table()
    numerical = TableTensor.from_tensor(torch.ones(2, 1))

    processor = StatefulProcessor()

    assert processor.fit(table) is processor
    assert processor.fit_transform(table) is table
    assert processor.transform(table) is table
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform(numerical)


def test_processor_fit_transform_handles_empty_table() -> None:
    table = TableTensor.from_tensor(torch.empty(3, 0))
    output = sp.PCA(num_components=2).fit_transform(table)

    assert output.size() == table.size()
    assert output.schema == table.schema
