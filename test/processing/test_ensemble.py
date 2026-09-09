# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    EnsembleProcessorAdapter,
    InvertibleMixin,
    Processor,
    Standardize,
)
from sdm.tensor import EnsembleTable


# TODO: Replace these stubs with real EnsembleProcessor subclasses once they
# land, and exercise the EnsembleProcessor contract through those instead.
# Generator forwarding is only observable through a stochastic transformation
# and is therefore left to those processors as well.
class IdentityEnsembleProcessor(EnsembleProcessor):
    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        pass

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table


class FusedEnsembleProcessor(IdentityEnsembleProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.used_fused_transform = False

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self.used_fused_transform = True
        return ensemble_table


class InvertibleIdentityEnsembleProcessor(
    IdentityEnsembleProcessor,
    EnsembleInvertibleMixin,
):
    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return EnsembleTable.from_table(ensemble_table.table(0), num_members=1)


class _StatelessProcessor(Processor, InvertibleMixin):
    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=-table.numerical)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return self._transform(table)


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
    processor = IdentityEnsembleProcessor()

    output = processor.fit_transform_ensemble(ensemble_table)

    assert output is ensemble_table
    assert output.num_members == 3
    assert output.table(0).columns == second.columns
    assert output.table(1).columns == first.columns
    assert output.table(2).columns == second.columns
    assert processor.transform_ensemble(ensemble_table) is ensemble_table


def test_ensemble_processor_requires_fit_before_transform() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable.from_table(table, num_members=2)
    processor = IdentityEnsembleProcessor()

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform_ensemble(ensemble_table)

    assert processor.fit_ensemble(ensemble_table) is processor
    assert processor.transform_ensemble(ensemble_table) is ensemble_table


def test_ensemble_invertible_mixin_requires_fit_and_delegates() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    ensemble_table = EnsembleTable.from_table(table, num_members=2)
    processor = InvertibleIdentityEnsembleProcessor()

    with pytest.raises(RuntimeError, match="not fitted"):
        processor.inverse_transform_ensemble(ensemble_table)

    processor.fit_transform_ensemble(ensemble_table)
    output = processor.inverse_transform_ensemble(ensemble_table)

    assert output.num_members == 1
    assert output.table(0).equal(table)
    assert processor.inverse_transform(table).equal(table)


def test_ensemble_processor_noops() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1, dtype=torch.int64))
    ensemble_table = EnsembleTable.from_table(table, num_members=2)
    processor = IdentityEnsembleProcessor()

    assert processor.fit_ensemble(ensemble_table) is processor
    assert processor.fit_transform_ensemble(ensemble_table) is ensemble_table
    assert processor.transform_ensemble(ensemble_table) is ensemble_table


def test_ensemble_processor_supports_table_tensor_lifecycle() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    fitted = IdentityEnsembleProcessor()
    processor = IdentityEnsembleProcessor()

    assert fitted.fit(table) is fitted
    assert fitted.transform(table).equal(table)
    assert processor.fit_transform(table).equal(table)
    assert processor.transform(table).equal(table)
    assert processor(table).equal(table)


def test_ensemble_processor_uses_fused_table_tensor_lifecycle() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    processor = FusedEnsembleProcessor()

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
    processor = IdentityEnsembleProcessor()

    assert processor.fit_ensemble(empty_ensemble_table) is processor
    assert (
        processor.fit_transform_ensemble(empty_ensemble_table)
        is empty_ensemble_table
    )
    assert (
        processor.transform_ensemble(empty_ensemble_table)
        is empty_ensemble_table
    )
    with pytest.raises(RuntimeError, match="not fitted"):
        processor.transform_ensemble(ensemble_table)


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


def test_adapter_fits_each_group_separately() -> None:
    ensemble_table = _two_group_ensemble_table()
    processor = EnsembleProcessorAdapter(Standardize(with_std=False))

    processor.fit_ensemble(ensemble_table)
    output = processor.transform_ensemble(ensemble_table)

    assert output.num_members == 3
    assert output.table(0).numerical.tolist() == [[-2.0], [2.0]]
    assert output.table(1).numerical.tolist() == [[-1.0], [1.0]]
    assert output.table(2).equal(output.table(0))


def test_adapter_inverse_restores_input() -> None:
    ensemble_table = _two_group_ensemble_table()
    processor = EnsembleProcessorAdapter(Standardize(with_std=False))

    output = processor.fit_transform_ensemble(ensemble_table)
    restored = processor.inverse_transform_ensemble(output)

    for member_id in range(ensemble_table.num_members):
        assert restored.table(member_id).equal(ensemble_table.table(member_id))


def test_stateless_adapter_supports_transform_and_inverse() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0]]))
    processor = EnsembleProcessorAdapter(_StatelessProcessor())

    transformed = processor.inverse_transform(table)

    assert transformed.numerical.tolist() == [[-1.0], [-2.0]]
    assert processor.transform(transformed).equal(table)
    assert processor.inverse_transform(transformed).equal(table)


def test_adapter_rejects_inverse_for_non_invertible_processor() -> None:
    table = TableTensor.from_tensor(torch.ones(2, 1))
    processor = EnsembleProcessorAdapter(
        Processor.as_processor(lambda value: value)
    )

    with pytest.raises(AttributeError, match="inverse_transform"):
        processor.inverse_transform(table)
