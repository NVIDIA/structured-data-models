import inspect
import io
from copy import deepcopy
from typing import cast

import pytest
import torch
from test.processing._instance_generator import (
    PROCESSOR_CASES,
    ProcessorCase,
    make_mixed_inputs,
)

import sdm.processing as sp
from sdm import Stype, TableTensor
from sdm.tensor import EnsembleTable
from sdm.testing import onlyCUDA


def test_public_processors_are_registered_or_scoped() -> None:
    public = {
        value
        for name in sp.__all__
        if isinstance((value := getattr(sp, name)), type)
        and issubclass(value, sp.Processor)
        and not inspect.isabstract(value)
    }
    registered = {type(case.processor) for case in PROCESSOR_CASES}
    containers = {sp.TaskDispatch, sp.TableDispatch}
    processor_specific = {sp.SentenceTransformer}
    assert public == registered | containers | processor_specific


def _assert_table_close(
    actual: TableTensor,
    expected: TableTensor,
    *,
    atol: float = 1e-5,
    check_dtype: bool = True,
) -> None:
    if actual.device != expected.device:
        actual = cast(TableTensor, actual.to(expected.device))
    assert actual.columns == expected.columns
    assert actual.size() == expected.size()
    torch.testing.assert_close(
        actual.numerical,
        expected.numerical,
        atol=atol,
        rtol=1e-5,
        equal_nan=True,
        check_dtype=check_dtype,
    )
    assert actual.categorical.equal(expected.categorical)
    assert torch.equal(actual.datetime, expected.datetime)
    assert actual.text.equal(expected.text)
    assert actual.id.equal(expected.id)


def _assert_data_close(
    actual: EnsembleTable,
    expected: EnsembleTable,
    *,
    atol: float = 1e-5,
    check_dtype: bool = True,
) -> None:
    assert actual.num_members == expected.num_members
    for member_id in range(actual.num_members):
        _assert_table_close(
            actual.table(member_id),
            expected.table(member_id),
            atol=atol,
            check_dtype=check_dtype,
        )


def _check_state_dict_restoration(case: ProcessorCase) -> None:
    context, query = case.make_inputs()
    fitted = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    restored = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    data = EnsembleTable.from_tables(
        tables=(context, query), member_table_ids=(1, 0, 1, 0)
    )
    fitted.fit_ensemble(
        data,
        generator=torch.Generator().manual_seed(0),
    )
    expected = fitted.transform_ensemble(data)
    stream = io.BytesIO()
    torch.save(fitted.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=False))
    _assert_data_close(restored.transform_ensemble(data), expected)


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_fit_transform_matches_fit_then_transform(
    case: ProcessorCase,
) -> None:
    context, query = case.make_inputs()
    split = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    fused = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    data = EnsembleTable.from_tables(
        tables=(context, query), member_table_ids=(1, 0, 1, 0)
    )
    split.fit_ensemble(
        data,
        generator=torch.Generator().manual_seed(0),
    )
    actual = fused.fit_transform_ensemble(
        data,
        generator=torch.Generator().manual_seed(0),
    )
    _assert_data_close(actual, split.transform_ensemble(data))


FITTED_CASES = tuple(
    case for case in PROCESSOR_CASES if case.processor.requires_fit
)


@pytest.mark.parametrize(
    "case",
    FITTED_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_state_dict_restores_fitted_processor(
    case: ProcessorCase,
) -> None:
    _check_state_dict_restoration(case)


class _BrokenFittedProcessor(sp.Processor):
    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.mean = torch.empty(0)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.mean = table.numerical.mean(dim=-2, keepdim=True)

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical - self.mean)


def test_state_dict_contract_rejects_unregistered_fitted_state() -> None:
    broken = ProcessorCase(_BrokenFittedProcessor(), make_mixed_inputs)
    with pytest.raises(RuntimeError):
        _check_state_dict_restoration(broken)


INVERTIBLE_CASES = tuple(
    case
    for case in PROCESSOR_CASES
    if isinstance(case.processor, sp.InvertibleMixin)
)


@pytest.mark.parametrize(
    "case",
    INVERTIBLE_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_inverse_transform_round_trip(case: ProcessorCase) -> None:
    context, _ = case.make_inputs()
    processor = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    data = EnsembleTable(context, num_members=2)
    transformed = processor.fit_transform_ensemble(
        data,
        generator=torch.Generator().manual_seed(0),
    )
    restored = cast(
        sp.EnsembleInvertibleMixin, processor
    ).inverse_transform_ensemble(transformed)
    _assert_data_close(restored, data, atol=1e-3)


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_preserves_rows_and_unhandled_stypes(
    case: ProcessorCase,
) -> None:
    context, query = case.make_inputs()
    processor = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    data = EnsembleTable.from_tables(
        tables=(context, query), member_table_ids=(1, 0, 1, 0)
    )
    output = processor.fit_transform_ensemble(
        data,
        generator=torch.Generator().manual_seed(0),
    )
    for member_id in range(output.num_members):
        before = data.table(member_id)
        after = output.table(member_id)
        assert after.size()[:-1] == before.size()[:-1]
        for stype in before.active_stypes - processor.handles_stypes:
            columns = before.columns[stype]
            assert all(column in after.column_names for column in columns)
            assert all(after.stype(column) == stype for column in columns)
            _assert_table_close(
                after.select_columns(columns), before.select_columns(columns)
            )


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_processor_state_moves_to_dtype(case: ProcessorCase) -> None:
    dtype = torch.float64
    context, query = case.make_inputs()
    target_context = context.replace_blocks(
        numerical=context.numerical.to(dtype)
    )
    target_query = query.replace_blocks(numerical=query.numerical.to(dtype))
    processor = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    source = EnsembleTable.from_tables(
        tables=(context, query), member_table_ids=(1, 0, 1, 0)
    )
    target = EnsembleTable.from_tables(
        tables=(target_context, target_query),
        member_table_ids=(1, 0, 1, 0),
    )
    processor.fit_ensemble(
        source,
        generator=torch.Generator().manual_seed(0),
    )
    expected = processor.transform_ensemble(source)
    processor.to(dtype=dtype)
    actual = processor.transform_ensemble(target)
    assert all(
        actual.table(member_id).dtype == dtype
        for member_id in range(actual.num_members)
    )
    _assert_data_close(actual, expected, check_dtype=False)


@onlyCUDA
@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_processor_state_moves_to_cuda(case: ProcessorCase) -> None:
    device = torch.device("cuda")
    context, query = case.make_inputs()
    processor = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    source = EnsembleTable.from_tables(
        tables=(context, query), member_table_ids=(1, 0, 1, 0)
    )
    target = source.replace_groups(
        [cast(TableTensor, group.to(device)) for group in source]
    )
    processor.fit_ensemble(
        source,
        generator=torch.Generator().manual_seed(0),
    )
    expected = processor.transform_ensemble(source)
    processor.to(device=device)
    actual = processor.transform_ensemble(target)
    assert all(
        actual.table(member_id).device.type == device.type
        for member_id in range(actual.num_members)
    )
    _assert_data_close(actual, expected)
