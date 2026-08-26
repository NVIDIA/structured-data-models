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

ProcessorData = TableTensor | EnsembleTable


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


def _fit(processor: sp.Processor, data: ProcessorData) -> None:
    if isinstance(processor, sp.EnsembleProcessor):
        processor.fit_ensemble(
            cast(EnsembleTable, data),
            generator=torch.Generator().manual_seed(0),
        )
    else:
        processor.fit(
            cast(TableTensor, data),
            generator=torch.Generator().manual_seed(0),
        )


def _transform(processor: sp.Processor, data: ProcessorData) -> ProcessorData:
    if isinstance(processor, sp.EnsembleProcessor):
        return processor.transform_ensemble(cast(EnsembleTable, data))
    return processor.transform(cast(TableTensor, data))


def _tables(data: ProcessorData) -> tuple[TableTensor, ...]:
    if isinstance(data, EnsembleTable):
        return tuple(data.table(i) for i in range(data.num_members))
    return (data,)


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
    actual: ProcessorData,
    expected: ProcessorData,
    *,
    atol: float = 1e-5,
    check_dtype: bool = True,
) -> None:
    actual_tables, expected_tables = _tables(actual), _tables(expected)
    assert len(actual_tables) == len(expected_tables)
    for actual_table, expected_table in zip(
        actual_tables, expected_tables, strict=True
    ):
        _assert_table_close(
            actual_table, expected_table, atol=atol, check_dtype=check_dtype
        )


def _check_state_dict_restoration(case: ProcessorCase) -> None:
    context, query = case.make_inputs()
    fitted = deepcopy(case.processor)
    restored = deepcopy(case.processor)
    if isinstance(fitted, sp.EnsembleProcessor):
        fit_data: ProcessorData = EnsembleTable.from_tables(
            tables=(context, query), member_table_ids=(1, 0, 1, 0)
        )
        query_data = fit_data
    else:
        fit_data = context
        query_data = query
    _fit(fitted, fit_data)
    expected = _transform(fitted, query_data)
    stream = io.BytesIO()
    torch.save(fitted.state_dict(), stream)
    stream.seek(0)
    restored.load_state_dict(torch.load(stream, weights_only=False))
    _assert_data_close(_transform(restored, query_data), expected)


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_fit_transform_matches_fit_then_transform(
    case: ProcessorCase,
) -> None:
    context, query = case.make_inputs()
    split = deepcopy(case.processor)
    fused = deepcopy(case.processor)
    data: ProcessorData = (
        EnsembleTable.from_tables(
            tables=(context, query), member_table_ids=(1, 0, 1, 0)
        )
        if isinstance(split, sp.EnsembleProcessor)
        else context
    )
    _fit(split, data)
    if isinstance(fused, sp.EnsembleProcessor):
        actual = fused.fit_transform_ensemble(
            cast(EnsembleTable, data),
            generator=torch.Generator().manual_seed(0),
        )
    else:
        actual = fused.fit_transform(
            cast(TableTensor, data),
            generator=torch.Generator().manual_seed(0),
        )
    _assert_data_close(actual, _transform(split, data))


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
    processor = deepcopy(case.processor)
    transformed = processor.fit_transform(
        context,
        generator=torch.Generator().manual_seed(0),
    )
    restored = cast(sp.InvertibleMixin, processor).inverse_transform(
        transformed
    )
    _assert_table_close(restored, context, atol=1e-3)


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_preserves_rows_and_unhandled_stypes(
    case: ProcessorCase,
) -> None:
    context, query = case.make_inputs()
    processor = deepcopy(case.processor)
    data: ProcessorData = (
        EnsembleTable.from_tables(
            tables=(context, query), member_table_ids=(1, 0, 1, 0)
        )
        if isinstance(processor, sp.EnsembleProcessor)
        else context
    )
    if isinstance(processor, sp.EnsembleProcessor):
        output = processor.fit_transform_ensemble(
            cast(EnsembleTable, data),
            generator=torch.Generator().manual_seed(0),
        )
    else:
        output = processor.fit_transform(
            cast(TableTensor, data),
            generator=torch.Generator().manual_seed(0),
        )
    for before, after in zip(_tables(data), _tables(output)):
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
    processor = deepcopy(case.processor)
    if isinstance(processor, sp.EnsembleProcessor):
        source: ProcessorData = EnsembleTable.from_tables(
            tables=(context, query), member_table_ids=(1, 0, 1, 0)
        )
        target: ProcessorData = EnsembleTable.from_tables(
            tables=(target_context, target_query),
            member_table_ids=(1, 0, 1, 0),
        )
        query_data = source
    else:
        source = context
        target = target_query
        query_data = query
    _fit(processor, source)
    expected = _transform(processor, query_data)
    processor.to(dtype=dtype)
    actual = _transform(processor, target)
    assert all(table.dtype == dtype for table in _tables(actual))
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
    target_context = cast(TableTensor, context.to(device))
    target_query = cast(TableTensor, query.to(device))
    processor = deepcopy(case.processor)
    if isinstance(processor, sp.EnsembleProcessor):
        source: ProcessorData = EnsembleTable.from_tables(
            tables=(context, query), member_table_ids=(1, 0, 1, 0)
        )
        target: ProcessorData = EnsembleTable.from_tables(
            tables=(target_context, target_query),
            member_table_ids=(1, 0, 1, 0),
        )
        query_data = source
    else:
        source = context
        target = target_query
        query_data = query
    _fit(processor, source)
    expected = _transform(processor, query_data)
    processor.to(device=device)
    actual = _transform(processor, target)
    assert all(table.device.type == device.type for table in _tables(actual))
    _assert_data_close(actual, expected)
