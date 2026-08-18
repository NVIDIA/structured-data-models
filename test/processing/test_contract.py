import inspect
import io
from typing import cast

import pytest
import torch
from test.processing._instance_generator import (
    PROCESSOR_CASES,
    ProcessorCase,
    ProcessorInputs,
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
    registered = {type(case.factory()) for case in PROCESSOR_CASES}
    containers = {sp.TaskDispatch, sp.TableDispatch}
    processor_specific = {sp.SentenceTransformer}
    assert public == registered | containers | processor_specific


def _ensemble(inputs: ProcessorInputs) -> EnsembleTable:
    return EnsembleTable.from_tables(
        tables=(inputs.context, inputs.query), member_table_ids=(1, 0, 1, 0)
    )


def _data(
    processor: sp.Processor,
    inputs: ProcessorInputs,
    *,
    query: bool = False,
) -> ProcessorData:
    if isinstance(processor, sp.EnsembleProcessor):
        return _ensemble(inputs)
    return inputs.query if query else inputs.context


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


def _serialized_state(processor: sp.Processor) -> dict[str, object]:
    stream = io.BytesIO()
    torch.save(processor.state_dict(), stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


def _check_state_dict_restoration(case: ProcessorCase) -> None:
    inputs = case.make_inputs(torch.device("cpu"), torch.float32)
    fitted, restored = case.factory(), case.factory()
    fit_data = _data(fitted, inputs)
    query_data = _data(fitted, inputs, query=True)
    _fit(fitted, fit_data)
    expected = _transform(fitted, query_data)
    restored.load_state_dict(_serialized_state(fitted))
    _assert_data_close(_transform(restored, query_data), expected)


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.factory()).__name__,
)
def test_fit_transform_matches_fit_then_transform(
    case: ProcessorCase,
) -> None:
    inputs = case.make_inputs(torch.device("cpu"), torch.float32)
    split, fused = case.factory(), case.factory()
    data = _data(split, inputs)
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
    case for case in PROCESSOR_CASES if case.factory().requires_fit
)


@pytest.mark.parametrize(
    "case",
    FITTED_CASES,
    ids=lambda case: type(case.factory()).__name__,
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
    broken = ProcessorCase(_BrokenFittedProcessor, make_mixed_inputs)
    with pytest.raises(RuntimeError):
        _check_state_dict_restoration(broken)


INVERTIBLE_CASES = tuple(
    case
    for case in PROCESSOR_CASES
    if isinstance(case.factory(), sp.InvertibleMixin)
)


@pytest.mark.parametrize(
    "case",
    INVERTIBLE_CASES,
    ids=lambda case: type(case.factory()).__name__,
)
def test_inverse_transform_round_trip(case: ProcessorCase) -> None:
    inputs = case.make_inputs(torch.device("cpu"), torch.float32)
    processor = case.factory()
    transformed = processor.fit_transform(
        inputs.context,
        generator=torch.Generator().manual_seed(0),
    )
    restored = cast(sp.InvertibleMixin, processor).inverse_transform(
        transformed
    )
    _assert_table_close(restored, inputs.context, atol=1e-3)


def _assert_unhandled_stypes_pass_through(
    processor: sp.Processor,
    before: TableTensor,
    after: TableTensor,
) -> None:
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
    ids=lambda case: type(case.factory()).__name__,
)
def test_preserves_rows_and_unhandled_stypes(
    case: ProcessorCase,
) -> None:
    inputs = case.make_inputs(torch.device("cpu"), torch.float32)
    processor = case.factory()
    data = _data(processor, inputs)
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
        _assert_unhandled_stypes_pass_through(processor, before, after)


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.factory()).__name__,
)
def test_processor_state_moves_to_dtype(case: ProcessorCase) -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    source = case.make_inputs(torch.device("cpu"), torch.float32)
    target = case.make_inputs(device, dtype)
    processor = case.factory()
    _fit(processor, _data(processor, source))
    expected = _transform(processor, _data(processor, source, query=True))
    processor.to(device=device, dtype=dtype)
    actual = _transform(processor, _data(processor, target, query=True))
    assert all(
        table.device.type == device.type and table.dtype == dtype
        for table in _tables(actual)
    )
    _assert_data_close(actual, expected, check_dtype=dtype == torch.float32)


@onlyCUDA
@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.factory()).__name__,
)
def test_processor_state_moves_to_cuda(case: ProcessorCase) -> None:
    device = torch.device("cuda")
    dtype = torch.float32
    source = case.make_inputs(torch.device("cpu"), torch.float32)
    target = case.make_inputs(device, dtype)
    processor = case.factory()
    _fit(processor, _data(processor, source))
    expected = _transform(processor, _data(processor, source, query=True))
    processor.to(device=device, dtype=dtype)
    actual = _transform(processor, _data(processor, target, query=True))
    assert all(
        table.device.type == device.type and table.dtype == dtype
        for table in _tables(actual)
    )
    _assert_data_close(actual, expected)
