import inspect
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

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
from sdm.tensor import EnsembleTable
from sdm.testing import onlyCUDA


def make_mixed_inputs() -> EnsembleTable:
    categories = (StringTensor.from_list(["a", "b"]),)
    numerical = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    numerical[:, 1] = 1
    context = TableTensor(
        numerical=numerical,
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [0], [1], [-1]], dtype=torch.int32),
            categories=categories,
        ),
        datetime=torch.arange(4, dtype=torch.int64)[:, None] * 86_400_000_000,
        text=StringTensor.from_list(
            [["alpha beta"], ["beta"], ["gamma"], ["alpha"]],
        ),
        id=ColumnarTensor((torch.arange(10, 14),)),
    )
    query = TableTensor(
        numerical=numerical + 2,
        categorical=CategoricalTensor(
            code=torch.tensor([[-1], [1], [0], [-1]], dtype=torch.int32),
            categories=categories,
        ),
        datetime=context.datetime + 60_000_000,
        text=StringTensor.from_list(
            [["alpha"], ["beta gamma"], ["unseen"], [""]],
        ),
        id=ColumnarTensor((torch.arange(20, 24),)),
    )
    return EnsembleTable.from_tables(
        tables=(context, query), member_table_ids=(0, 1, 0, 1)
    )


def _make_impute_mean_inputs() -> EnsembleTable:
    data = make_mixed_inputs()
    context, query = data.table(0), data.table(1)
    numerical = query.numerical.clone()
    numerical[0, 0] = float("nan")
    return EnsembleTable.from_tables(
        tables=(context, query.replace_blocks(numerical=numerical)),
        member_table_ids=(0, 1, 0, 1),
    )


def _make_align_categories_inputs() -> EnsembleTable:
    data = make_mixed_inputs()
    context, query = data.table(0), data.table(1)
    categorical = CategoricalTensor(
        code=torch.tensor([[0], [1], [-1], [0]], dtype=torch.int32),
        categories=(StringTensor.from_list(["b", "c"]),),
    )
    return EnsembleTable.from_tables(
        tables=(context, query.replace_blocks(categorical=categorical)),
        member_table_ids=(0, 1, 0, 1),
    )


def _make_reduction_inputs() -> EnsembleTable:
    context = TableTensor.from_tensor(
        torch.tensor(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            dtype=torch.float32,
        )
    )
    query = TableTensor.from_tensor(context.numerical + 1.0)
    return EnsembleTable.from_tables(
        tables=(context, query), member_table_ids=(0, 1, 0, 1)
    )


@dataclass(frozen=True)
class ProcessorCase:
    processor: sp.Processor
    make_inputs: Callable[[], EnsembleTable] = make_mixed_inputs


def _make_processor_pair(
    processor: sp.Processor,
) -> tuple[sp.EnsembleProcessor, sp.EnsembleProcessor]:
    return (
        sp.EnsembleProcessor.as_processor(deepcopy(processor)),
        sp.EnsembleProcessor.as_processor(deepcopy(processor)),
    )


PROCESSOR_CASES = (
    ProcessorCase(sp.Identity()),
    ProcessorCase(sp.Callable(lambda table: table)),
    ProcessorCase(sp.DropStypes(Stype.id)),
    ProcessorCase(sp.ToNumerical()),
    ProcessorCase(sp.ShuffleColumns()),
    ProcessorCase(sp.SelectColumns(2)),
    ProcessorCase(sp.TFIDF(ngram_range=(2, 2))),
    ProcessorCase(sp.Clip(-2.0, 6.0)),
    ProcessorCase(sp.ClipQuantiles()),
    ProcessorCase(sp.ClipSigma()),
    ProcessorCase(sp.ImputeMean(), _make_impute_mean_inputs),
    ProcessorCase(sp.PowerTransform()),
    ProcessorCase(
        sp.QuantileTransform(n_quantiles=4, subsample=None),
    ),
    ProcessorCase(sp.Standardize()),
    ProcessorCase(sp.DropConstantColumns()),
    ProcessorCase(sp.PCA(2)),
    ProcessorCase(sp.RandomProjection(2)),
    ProcessorCase(sp.AlignCategories(), _make_align_categories_inputs),
    ProcessorCase(sp.ShuffleCategories()),
    ProcessorCase(sp.ImputeMode()),
    ProcessorCase(sp.AddCalendarFields(["month"])),
    ProcessorCase(sp.Softmax()),
    ProcessorCase(sp.ReduceEstimators(), _make_reduction_inputs),
    ProcessorCase(sp.EnsembleProcessorAdapter(sp.Standardize())),
    ProcessorCase(
        sp.Sequential(
            sp.StypeDispatch(
                numerical=sp.Choice(
                    sp.Standardize(),
                    sp.ShuffleColumns(),
                    method="round_robin",
                )
            ),
            sp.StypeDispatch(numerical=sp.ShuffleColumns()),
        ),
    ),
    ProcessorCase(
        sp.StypeDispatch(
            numerical=sp.Standardize(),
            categorical=sp.Identity(),
        ),
    ),
    ProcessorCase(
        sp.Choice(
            sp.Standardize(),
            sp.ShuffleColumns(),
            method="round_robin",
        ),
    ),
)


def test_all_public_processors_have_contract_cases() -> None:
    public_processors = set()
    for name in sp.__all__:
        value = getattr(sp, name)
        if (
            isinstance(value, type)
            and issubclass(value, sp.Processor)
            and not inspect.isabstract(value)
        ):
            public_processors.add(value)

    covered_processors = {type(case.processor) for case in PROCESSOR_CASES}
    specialized_processors = {
        sp.TaskDispatch,
        sp.TableDispatch,
        sp.SentenceTransformer,
    }
    assert public_processors == covered_processors | specialized_processors


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_fit_transform_matches_fit_then_transform(
    case: ProcessorCase,
) -> None:
    data = case.make_inputs()
    processor_a, processor_b = _make_processor_pair(case.processor)
    processor_a.fit_ensemble(
        data,
        generator=torch.Generator().manual_seed(0),
    )
    actual = processor_a.transform_ensemble(data)
    expected = processor_b.fit_transform_ensemble(
        data,
        generator=torch.Generator().manual_seed(0),
    )
    assert actual.num_members == expected.num_members
    for member_id in range(actual.num_members):
        assert actual.table(member_id).equal(expected.table(member_id))


@pytest.mark.parametrize(
    "case",
    tuple(case for case in PROCESSOR_CASES if case.processor.requires_fit),
    ids=lambda case: type(case.processor).__name__,
)
def test_save_and_load_preserves_fitted_processor_behavior(
    case: ProcessorCase,
) -> None:
    data = case.make_inputs()
    processor_a, processor_b = _make_processor_pair(case.processor)
    processor_a.fit_ensemble(data)
    expected = processor_a.transform_ensemble(data)
    processor_b.load_state_dict(processor_a.state_dict())
    actual = processor_b.transform_ensemble(data)
    assert actual.num_members == expected.num_members
    for member_id in range(actual.num_members):
        assert actual.table(member_id).equal(expected.table(member_id))


@pytest.mark.parametrize(
    "case",
    tuple(
        case
        for case in PROCESSOR_CASES
        if isinstance(case.processor, sp.InvertibleMixin)
    ),
    ids=lambda case: type(case.processor).__name__,
)
def test_inverse_transform_round_trip(case: ProcessorCase) -> None:
    data = case.make_inputs()
    processor = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    transformed = processor.fit_transform_ensemble(data)
    restored = cast(
        sp.EnsembleInvertibleMixin, processor
    ).inverse_transform_ensemble(transformed)
    assert restored.num_members == data.num_members
    for member_id in range(restored.num_members):
        torch.testing.assert_close(
            restored.table(member_id).numerical,
            data.table(member_id).numerical,
            atol=1e-3,
            rtol=1e-5,
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_preserves_rows_and_unhandled_stypes(
    case: ProcessorCase,
) -> None:
    data = case.make_inputs()
    processor = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    output = processor.fit_transform_ensemble(data)
    for member_id in range(output.num_members):
        before = data.table(member_id)
        after = output.table(member_id)
        assert after.size()[:-1] == before.size()[:-1]
        for stype in before.active_stypes - processor.handles_stypes:
            columns = before.columns[stype]
            assert all(column in after.column_names for column in columns)
            assert all(after.stype(column) == stype for column in columns)
            assert after.select_columns(columns).equal(
                before.select_columns(columns)
            )


@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_processor_state_moves_to_dtype(case: ProcessorCase) -> None:
    dtype = torch.float64
    source = case.make_inputs()
    target = source.replace_groups(
        [
            group.replace_blocks(numerical=group.numerical.to(dtype))
            for group in source
        ]
    )
    processor = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    processor.fit_ensemble(source)
    expected = processor.transform_ensemble(source)
    processor.to(dtype=dtype)
    actual = processor.transform_ensemble(target)
    assert all(
        actual.table(member_id).dtype == dtype
        for member_id in range(actual.num_members)
    )
    assert actual.num_members == expected.num_members
    for member_id in range(actual.num_members):
        actual_table = actual.table(member_id)
        expected_table = expected.table(member_id)
        assert actual_table.columns == expected_table.columns
        torch.testing.assert_close(
            actual_table.numerical,
            expected_table.numerical,
            equal_nan=True,
            check_dtype=False,
        )
        assert actual_table.categorical.equal(expected_table.categorical)


@onlyCUDA
@pytest.mark.parametrize(
    "case",
    PROCESSOR_CASES,
    ids=lambda case: type(case.processor).__name__,
)
def test_processor_state_moves_to_cuda(case: ProcessorCase) -> None:
    device = torch.device("cuda")
    source = case.make_inputs()
    target = source.replace_groups(
        [cast(TableTensor, group.to(device)) for group in source]
    )
    processor = sp.EnsembleProcessor.as_processor(deepcopy(case.processor))
    processor.fit_ensemble(source)
    expected = processor.transform_ensemble(source)
    processor.to(device=device)
    actual = processor.transform_ensemble(target)
    assert all(
        actual.table(member_id).device.type == device.type
        for member_id in range(actual.num_members)
    )
    assert actual.num_members == expected.num_members
    for member_id in range(actual.num_members):
        actual_table = actual.table(member_id)
        expected_table = expected.table(member_id)
        assert actual_table.columns == expected_table.columns
        torch.testing.assert_close(
            actual_table.numerical.cpu(),
            expected_table.numerical,
            equal_nan=True,
        )
        assert actual_table.categorical.cpu().equal(expected_table.categorical)
