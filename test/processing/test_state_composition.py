import copy
import io
from collections.abc import Callable

import torch

import sdm.processing as sp
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.tensor import EnsembleTable


def _round_trip(processor: sp.Processor) -> dict[str, object]:
    stream = io.BytesIO()
    torch.save(copy.deepcopy(processor.state_dict()), stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


def _mixed_table(*, offset: float = 0.0, prefix: str = "") -> TableTensor:
    return TableTensor(
        columns={
            "numerical": (f"{prefix}value", f"{prefix}constant"),
            "categorical": (f"{prefix}category",),
        },
        numerical=torch.tensor(
            [[0.0 + offset, 1.0], [2.0 + offset, 1.0], [5.0 + offset, 1.0]]
        ),
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [-1]], dtype=torch.int32),
            categories=(StringTensor.from_list(["a", "b"]),),
        ),
    )


def _assert_ensemble_equal(
    actual: EnsembleTable,
    expected: EnsembleTable,
) -> None:
    assert actual.num_members == expected.num_members
    assert all(
        actual.table(member_id).equal(expected.table(member_id))
        for member_id in range(actual.num_members)
    )


def _nested_pipeline() -> sp.Sequential:
    return sp.Sequential(
        sp.StypeDispatch(
            numerical=sp.Sequential(sp.ImputeMean(), sp.Standardize()),
            categorical=sp.Choice(
                sp.ImputeMode(),
                sp.ShuffleCategories(),
                method="round_robin",
            ),
        ),
        sp.StypeDispatch(numerical=sp.ShuffleColumns()),
    )


def test_deeply_nested_ensemble_pipeline_state_dict_round_trip() -> None:
    ensemble = EnsembleTable.from_tables(
        tables=(_mixed_table(), _mixed_table(offset=10.0, prefix="other_")),
        member_table_ids=(1, 0, 1, 0),
    )
    processor = _nested_pipeline().fit_ensemble(ensemble)
    expected = processor.transform_ensemble(ensemble)

    restored = _nested_pipeline()
    restored.load_state_dict(_round_trip(processor))

    _assert_ensemble_equal(restored.transform_ensemble(ensemble), expected)


def _task_dispatch() -> sp.TaskDispatch:
    return sp.TaskDispatch(
        regression=sp.Sequential(sp.Standardize(), sp.ShuffleColumns())
    )


def test_task_dispatch_state_dict_round_trip() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0], [4.0]]))
    processor = _task_dispatch()
    processor._task = "regression"
    processor.fit(table)
    expected = processor.transform(table)

    restored = _task_dispatch()
    restored.load_state_dict(_round_trip(processor))

    assert restored.transform(table).equal(expected)


def _table_dispatch() -> sp.TableDispatch:
    return sp.TableDispatch(task=sp.Standardize())


def test_table_dispatch_state_dict_round_trip() -> None:
    table = TableTensor.from_tensor(torch.tensor([[1.0], [2.0], [4.0]]))
    processor = _table_dispatch()
    processor._route = "task"
    processor.fit(table)
    expected = processor.transform(table)

    restored = _table_dispatch()
    restored.load_state_dict(_round_trip(processor))

    assert restored.transform(table).equal(expected)


def test_ensemble_processor_state_dict_round_trip() -> None:
    table = _mixed_table()
    ensemble = EnsembleTable(table, num_members=4)
    factories: tuple[Callable[[], sp.EnsembleProcessor], ...] = (
        sp.DropConstantColumns,
        sp.ShuffleColumns,
        sp.ShuffleCategories,
        lambda: sp.RandomProjection(2),
    )

    for factory in factories:
        processor = factory().fit_ensemble(ensemble)
        expected = processor.transform_ensemble(ensemble)
        restored = factory()
        restored.load_state_dict(_round_trip(processor))
        _assert_ensemble_equal(restored.transform_ensemble(ensemble), expected)
