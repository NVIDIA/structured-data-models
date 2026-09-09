import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, cast

import pyarrow as pa
import pytest
import torch

from sdm import ColumnarTensor, RelationalData, TableTensor, TaskLink
from sdm.relational.backend import PyGLibRelationalSampler
from sdm.relational.join import join_index

SamplerFactory = Callable[[TableTensor], PyGLibRelationalSampler]


@pytest.fixture
def sampler_factory(monkeypatch: pytest.MonkeyPatch) -> SamplerFactory:
    # Isolate CPU seed resolution from the optional sampling kernel.
    monkeypatch.setitem(sys.modules, "pyg_lib", ModuleType("pyg_lib"))

    def sample(**kwargs: Any) -> tuple[None, None, dict[str, torch.Tensor]]:
        nodes = {
            name: torch.stack((torch.arange(seed.numel()), seed), dim=1)
            for name, seed in kwargs["seed_dict"].items()
        }
        return None, None, nodes

    monkeypatch.setattr(
        torch.ops.pyg, "hetero_neighbor_sample", sample, raising=False
    )

    def make(table: TableTensor) -> PyGLibRelationalSampler:
        return PyGLibRelationalSampler(
            RelationalData(tables={"entities": table}, relationships=[]), {}
        )

    return make


def _table(values: list[Any], dtype: pa.DataType | None = None) -> TableTensor:
    return TableTensor.from_arrow(
        pa.table({"id": pa.array(values, type=dtype or pa.int64())}),
        stypes={"id": "id"},
    )


def _check(sampler: PyGLibRelationalSampler, task: TableTensor) -> None:
    columns = tuple(task.to_arrow().column_names)
    link = TaskLink(
        task_columns=columns, table="entities", table_columns=columns
    )
    left, right = join_index(
        task, sampler.data.tables["entities"], columns, columns
    )
    left, order = left.sort()
    if not left.equal(torch.arange(len(task))):
        with pytest.raises(ValueError, match="match exactly one row"):
            sampler.sample(task, link, [0])
        return
    result = sampler.sample(task, link, [0])
    if len(task) == 0:
        assert result == {}
    else:
        examples, rows = result["entities"]
        assert examples.equal(torch.arange(len(task)))
        assert rows.equal(right[order])


@pytest.mark.parametrize(
    ("entity", "query", "dtype"),
    [
        ([90, -8, 12, 4], [4, 90, 4, -8], pa.int64()),
        ([90, -8, 12, 4], [13], pa.int64()),
        ([90, -8, 12, 4], [100], pa.int64()),
        ([1, 2], [], pa.int64()),
        ([], [1], pa.int64()),
        ([1, 1, 2], [2], pa.int64()),
        ([1, 1, 2], [1], pa.int64()),
        ([1, None, 2], [2], pa.int64()),
        ([1, None, 2], [None], pa.int64()),
        ([1, 2, 3], [None, 2], pa.int64()),
        ([-(2**63), 2**63 - 1], [2**63 - 1, -(2**63)], pa.int64()),
        ([3, 0, 255], [255, 0], pa.uint8()),
        ([-128, 0, 127], [127, -128, 0], pa.int8()),
        ([-32768, 0, 32767], [32767, -32768, 0], pa.int16()),
        ([-(2**31), 0, 2**31 - 1], [2**31 - 1, -(2**31), 0], pa.int32()),
        (["b", "a"], ["a", "b", "a"], pa.string()),
    ],
)
def test_seed_lookup_matches_join(
    sampler_factory: SamplerFactory,
    entity: list[Any],
    query: list[Any],
    dtype: pa.DataType,
) -> None:
    sampler = sampler_factory(_table(entity, dtype))
    for _ in range(2):
        _check(sampler, _table(query, dtype))


def test_seed_lookup_tracks_key_changes(
    sampler_factory: SamplerFactory,
) -> None:
    sampler = sampler_factory(_table([20, 40, 10]))
    tables = cast(dict[str, TableTensor], sampler.data.tables)
    _check(sampler, _table([10, 20]))
    sampler.data.tables["entities"].id[..., 0].copy_(
        torch.tensor([10, 20, 40])
    )
    _check(sampler, _table([10, 20]))
    tables["entities"] = _table([40, 10, 20])
    _check(sampler, _table([10, 20]))
    tables["entities"] = tables["entities"][torch.tensor([2, 1, 0])]
    _check(sampler, _table([10, 20]))


def test_seed_lookup_inference_and_strided_keys(
    sampler_factory: SamplerFactory,
) -> None:
    keys = torch.tensor([20, 0, 40, 0, 10, 0])[::2]
    table = TableTensor(columns={"id": ("id",)}, id=ColumnarTensor((keys,)))
    sampler = sampler_factory(table)
    with torch.inference_mode():
        _check(sampler, _table([10, 20]))
        table = _table([20, 40, 10])
        sampler = sampler_factory(table)
        _check(sampler, _table([10, 20]))
        table.id[..., 0].copy_(torch.tensor([10, 20, 40]))
        _check(sampler, _table([10, 20]))


def test_seed_lookup_composite_keys(sampler_factory: SamplerFactory) -> None:
    table = TableTensor.from_arrow(
        pa.table({"id": [1, 1, 2], "region": ["a", "b", "a"]}),
        stypes={"id": "id", "region": "id"},
    )
    sampler = sampler_factory(table)
    _check(sampler, table[torch.tensor([2, 0, 1, 0])])


def test_seed_lookup_mixed_dtypes_preserves_join_error(
    sampler_factory: SamplerFactory,
) -> None:
    sampler = sampler_factory(_table([1, 2, 3]))
    with pytest.raises(pa.ArrowInvalid, match="Incompatible data types"):
        sampler.sample(
            _table([2, 1], pa.int32()),
            TaskLink(
                task_columns=("id",), table="entities", table_columns=("id",)
            ),
            [0],
        )
