import pickle
from textwrap import dedent
from typing import Any, cast

import pytest
import torch

import sdm.processing as sp
from sdm import (
    CategoricalTensor,
    StringTensor,
    TableTensor,
)
from sdm.tensor import EnsembleTable


def _mixed_table(numerical: torch.Tensor | None = None) -> TableTensor:
    if numerical is None:
        numerical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    n_rows = numerical.shape[0]
    categorical = CategoricalTensor(
        code=(torch.arange(n_rows) % 2).unsqueeze(1),
        categories=(StringTensor.from_list(["a", "b"]),),
    )
    return TableTensor(
        columns={
            "numerical": ("x0", "x1"),
            "categorical": ("kind",),
        },
        numerical=numerical,
        categorical=categorical,
    )


def _table(numerical: torch.Tensor | None = None) -> TableTensor:
    if numerical is None:
        numerical = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return TableTensor.from_tensor(numerical, columns=("x0", "x1"))


def _add_one(table: TableTensor) -> TableTensor:
    return table.replace_blocks(numerical=table.numerical + 1)


def test_empty_pipeline_returns_input_table() -> None:
    table = _table()

    assert sp.Sequential().transform(table).equal(table)
    assert sp.Sequential().fit_transform(table).equal(table)
    assert sp.Sequential().inverse_transform(table).equal(table)


def test_pipeline_transforms_numerical() -> None:
    table = _table()

    output = sp.Sequential(sp.Standardize()).fit_transform(table)

    assert not torch.equal(output.numerical, table.numerical)


def test_pipeline_accepts_lambda() -> None:
    table = _table()
    pipeline = sp.Sequential(
        lambda table: table.replace_blocks(numerical=table.numerical.square())
    )

    output = pipeline.transform(table)

    assert not pipeline.requires_fit
    assert torch.equal(output.numerical, table.numerical.square())


def test_pipeline_accepts_regular_callable() -> None:
    table = _table()
    pipeline = sp.Sequential(_add_one)

    assert pipeline.fit(table) is pipeline
    restored = pickle.loads(pickle.dumps(pipeline))

    assert torch.equal(
        restored.transform(table).numerical,
        table.numerical + 1,
    )


def test_pipeline_mixes_processors_and_callables() -> None:
    table = _table(
        torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [4.0, 8.0]],
        )
    )
    pipeline = sp.Sequential(
        lambda table: table.replace_blocks(numerical=table.numerical.square()),
        sp.Standardize(),
    )
    expected = sp.Standardize().fit_transform(
        table.replace_blocks(numerical=table.numerical.square())
    )

    assert pipeline.fit(table) is pipeline
    output = pipeline.transform(table)

    torch.testing.assert_close(output.numerical, expected.numerical)


def test_pipeline_accepts_nested_sequential_with_callable() -> None:
    table = _table()
    pipeline = sp.Sequential(
        lambda table: table.replace_blocks(numerical=table.numerical + 1),
        sp.Sequential(
            lambda table: table.replace_blocks(numerical=table.numerical * 2)
        ),
    )

    output = pipeline.transform(table)

    assert torch.equal(output.numerical, (table.numerical + 1) * 2)


def test_pipeline_append_updates_requires_fit() -> None:
    pipeline = sp.Sequential(lambda table: table)
    assert not pipeline.requires_fit

    assert pipeline.append(sp.Standardize()) is pipeline

    assert pipeline.requires_fit


def test_pipeline_prepend_updates_order_and_requires_fit() -> None:
    table = _table()
    pipeline = sp.Sequential(
        lambda table: table.replace_blocks(numerical=table.numerical * 2)
    )

    assert pipeline.prepend(sp.Standardize()) is pipeline

    expected = sp.Sequential(
        sp.Standardize(),
        lambda table: table.replace_blocks(numerical=table.numerical * 2),
    ).fit_transform(table)
    output = pipeline.fit_transform(table)

    assert pipeline.requires_fit
    assert output.equal(expected)


def test_pipeline_prepend_flattens_nested_sequential() -> None:
    pipeline = sp.Sequential(_add_one)

    pipeline.prepend(
        sp.Sequential(
            lambda table: table.replace_blocks(
                numerical=table.numerical.square()
            ),
            lambda table: table.replace_blocks(numerical=table.numerical * 2),
        )
    )
    output = pipeline.transform(_table())

    assert len(pipeline) == 3
    assert torch.equal(output.numerical, _table().numerical.square() * 2 + 1)


def test_pipeline_prepend_invalidates_fitted_state() -> None:
    pipeline = sp.Sequential(sp.Standardize()).fit(_table())

    pipeline.prepend(lambda table: table)

    with pytest.raises(RuntimeError, match="'Sequential' is not fitted"):
        pipeline.transform(_table())


def test_pipeline_rejects_invalid_step() -> None:
    with pytest.raises(TypeError, match=r"Input must be"):
        sp.Sequential(cast(Any, object()))


def test_pipeline_passes_generator_to_steps() -> None:
    table = _table(torch.arange(200.0).view(100, 2))

    def _fit_transform(seed: int) -> TableTensor:
        return sp.Sequential(
            sp.ShuffleColumns(method="random"),
            sp.QuantileTransform(n_quantiles=6, subsample=32),
        ).fit_transform(
            table,
            generator=torch.Generator().manual_seed(seed),
        )

    first = _fit_transform(0)
    second = _fit_transform(0)

    assert first.equal(second)


def test_repr() -> None:
    assert repr(sp.Sequential()) == "Sequential()"
    assert repr(
        sp.Sequential(sp.Standardize(), sp.PowerTransform())
    ) == dedent("""\
        Sequential(
          Standardize(),
          PowerTransform(),
        )""")
    assert repr(sp.Sequential(lambda table: table)) == dedent("""\
        Sequential(
          Callable(<lambda>),
        )""")


def test_pipeline_checks_fitted_state() -> None:
    pipeline = sp.Sequential(sp.Softmax(), sp.Standardize())

    with pytest.raises(RuntimeError, match="'Sequential' is not fitted"):
        pipeline.transform(_table())


def test_pipeline_rejects_unsupported_stype() -> None:
    pipeline = sp.Sequential(sp.Standardize())

    with pytest.raises(ValueError, match="categorical"):
        pipeline.fit_transform(_mixed_table())


def test_inverse_transform_rejects_non_invertible_step() -> None:
    processor = sp.Sequential(sp.ImputeMean())
    transformed = processor.fit_transform(_table())

    with pytest.raises(AttributeError, match="inverse_transform"):
        processor.inverse_transform(transformed)


def test_inverse_transform_runs_steps_in_reverse_order() -> None:
    # PowerTransform and Standardize do not commute, so the round trip only
    # reconstructs the input if the inverse applies the steps in reverse.
    table = _table(
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        )
    )

    pipeline = sp.Sequential(sp.PowerTransform(), sp.Standardize())
    transformed = pipeline.fit_transform(table)
    restored = pipeline.inverse_transform(transformed)

    assert torch.allclose(restored.numerical, table.numerical, atol=1e-4)


def test_sequential_ensemble_matches_member_execution() -> None:
    first = _table(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    second = _table(torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )
    processor = sp.Sequential(
        sp.Sequential(
            lambda value: value.replace_blocks(
                numerical=value.numerical.square()
            )
        ),
        _add_one,
    )

    output = processor.fit_transform_ensemble(table)

    for member_id, source in enumerate((second, first, second)):
        expected = _add_one(
            source.replace_blocks(numerical=source.numerical.square())
        )
        assert output.table(member_id).equal(expected)


def test_empty_ensemble_pipeline_passes_through_members() -> None:
    first = _table(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    second = _table(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    table = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(1, 0, 1),
    )
    processor = sp.Sequential()

    for output in (
        processor.transform_ensemble(table),
        processor.fit_transform_ensemble(table),
        processor.inverse_transform_ensemble(table),
    ):
        for member_id in range(table.num_members):
            assert output.table(member_id).equal(table.table(member_id))
