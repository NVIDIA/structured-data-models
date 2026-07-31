import pytest
import torch

from sdm import Stype, TableTensor
from sdm.processing import VariableSchemaProcessor


class _CenterRepresentations(VariableSchemaProcessor):
    supported_stypes = frozenset({Stype.numerical})

    def __init__(self) -> None:
        super().__init__()
        self.means: torch.Tensor | None = None

    def _transform(self, table: TableTensor) -> TableTensor:
        return table

    def _fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        del generator
        self.means = table.numerical.mean(dim=-2, keepdim=True)

    def _transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]:
        assert self.means is not None
        return tuple(
            table[index].replace_blocks(
                numerical=table[index].numerical - self.means[index],
            )
            for index in range(table.size(0))
        )


def _table() -> TableTensor:
    return TableTensor.from_tensor(
        torch.tensor(
            [
                [[1.0, 3.0], [3.0, 5.0]],
                [[10.0, 20.0], [14.0, 28.0]],
            ]
        )
    )


def _assert_centered(output: tuple[TableTensor, ...]) -> None:
    assert len(output) == 2
    for table in output:
        torch.testing.assert_close(
            table.numerical.mean(dim=-2),
            table.numerical.new_zeros(2),
        )


def test_variable_schema_processor_fit_and_transform_batch() -> None:
    table = _table()
    processor = _CenterRepresentations()

    with pytest.raises(RuntimeError, match="fit_batch"):
        processor.transform_batch(table)

    assert processor.fit_batch(table) is processor
    _assert_centered(processor.transform_batch(table))


def test_variable_schema_processor_fit_transform_batch() -> None:
    output = _CenterRepresentations().fit_transform_batch(_table())

    _assert_centered(output)


def test_variable_schema_processor_tracks_batch_fit_independently() -> None:
    table = _table()

    single_fitted = _CenterRepresentations().fit(table[0])
    with pytest.raises(RuntimeError, match="fit_batch"):
        single_fitted.transform_batch(table)

    batch_fitted = _CenterRepresentations().fit_batch(table)
    with pytest.raises(RuntimeError, match="not fitted"):
        batch_fitted.transform(table[0])
