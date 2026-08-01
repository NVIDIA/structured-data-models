import torch

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, NumericalProcessor


class _Center(NumericalProcessor, InvertibleMixin):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.mean = table.numerical.mean(dim=0)

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical - self.mean)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(numerical=table.numerical + self.mean)


def _integer_table(values: list[list[int]]) -> TableTensor:
    return TableTensor(
        columns={Stype.numerical: ("a", "b")},
        numerical=torch.tensor(values),
    )


def test_numerical_processor_promotes_all_inputs_to_float() -> None:
    table = _integer_table([[1, 3], [3, 5]])
    processor = _Center().fit(table)

    transformed = processor.transform(table)
    fit_transformed = _Center().fit_transform(table)
    inverted = processor.inverse_transform(_integer_table([[0, 0]]))

    expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    assert transformed.numerical.dtype == torch.get_default_dtype()
    assert fit_transformed.numerical.dtype == torch.get_default_dtype()
    assert inverted.numerical.dtype == torch.get_default_dtype()
    torch.testing.assert_close(transformed.numerical, expected)
    torch.testing.assert_close(fit_transformed.numerical, expected)
    torch.testing.assert_close(inverted.numerical, torch.tensor([[2.0, 4.0]]))
    assert transformed.columns == table.columns
    assert table.numerical.dtype == torch.int64
