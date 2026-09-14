import torch

from sdm import TableTensor
from sdm.processing import ReduceEstimators, Sequential, SortQuantiles
from sdm.testing import withCUDA


@withCUDA
def test_sort_quantiles(device: torch.device) -> None:
    data = torch.randn(2, 3, device=device)
    table = TableTensor.from_tensor(data, columns=["q10", "q50", "q90"])

    out = SortQuantiles().transform(table)
    assert out.size() == table.size()
    assert out.columns == table.columns
    assert out.device == table.device
    torch.testing.assert_close(out.numerical, data.sort(dim=-1).values)


def test_sort_quantiles_aligns_estimators_before_reduction() -> None:
    data = torch.tensor([[[1.0, 3.0]], [[7.0, 5.0]]])
    table = TableTensor.from_tensor(data, columns=["q10", "q90"])

    out = Sequential(SortQuantiles(), ReduceEstimators()).transform(table)

    torch.testing.assert_close(out.numerical, torch.tensor([[3.0, 5.0]]))
