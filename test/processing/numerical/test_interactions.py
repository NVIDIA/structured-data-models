import torch

from sdm import TableTensor
from sdm.processing import PairwiseInteractions


def test_pairwise_interactions_appends_products_and_differences() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = PairwiseInteractions(max_columns=12).transform(
        TableTensor.from_tensor(x)
    )
    assert out.numerical.shape == (2, 3 + 3 + 3)
    assert torch.equal(out.numerical[:, 3], x[:, 0] * x[:, 1])
    assert torch.equal(out.numerical[:, 6], x[:, 0] - x[:, 1])
    wide = TableTensor.from_tensor(torch.zeros(2, 13))
    assert PairwiseInteractions(max_columns=12).transform(
        wide
    ).numerical.shape == (2, 13)
