import pytest
import torch
from sdm.models.tabfm.embedding import CellEmbedder
from torch.nn import Embedding


def test_cell_embedder_ignores_query_targets() -> None:
    module = CellEmbedder(channels=4, max_classes=3, num_frequencies=2)
    x = torch.randn(2, 5, 4)
    target = torch.randint(0, 3, (2, 5))
    train_size = torch.tensor([3, 2])
    row = torch.arange(5)[None]
    changed = torch.where(row >= train_size[:, None], target + 1, target) % 3

    baseline = module(x=x, target=target, train_size=train_size)
    output = module(x=x, target=changed, train_size=train_size)

    torch.testing.assert_close(output, baseline, rtol=0, atol=0)


def test_cell_embedder_requires_a_valid_target_configuration() -> None:
    x = torch.randn(2, 5, 4)
    target = torch.zeros(2, 5)

    with pytest.raises(ValueError, match="max_classes"):
        CellEmbedder(channels=4)(
            x=x,
            target=target,
            train_size=torch.tensor([3, 2]),
        )
    with pytest.raises(ValueError, match="train_size"):
        CellEmbedder(channels=4, max_classes=3)(x=x, target=target)
    with pytest.raises(ValueError, match="target"):
        CellEmbedder(channels=4, max_classes=3)(
            x=x,
            target=target[:, :-1],
            train_size=torch.tensor([3, 2]),
        )
