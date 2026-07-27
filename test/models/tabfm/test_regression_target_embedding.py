import torch
from sdm.models.tabfm.embedding import CellEmbedder
from torch.nn import Sequential


def test_cell_embedder_adds_regression_targets_to_context_only() -> None:
    module = CellEmbedder(
        channels=4,
        num_frequencies=2,
    )
    x = torch.randn(2, 5, 4)
    target = torch.randn(2, 5)
    train_size = torch.tensor([3, 2])

    output = module(x=x, target=target, train_size=train_size)
    feature_embedding = module(x=x)

    assert isinstance(module.y_embedder_lookup, Sequential)
    target_embedding = module.y_embedder_lookup(target[..., None])
    row = torch.arange(x.size(1))[None]
    expected = torch.where(
        (row < train_size[:, None])[..., None, None],
        feature_embedding + target_embedding[:, :, None],
        feature_embedding,
    )
    torch.testing.assert_close(output, expected)


def test_cell_embedder_ignores_regression_query_targets() -> None:
    module = CellEmbedder(channels=4, num_frequencies=2)
    x = torch.randn(2, 5, 4)
    target = torch.randn(2, 5)
    train_size = torch.tensor([3, 2])
    row = torch.arange(5)[None]
    changed = torch.where(row >= train_size[:, None], target + 1, target)

    baseline = module(x=x, target=target, train_size=train_size)
    output = module(x=x, target=changed, train_size=train_size)

    torch.testing.assert_close(output, baseline, rtol=0, atol=0)
