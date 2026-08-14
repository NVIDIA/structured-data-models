import copy

import torch

from sdm.models.tabfm.column import _ColumnEmbedding


def _module(
    *,
    col_chunk_size: int | None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> _ColumnEmbedding:
    return _ColumnEmbedding(
        channels=4,
        num_blocks=2,
        num_heads=2,
        feedforward_channels=6,
        num_inducing_points=3,
        col_chunk_size=col_chunk_size,
        device=device,
        dtype=dtype,
    )


def test_column_chunking_preserves_outputs() -> None:
    unchunked = _module(col_chunk_size=None)
    chunked = copy.deepcopy(unchunked)
    chunked.col_chunk_size = 2
    x = torch.linspace(-1.0, 1.0, 72).view(2, 3, 3, 4)
    context_size = torch.tensor([1, 2])

    torch.testing.assert_close(
        chunked(x, context_size),
        unchunked(x, context_size),
    )


def test_column_embedding_isolates_columns() -> None:
    module = _module(col_chunk_size=None)
    x = torch.linspace(-1.0, 1.0, 32).view(1, 4, 2, 4)
    changed = x.clone()
    changed[:, 2, 0].add_(10)

    output = module(x, torch.tensor([2]))
    changed_output = module(changed, torch.tensor([2]))

    torch.testing.assert_close(output[:, :, 1], changed_output[:, :, 1])
    assert not torch.allclose(output[:, 2, 0], changed_output[:, 2, 0])
