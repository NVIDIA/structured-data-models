import copy

import pytest
import torch

from sdm.models.tabfm.row import _RowInteraction


def _row_interaction(
    *,
    output_full: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    row_chunk_size: int | None = None,
) -> _RowInteraction:
    return _RowInteraction(
        num_blocks=2,
        channels=2,
        num_heads=1,
        feedforward_channels=3,
        num_cls=1,
        output_full=output_full,
        row_chunk_size=row_chunk_size,
        device=device,
        dtype=dtype,
    )


@pytest.mark.parametrize("output_full", [True, False])
@pytest.mark.parametrize("num_rows", [3, 0])
def test_row_chunking_preserves_outputs(
    output_full: bool,
    num_rows: int,
) -> None:
    unchunked = _RowInteraction(
        num_blocks=1,
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        num_cls=1,
        output_full=output_full,
        row_chunk_size=None,
    )
    chunked = copy.deepcopy(unchunked)
    chunked.row_chunk_size = 4
    x = torch.randn(2, num_rows, 4, 4)
    active_features = torch.tensor([3, 1])

    torch.testing.assert_close(
        chunked(x, active_features),
        unchunked(x, active_features),
    )


def test_padded_features_are_keys_only_when_active() -> None:
    module = _RowInteraction(
        num_blocks=2,
        channels=4,
        num_heads=2,
        feedforward_channels=6,
        num_cls=1,
        output_full=True,
        row_chunk_size=2,
    ).eval()
    x = torch.randn(2, 3, 5, 4)
    active_features = torch.tensor([3, 1])
    changed = x.clone()
    changed[0, :, 4:] += 100
    changed[1, :, 2:] -= 100

    output = module(x, active_features)
    changed_output = module(changed, active_features)

    torch.testing.assert_close(output[0, :, :4], changed_output[0, :, :4])
    torch.testing.assert_close(output[1, :, :2], changed_output[1, :, :2])
    assert not torch.equal(output[0, :, 4:], changed_output[0, :, 4:])
    assert not torch.equal(output[1, :, 2:], changed_output[1, :, 2:])

    module.output_full = False
    torch.testing.assert_close(
        module(x, active_features),
        module(changed, active_features),
    )
