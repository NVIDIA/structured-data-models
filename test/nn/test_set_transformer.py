import pytest
import torch
from sdm.nn import (
    InducedSelfAttentionBlock,
    SetTransformer,
)
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("qassmax", [False, True])
def test_induced_self_attention_block(
    device: torch.device,
    qassmax: bool,
) -> None:
    batch_size = 2
    set_size = 6
    channels = 8
    module = InducedSelfAttentionBlock(
        channels=channels,
        num_heads=2,
        feedforward_channels=16,
        num_inducing_points=4,
        qassmax=qassmax,
        device=device,
    )
    x = torch.randn(batch_size, set_size, channels, device=device)

    out = module(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device

    # `context_size` and the equivalent `context_mask` agree.
    context_size = 4
    context_mask = torch.arange(set_size, device=device) < context_size  # [S]
    out_size = module(x, context_size=context_size)
    out_mask = module(x, context_mask=context_mask)
    torch.testing.assert_close(out_size, out_mask)

    # Target elements beyond the context do not leak into context outputs.
    x_perturbed = x.clone()
    x_perturbed[:, context_size:] = 1000 * torch.randn_like(
        x_perturbed[:, context_size:]
    )
    out_perturbed = module(x_perturbed, context_size=context_size)
    torch.testing.assert_close(
        out_size[:, :context_size],
        out_perturbed[:, :context_size],
    )

    # Empty context falls back to attending over the full set.
    out_empty = module(x, context_size=0)
    torch.testing.assert_close(out_empty, out)


@withCUDA
def test_set_transformer(device: torch.device) -> None:
    batch_size = 2
    set_size = 6
    channels = 8
    module = SetTransformer(
        channels=channels,
        num_heads=2,
        feedforward_channels=16,
        num_layers=3,
        num_inducing_points=4,
        device=device,
    )
    x = torch.randn(batch_size, set_size, channels, device=device)

    out = module(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.device == x.device

    context_size = 4
    context_mask = torch.arange(set_size, device=device) < context_size
    out_size = module(x, context_size=context_size)
    out_mask = module(x, context_mask=context_mask)
    torch.testing.assert_close(out_size, out_mask)


def test_set_transformer_errors() -> None:
    channels = 8
    module = InducedSelfAttentionBlock(
        channels=channels,
        num_heads=2,
        feedforward_channels=16,
    )
    x = torch.randn(2, 6, channels)

    with pytest.raises(ValueError, match="Cannot pass both"):
        module(
            x,
            context_size=3,
            context_mask=torch.ones(6, dtype=torch.bool),
        )

    with pytest.raises(
        ValueError,
        match=r"`context_mask` must have dtype torch\.bool",
    ):
        module(x, context_mask=torch.ones(6, dtype=torch.float32))
