import pytest
import torch
from sdm.models.tabfm.embedding import CellEmbedder
from sdm.testing import withCUDA


def _reference_group(
    x: torch.Tensor,
    group_size: int,
    d: torch.Tensor | None,
) -> torch.Tensor:
    _, _, num_features = x.shape
    feature_index = torch.arange(num_features, device=x.device)
    grouped = []
    for index in range(group_size):
        offset = (2**index) - 1
        if d is None:
            grouped.append(x[..., (feature_index + offset) % num_features])
            continue

        gather_index = (feature_index[None] + offset) % d[:, None].clamp_min(1)
        gather_index = gather_index[:, None].expand(-1, x.size(1), -1)
        grouped.append(x.gather(dim=-1, index=gather_index))
    return torch.stack(grouped, dim=-1)


def _reference_forward(
    module: CellEmbedder,
    x: torch.Tensor,
    cat_mask: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    grouped = _reference_group(x, module.feature_group_size, d)[..., None]
    grouped = grouped.float()
    angles = grouped * module.fourier_frequencies.float()
    fourier = torch.cat([angles.sin(), angles.cos()], dim=-1).to(x.dtype)
    numerical = module.in_linear(fourier)
    angles = grouped * module.fourier_frequencies_cat.float()
    fourier = torch.cat([angles.sin(), angles.cos()], dim=-1).to(x.dtype)
    categorical = module.in_linear_cat(fourier)
    grouped_mask = _reference_group(
        cat_mask[:, None].float(),
        module.feature_group_size,
        d,
    ).bool()[..., None]
    cell = torch.where(grouped_mask, categorical, numerical).sum(dim=-2)
    output = cell
    feature = torch.arange(x.size(2), device=x.device)
    return output.masked_fill(
        ~(feature[None] < d[:, None])[:, None, :, None], 0
    )


def test_cell_embedder_group_offsets_and_active_width() -> None:
    module = CellEmbedder(
        channels=4,
        feature_group_size=3,
        num_frequencies=2,
    )
    x = torch.arange(5, dtype=torch.float32).view(1, 1, 5)

    expected = torch.tensor(
        [[[[0, 1, 3], [1, 2, 4], [2, 3, 0], [3, 4, 1], [4, 0, 2]]]],
        dtype=x.dtype,
    )
    active_expected = torch.tensor(
        [[[[0, 1, 0], [1, 2, 1], [2, 0, 2], [0, 1, 0], [1, 2, 1]]]],
        dtype=x.dtype,
    )

    torch.testing.assert_close(module._group(x), expected)
    torch.testing.assert_close(
        module._group(x, d=torch.tensor([3])),
        active_expected,
    )


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cell_embedder_mixed_features(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(0)
    module = CellEmbedder(
        channels=4,
        feature_group_size=3,
        num_frequencies=2,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        module.fourier_frequencies.normal_()
        module.fourier_frequencies_cat.normal_()
    x = torch.randn(2, 5, 4, device=device, dtype=dtype)
    cat_mask = torch.tensor(
        [[False, True, False, True], [True, False, True, False]],
        device=device,
    )

    d = torch.tensor([4, 2], device=device)
    output = module(
        x=x,
        cat_mask=cat_mask,
        d=d,
    )
    expected = _reference_forward(
        module=module,
        x=x,
        cat_mask=cat_mask,
        d=d,
    )

    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
    assert output.shape == (2, 5, 4, 4)
    assert output.dtype == dtype
    assert output.device == device
    assert torch.count_nonzero(output[1, :, 2:]) == 0


def test_cell_embedder_routes_categorical_groups() -> None:
    module = CellEmbedder(
        channels=1,
        feature_group_size=1,
        num_frequencies=1,
    )
    with torch.no_grad():
        module.fourier_frequencies.fill_(1)
        module.fourier_frequencies_cat.fill_(2)
        module.in_linear.weight.fill_(1)
        module.in_linear.bias.zero_()
        module.in_linear_cat.weight.fill_(2)
        module.in_linear_cat.bias.zero_()
    x = torch.tensor([[[0.5, 1.0]]])
    cat_mask = torch.tensor([[False, True]])

    output = module(
        x=x,
        cat_mask=cat_mask,
    )

    numerical = x[0, 0, 0].sin() + x[0, 0, 0].cos()
    categorical = 2 * ((2 * x[0, 0, 1]).sin() + (2 * x[0, 0, 1]).cos())
    torch.testing.assert_close(
        output[0, 0, :, 0],
        torch.stack([numerical, categorical]),
    )


@pytest.mark.parametrize(
    "channels",
    [
        0,
    ],
)
def test_cell_embedder_rejects_invalid_configuration(
    channels: int,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        CellEmbedder(channels=channels)


def test_cell_embedder_rejects_invalid_inputs() -> None:
    module = CellEmbedder(channels=4)
    x = torch.randn(2, 5, 4)

    with pytest.raises(ValueError, match="cat_mask"):
        module(
            x=x,
            cat_mask=torch.zeros(2, 4),
        )
