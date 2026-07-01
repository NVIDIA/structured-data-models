import pytest
import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from sdm.processing import (
    Clip,
    SigmaClip,
    StandardScale,
)
from sdm.testing import withCUDA


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int64, torch.float32])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_tabiclv2(
    device: torch.device,
    dtype: torch.dtype,
    batch_shape: tuple[int, ...],
) -> None:
    model = TabICLv2(pretrained=False, device=device)
    if device.type == "cpu":
        assert repr(model) == "TabICLv2()"
    else:
        assert repr(model) == "TabICLv2(device=cuda:0)"

    R, C, R_train = 8, 6, 5

    x = torch.randn(*batch_shape, R, C, device=device)
    if dtype.is_floating_point:
        y = torch.randn(*batch_shape, R_train, device=device)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, 999)
    else:
        y = torch.randint(0, 10, (*batch_shape, R_train), device=device)
        out = model(x, y)
        assert out.size() == (*batch_shape, R - R_train, 10)

    assert out.dtype == x.dtype
    assert out.device == x.device

    assert torch.allclose(model(x, y.unsqueeze(-1)), out)

    if len(batch_shape) > 0:
        looped = torch.stack(
            [model(x[i], y[i]) for i in range(batch_shape[0])]
        )
        assert torch.allclose(out, looped, atol=1e-5)


def test_tabiclv2_default_recipe_uses_tabicl_regression_n1_defaults() -> None:
    recipe = TabICLv2.default_recipe()

    assert tuple(type(step) for step in recipe.features) == (
        StandardScale,
        Clip,
        SigmaClip,
    )
    assert len(recipe.target) == 0
    assert len(recipe.output) == 0


def _nanstd(input: torch.Tensor, *, dim: int) -> torch.Tensor:
    mask = ~torch.isnan(input)
    count = mask.sum(dim=dim)
    mean = torch.nanmean(input, dim=dim)
    centered = input - mean
    centered = torch.where(mask, centered, torch.zeros_like(centered))
    sum_squares = centered.square().sum(dim=dim)
    denominator = (count - (count > 1).to(count.dtype)).clamp_min(1)
    return torch.where(
        count > 0,
        (sum_squares / denominator).sqrt(),
        torch.nan,
    )


def _original_tabiclv2_regression_n1_features(
    input: torch.Tensor,
) -> torch.Tensor:
    mean = input.mean(dim=0)
    scale = input.std(dim=0, correction=0) + 1e-6
    input = ((input - mean) / scale).clamp(min=-100.0, max=100.0)

    threshold = 4.0
    min_std = input.new_tensor(1e-6)
    mean = torch.nanmean(input, dim=0)
    std = torch.maximum(_nanstd(input, dim=0), min_std)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std
    outlier_mask = (input < lower_bound) | (input > upper_bound)
    clean = torch.where(outlier_mask, torch.nan, input)

    mean = torch.nanmean(clean, dim=0)
    std = torch.maximum(_nanstd(clean, dim=0), min_std)
    lower_bound = mean - threshold * std
    upper_bound = mean + threshold * std

    log_abs = torch.log1p(input.abs())
    clipped = torch.maximum(-log_abs + lower_bound, input)
    return torch.minimum(log_abs + upper_bound, clipped)


def _tabarena_california_housing_slice(device: torch.device) -> TableTensor:
    # A fixed numerical slice from the California Housing regression dataset.
    return TableTensor.from_tensor(
        torch.tensor(
            [
                [8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23],
                [8.3014, 21.0, 6.2381, 0.9719, 2401.0, 2.1098, 37.86, -122.22],
                [7.2574, 52.0, 8.2881, 1.0734, 496.0, 2.8023, 37.85, -122.24],
                [5.6431, 52.0, 5.8174, 1.0731, 558.0, 2.5479, 37.85, -122.25],
                [3.8462, 52.0, 6.2819, 1.0811, 565.0, 2.1815, 37.85, -122.25],
                [4.0368, 52.0, 4.7617, 1.1036, 413.0, 2.1399, 37.85, -122.25],
                [3.6591, 52.0, 4.9319, 0.9514, 1094.0, 2.1284, 37.84, -122.25],
                [3.12, 52.0, 4.7975, 1.0618, 1157.0, 1.7883, 37.84, -122.25],
            ],
            dtype=torch.float64,
            device=device,
        ),
        columns=(
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ),
    )


@withCUDA
def test_tabiclv2_default_recipe_matches_original_regression_n1_features(
    device: torch.device,
) -> None:
    table = _tabarena_california_housing_slice(device)
    recipe = TabICLv2.default_recipe()

    recipe.features.fit(table)
    output = recipe.transform_features(table)

    expected = _original_tabiclv2_regression_n1_features(table.numerical)
    assert torch.allclose(output.numerical, expected)
    assert output.columns == table.columns
