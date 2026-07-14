import pytest
import torch
from sdm import TableTensor
from sdm.processing import QuantileClip
from sdm.testing import withCUDA


@withCUDA
def test_quantile_clip_bounds_and_transform(device: torch.device) -> None:
    inp = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [100.0, 40.0],
        ],
        device=device,
    )

    processor = QuantileClip(q_low=0.25, q_high=0.75).fit(
        TableTensor.from_tensor(inp)
    )
    expected_bounds = torch.quantile(
        inp,
        torch.tensor([0.25, 0.75], device=device),
        dim=0,
    )
    expected = inp.clamp(min=expected_bounds[0], max=expected_bounds[1])

    assert torch.allclose(processor.lower_bound, expected_bounds[0])
    assert torch.allclose(processor.upper_bound, expected_bounds[1])
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    assert torch.equal(transformed, expected)
    assert transformed.device == device
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(expected)
        ).numerical,
        expected,
    )


@withCUDA
def test_quantile_clip_default_uses_min_max_bounds(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[0.0, 5.0], [2.0, 7.0], [4.0, 9.0]],
        device=device,
    )

    processor = QuantileClip().fit(TableTensor.from_tensor(inp))

    assert torch.equal(
        processor.lower_bound,
        torch.tensor([0.0, 5.0], device=device),
    )
    assert torch.equal(
        processor.upper_bound,
        torch.tensor([4.0, 9.0], device=device),
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    assert torch.equal(transformed, inp)
    assert transformed.device == device


@withCUDA
def test_quantile_clip_constant_columns_are_exact(
    device: torch.device,
) -> None:
    inp = torch.full((4, 2), 3.0, device=device)

    processor = QuantileClip(q_low=0.02, q_high=0.98).fit(
        TableTensor.from_tensor(inp)
    )

    assert torch.equal(
        processor.lower_bound, torch.full((2,), 3.0, device=device)
    )
    assert torch.equal(
        processor.upper_bound, torch.full((2,), 3.0, device=device)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    assert torch.equal(transformed, inp)
    assert transformed.device == device


@withCUDA
def test_quantile_clip_nan_columns_follow_torch_quantile(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [
            [0.0, 1.0],
            [torch.nan, 2.0],
            [4.0, 3.0],
        ],
        device=device,
    )

    processor = QuantileClip(q_low=0.0, q_high=1.0).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.isnan(processor.lower_bound[0])
    assert torch.isnan(processor.upper_bound[0])
    assert torch.isnan(transformed[:, 0]).all()
    assert torch.equal(transformed[:, 1], inp[:, 1])
    assert transformed.device == device


def test_quantile_clip_rejects_invalid_quantiles() -> None:
    with pytest.raises(ValueError, match="q_low <= q_high"):
        QuantileClip(q_low=0.75, q_high=0.25)
