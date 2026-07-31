import pytest
import torch

from sdm import TableTensor
from sdm.processing import ClipQuantiles
from sdm.testing import withCUDA


@withCUDA
def test_clip_quantiles_bounds_and_transform(device: torch.device) -> None:
    inp = torch.tensor(
        [
            [0.0, 10.0],
            [1.0, 20.0],
            [2.0, 30.0],
            [100.0, 40.0],
        ],
        device=device,
    )

    processor = ClipQuantiles(q_low=0.25, q_high=0.75).fit(
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


@withCUDA
def test_clip_quantiles_default_uses_min_max_bounds(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[0.0, 5.0], [2.0, 7.0], [4.0, 9.0]],
        device=device,
    )

    processor = ClipQuantiles().fit(TableTensor.from_tensor(inp))

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
def test_clip_quantiles_constant_columns_are_exact(
    device: torch.device,
) -> None:
    inp = torch.full((4, 2), 3.0, device=device)

    processor = ClipQuantiles(q_low=0.02, q_high=0.98).fit(
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


def test_clip_quantiles_rejects_invalid_quantiles() -> None:
    with pytest.raises(ValueError, match="q_low <= q_high"):
        ClipQuantiles(q_low=0.75, q_high=0.25)


@withCUDA
def test_clip_quantiles_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [[[0.0], [2.0]], [[10.0], [20.0]]],
        device=device,
    )
    query = torch.tensor(
        [[[-1.0], [3.0]], [[5.0], [25.0]]],
        device=device,
    )

    processor = ClipQuantiles().fit(TableTensor.from_tensor(context))
    transformed = processor.transform(TableTensor.from_tensor(query)).numerical

    assert torch.equal(
        processor.lower_bound,
        torch.tensor([[[0.0]], [[10.0]]], device=device),
    )
    assert torch.equal(
        processor.upper_bound,
        torch.tensor([[[2.0]], [[20.0]]], device=device),
    )
    assert torch.equal(
        transformed,
        torch.tensor(
            [[[0.0], [2.0]], [[10.0], [20.0]]],
            device=device,
        ),
    )
