import pytest
import torch

from sdm import TableTensor
from sdm.processing import ClipSigma
from sdm.testing import withCUDA


@withCUDA
def test_clip_sigma_two_stage_outlier_behavior(
    device: torch.device,
) -> None:
    dtype = torch.float64
    inp = torch.tensor(
        [[0.0], [1.0], [2.0], [100.0]],
        dtype=dtype,
        device=device,
    )

    processor = ClipSigma(threshold=1.0).fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        processor.lower_bound,
        torch.tensor([[0.0]], dtype=dtype, device=device),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([[2.0]], dtype=dtype, device=device),
    )
    assert transformed[-1, 0] < inp[-1, 0]
    assert torch.allclose(
        transformed[-1, 0],
        torch.log1p(torch.tensor(100.0, dtype=dtype, device=device)) + 2.0,
    )


def test_clip_sigma_matches_tabicl_reference_values() -> None:
    dtype = torch.float64
    inp = torch.tensor(
        [
            [-8.0, 1.0],
            [-1.0, 2.0],
            [0.0, 3.0],
            [1.0, 4.0],
            [20.0, 5.0],
        ],
        dtype=dtype,
    )

    processor = ClipSigma(threshold=1.5).fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        processor.lower_bound,
        torch.tensor([[-8.123724356958, 0.628291754874]], dtype=dtype),
    )
    assert torch.allclose(
        processor.upper_bound,
        torch.tensor([[4.123724356958, 5.371708245126]], dtype=dtype),
    )
    assert torch.allclose(
        transformed,
        torch.tensor(
            [
                [-8.0, 1.0],
                [-1.0, 2.0],
                [0.0, 3.0],
                [1.0, 4.0],
                [7.168246794681, 5.0],
            ],
            dtype=dtype,
        ),
    )


def test_clip_sigma_rejects_nonpositive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        ClipSigma(threshold=0.0)


@withCUDA
def test_clip_sigma_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [100.0]],
            [[10.0], [12.0], [14.0], [200.0]],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = ClipSigma(threshold=1.0).fit(TableTensor.from_tensor(context))
    transformed = processor.transform(
        TableTensor.from_tensor(
            torch.tensor(
                [[[-100.0], [100.0]], [[-200.0], [200.0]]],
                dtype=torch.float64,
                device=device,
            )
        )
    ).numerical

    torch.testing.assert_close(
        transformed,
        torch.tensor(
            [
                [[-4.61512051684126], [6.61512051684126]],
                [[4.696695091940924], [19.303304908059076]],
            ],
            dtype=torch.float64,
            device=device,
        ),
    )


@withCUDA
@pytest.mark.parametrize("threshold", [0.1, 1.5])
def test_clip_sigma_matches_masked_fit_on_noncontiguous_data(
    device: torch.device,
    threshold: float,
) -> None:
    data = (
        torch.tensor(
            [[-100.0, 2.0], [-1.0, 2.0], [1.0, 2.0], [100.0, 2.0]],
            dtype=torch.float64,
            device=device,
        )
        .T.contiguous()
        .T
    )
    original = data.clone()
    mean = data.mean(dim=-2, keepdim=True)
    std = data.std(dim=-2, keepdim=True).clamp_min(1e-6)
    keep = (data >= mean - threshold * std) & (data <= mean + threshold * std)
    count = keep.sum(dim=-2, keepdim=True)
    clean_mean = torch.where(keep, data, 0.0).sum(
        dim=-2, keepdim=True
    ) / count.clamp_min(1)
    centered = (data - clean_mean).masked_fill(~keep, 0.0)
    clean_std = (
        centered.square().sum(dim=-2, keepdim=True)
        / (count - (count > 1).to(count.dtype)).clamp_min(1)
    ).sqrt()
    mean = torch.where(count > 0, clean_mean, mean)
    std = torch.where(count > 0, clean_std, std).clamp_min(1e-6)

    processor = ClipSigma(threshold=threshold).fit(
        TableTensor.from_tensor(data)
    )

    torch.testing.assert_close(processor.lower_bound, mean - threshold * std)
    torch.testing.assert_close(processor.upper_bound, mean + threshold * std)
    torch.testing.assert_close(data, original)
