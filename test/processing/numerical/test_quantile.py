import pytest
import torch
from sdm import TableTensor
from sdm.processing import QuantileTransform
from sdm.testing import onlyCUDA, withCUDA


def test_quantile_transform_rejects_nonpositive_n_quantiles() -> None:
    with pytest.raises(ValueError, match="n_quantiles"):
        QuantileTransform(n_quantiles=0)


def test_quantile_transform_rejects_nonpositive_subsample() -> None:
    with pytest.raises(ValueError, match="subsample"):
        QuantileTransform(subsample=0)


@withCUDA
def test_quantile_transform_uniform_fit_transform_and_inverse_round_trip(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = QuantileTransform(n_quantiles=4, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    assert torch.allclose(processor.references, expected[:, 0])
    assert torch.allclose(transformed, expected)
    assert transformed.device == device
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_quantile_transform_wide_inverse_round_trip(
    device: torch.device,
) -> None:
    inp = torch.linspace(
        -3, 3, steps=64 * 40, dtype=torch.float64, device=device
    ).view(64, 40)

    processor = QuantileTransform(n_quantiles=64, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    inverse = processor.inverse_transform(
        TableTensor.from_tensor(transformed)
    ).numerical

    assert torch.allclose(inverse, inp, atol=1e-8)


@withCUDA
def test_quantile_transform_repeated_values_map_to_midpoint(
    device: torch.device,
) -> None:
    inp = torch.tensor([[0.0], [1.0], [1.0], [2.0]], device=device)

    processor = QuantileTransform(n_quantiles=4, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        transformed.squeeze(1),
        torch.tensor([0.0, 0.5, 0.5, 1.0], device=device),
    )
    assert transformed.device == device
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_quantile_transform_constant_columns_round_trip(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
        device=device,
    )

    processor = QuantileTransform(n_quantiles=3, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(transformed, torch.zeros_like(inp))
    assert transformed.device == device
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_quantile_transform_single_quantile_maps_to_single_reference(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [
            [2.0, 1.0],
            [3.0, 5.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = QuantileTransform(n_quantiles=1, subsample=None).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(transformed, torch.zeros_like(inp))
    assert transformed.device == device
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        processor.quantiles[0].expand_as(inp),
    )


@withCUDA
def test_quantile_transform_normal_distribution_is_finite_at_bounds(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[-2.0], [-1.0], [0.0], [4.0], [8.0]],
        device=device,
    )

    processor = QuantileTransform(
        n_quantiles=5,
        subsample=None,
        output_distribution="normal",
    ).fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert transformed.isfinite().all()
    assert transformed.device == device
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
        atol=1e-5,
    )


@onlyCUDA
def test_quantile_transform_rejects_mismatched_generator_device() -> None:
    table = TableTensor.from_tensor(torch.rand(8, 2, device="cuda"))

    with pytest.raises(RuntimeError, match="device type for generator"):
        QuantileTransform(subsample=4).fit(
            table,
            generator=torch.Generator(),
        )


def test_quantile_transform_subsample_is_reproducible_with_generator() -> None:
    # Distinct values: any other row subset changes the quantiles.
    inp = torch.arange(200.0).view(100, 2)

    first = QuantileTransform(n_quantiles=6, subsample=32).fit(
        TableTensor.from_tensor(inp),
        generator=torch.Generator().manual_seed(0),
    )
    second = QuantileTransform(n_quantiles=6, subsample=32).fit(
        TableTensor.from_tensor(inp),
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.equal(first.quantiles, second.quantiles)
