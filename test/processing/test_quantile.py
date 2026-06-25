import pytest
import torch
from sdm.processing import Quantile


def test_quantile_rejects_nonpositive_n_quantiles() -> None:
    with pytest.raises(ValueError, match="n_quantiles"):
        Quantile(n_quantiles=0)


def test_quantile_rejects_nonpositive_subsample() -> None:
    with pytest.raises(ValueError, match="subsample"):
        Quantile(subsample=0)


def test_quantile_uniform_fit_transform_and_inverse_round_trip() -> None:
    input = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=torch.float64,
    )

    processor = Quantile(n_quantiles=4, subsample=None).fit(input)
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0],
            [1.0, 1.0],
        ],
        dtype=torch.float64,
    )

    transformed = processor.transform(input)
    assert torch.allclose(processor.references, expected[:, 0])
    assert torch.allclose(transformed, expected)
    assert torch.allclose(processor.inverse_transform(transformed), input)


def test_quantile_repeated_values_map_to_midpoint() -> None:
    input = torch.tensor([[0.0], [1.0], [1.0], [2.0]])

    processor = Quantile(n_quantiles=4, subsample=None).fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        transformed.squeeze(1),
        torch.tensor([0.0, 0.5, 0.5, 1.0]),
    )
    assert torch.allclose(processor.inverse_transform(transformed), input)


def test_quantile_constant_columns_round_trip() -> None:
    input = torch.tensor([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]])

    processor = Quantile(n_quantiles=3, subsample=None).fit(input)
    transformed = processor.transform(input)

    assert torch.equal(transformed, torch.zeros_like(input))
    assert torch.equal(processor.inverse_transform(transformed), input)


def test_quantile_single_quantile_maps_to_single_reference() -> None:
    input = torch.tensor(
        [
            [2.0, 1.0],
            [3.0, 5.0],
        ],
        dtype=torch.float64,
    )

    processor = Quantile(n_quantiles=1, subsample=None).fit(input)
    transformed = processor.transform(input)

    assert torch.equal(transformed, torch.zeros_like(input))
    assert torch.equal(
        processor.inverse_transform(transformed),
        processor.quantiles[0].expand_as(input),
    )


def test_quantile_preserves_nan_positions() -> None:
    input = torch.tensor(
        [
            [0.0, 1.0],
            [torch.nan, 2.0],
            [2.0, torch.nan],
            [3.0, 4.0],
        ]
    )

    processor = Quantile(n_quantiles=4, subsample=None).fit(input)
    transformed = processor.transform(input)
    inverse = processor.inverse_transform(transformed)

    assert torch.equal(torch.isnan(transformed), torch.isnan(input))
    assert torch.equal(torch.isnan(inverse), torch.isnan(input))
    assert torch.isfinite(transformed[~torch.isnan(transformed)]).all()


def test_quantile_normal_distribution_is_finite_at_bounds() -> None:
    input = torch.tensor([[-2.0], [-1.0], [0.0], [4.0], [8.0]])

    processor = Quantile(
        n_quantiles=5,
        subsample=None,
        output_distribution="normal",
    ).fit(input)
    transformed = processor.transform(input)

    assert not hasattr(processor, "_distribution")
    assert transformed.isfinite().all()
    assert torch.allclose(
        processor.inverse_transform(transformed),
        input,
        atol=1e-5,
    )


def test_quantile_normal_distribution_preserves_nan_positions() -> None:
    input = torch.tensor([[0.0], [torch.nan], [2.0], [3.0]])

    processor = Quantile(
        n_quantiles=4,
        subsample=None,
        output_distribution="normal",
    ).fit(input)
    transformed = processor.transform(input)
    inverse = processor.inverse_transform(transformed)

    assert torch.equal(torch.isnan(transformed), torch.isnan(input))
    assert torch.equal(torch.isnan(inverse), torch.isnan(input))
    finite = ~torch.isnan(input)
    assert torch.allclose(inverse[finite], input[finite])


def test_quantile_subsample_is_reproducible_by_default() -> None:
    input = torch.arange(60.0).view(30, 2)

    first = Quantile(n_quantiles=4, subsample=12).fit(input)
    second = Quantile(n_quantiles=4, subsample=12).fit(input)

    assert torch.equal(first.quantiles, second.quantiles)
