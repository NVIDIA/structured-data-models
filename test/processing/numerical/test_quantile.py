import pytest
import torch

from sdm import EnsembleTable, TableTensor
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


@withCUDA
def test_quantile_transform_deterministic_fit_remains_shared(
    device: torch.device,
) -> None:
    table = TableTensor.from_tensor(
        torch.arange(64.0, device=device).view(32, 2)
    )
    output = QuantileTransform(
        n_quantiles=8,
        subsample=None,
    ).fit_transform_ensemble(EnsembleTable(table, num_members=8))

    assert (
        sum(packed.size(0) for packed in output.iter_packed_representations())
        == 1
    )
    for member_id in range(output.num_members):
        assert output.representation(member_id).equal(output.representation(0))


@withCUDA
def test_quantile_transform_subsample_matches_independent_processors(
    device: torch.device,
) -> None:
    context = TableTensor.from_tensor(
        torch.arange(256.0, device=device).view(128, 2)
    )
    query = TableTensor.from_tensor(
        torch.arange(32.0, device=device).view(16, 2) + 0.5
    )
    processor = QuantileTransform(n_quantiles=8, subsample=32)

    context_output = processor.fit_transform_ensemble(
        EnsembleTable(context, num_members=8),
        generator=torch.Generator(device=device).manual_seed(7),
    )
    query_output = processor.transform_ensemble(
        EnsembleTable(query, num_members=8)
    )
    restored = processor.inverse_transform_ensemble(context_output)

    generator = torch.Generator(device=device).manual_seed(7)
    references = [
        QuantileTransform(n_quantiles=8, subsample=32) for _ in range(8)
    ]
    for member_id, reference in enumerate(references):
        expected_context = reference.fit_transform(
            context,
            generator=generator,
        )
        expected_query = reference.transform(query)
        expected_restored = reference.inverse_transform(expected_context)
        assert context_output.representation(member_id).equal(expected_context)
        assert query_output.representation(member_id).equal(expected_query)
        assert restored.representation(member_id).equal(expected_restored)

    with pytest.raises(RuntimeError, match="same number"):
        processor.transform_ensemble(EnsembleTable(query, num_members=7))
    with pytest.raises(RuntimeError, match="fitted for an ensemble"):
        processor.transform(query)
    with pytest.raises(RuntimeError, match="fitted for an ensemble"):
        processor.inverse_transform(context_output.representation(0))


def test_quantile_transform_refit_clears_ensemble_state() -> None:
    table = TableTensor.from_tensor(torch.arange(64.0).view(32, 2))
    processor = QuantileTransform(n_quantiles=8, subsample=None)

    processor.fit_transform_ensemble(EnsembleTable(table, num_members=4))
    transformed = processor.fit_transform(table)
    expected = QuantileTransform(
        n_quantiles=8,
        subsample=None,
    ).fit_transform(table)

    assert transformed.equal(expected)


def test_quantile_transform_ensemble_inverse_requires_fit() -> None:
    table = TableTensor.from_tensor(torch.arange(64.0).view(32, 2))

    with pytest.raises(RuntimeError, match="not fitted"):
        QuantileTransform(
            n_quantiles=8,
            subsample=None,
        ).inverse_transform_ensemble(EnsembleTable(table, num_members=4))


def test_quantile_transform_ensemble_requires_ensemble_fit() -> None:
    table = TableTensor.from_tensor(torch.arange(64.0).view(32, 2))
    processor = QuantileTransform(
        n_quantiles=8,
        subsample=None,
    ).fit(table)
    ensemble = EnsembleTable(table, num_members=4)

    with pytest.raises(RuntimeError, match="fitted for a single table"):
        processor.transform_ensemble(ensemble)
    with pytest.raises(RuntimeError, match="fitted for a single table"):
        processor.inverse_transform_ensemble(ensemble)
