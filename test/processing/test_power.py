import torch
from sdm import TableTensor
from sdm.processing import Power
from sdm.testing import withCUDA


@withCUDA
def test_power_standardized_fit_transform_and_inverse_round_trip(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [
            [-2.0, 1.0, 4.0],
            [-1.0, 2.0, 4.0],
            [0.0, 4.0, 4.0],
            [1.0, 8.0, 4.0],
            [2.0, 16.0, 4.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = Power().fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    finite_var = transformed[:, :2].var(dim=0, correction=0)

    assert torch.allclose(
        transformed[:, :2].mean(dim=0),
        torch.zeros(2, dtype=inp.dtype, device=device),
    )
    assert torch.allclose(
        finite_var,
        torch.ones(2, dtype=inp.dtype, device=device),
    )
    assert torch.allclose(
        transformed[:, 2],
        torch.zeros(inp.shape[0], dtype=inp.dtype, device=device),
    )
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
        atol=1e-8,
    )
    assert transformed.device == device


@withCUDA
def test_power_wide_inverse_round_trip(device: torch.device) -> None:
    inp = torch.linspace(-3, 3, steps=32 * 40, device=device).view(32, 40)

    processor = Power().fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    inverse = processor.inverse_transform(
        TableTensor.from_tensor(transformed)
    ).numerical

    assert torch.allclose(inverse, inp, atol=1e-4, rtol=1e-4)


@withCUDA
def test_power_without_standardization_is_near_identity(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = Power(standardize=False).fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        processor.lambdas,
        torch.ones(1, dtype=inp.dtype, device=device),
        atol=1e-5,
    )
    assert torch.allclose(transformed, inp, atol=1e-5)
    assert transformed.device == device
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_power_learns_skewed_lambda_regression(device: torch.device) -> None:
    inp = torch.tensor(
        [[0.0], [1.0], [2.0], [4.0], [8.0], [16.0], [32.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = Power(standardize=False).fit(TableTensor.from_tensor(inp))
    expected = torch.tensor(
        [-0.057856304067531325],
        dtype=inp.dtype,
        device=device,
    )

    assert torch.allclose(processor.lambdas, expected, atol=1e-4)
    assert not torch.allclose(
        processor.lambdas,
        torch.ones_like(processor.lambdas),
    )


@withCUDA
def test_power_constant_columns_use_identity_lambda(
    device: torch.device,
) -> None:
    inp = torch.full((4, 2), 3.0, device=device)

    processor = Power().fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(processor.lambdas, torch.ones(2, device=device))
    assert torch.equal(transformed, torch.zeros_like(inp))
    assert transformed.device == device
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_power_inverse_overflow_with_positive_lambda_clamps_to_max(
    device: torch.device,
) -> None:
    # This column fits a positive lambda, whose inverse-domain has no finite
    # upper bound, so ``upper_bound`` must be +inf (matching the lambda == 0
    # case). A non-finite model output must fall back to the fitted per-column
    # max.
    inp = torch.tensor(
        [[0.0], [1.0], [4.0], [9.0], [16.0], [25.0], [36.0], [49.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = Power().fit(TableTensor.from_tensor(inp))
    assert (processor.lambdas > 0).all()
    assert torch.isinf(processor.upper_bound).all()

    extreme = torch.tensor(
        [[float("inf")]], dtype=torch.float64, device=device
    )
    inverse = processor.inverse_transform(
        TableTensor.from_tensor(extreme)
    ).numerical

    assert torch.isfinite(inverse).all()
    assert torch.equal(inverse, processor.max.reshape_as(inverse))
    assert inverse.device == device
