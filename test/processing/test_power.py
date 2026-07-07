import torch
from sdm import TableTensor
from sdm.processing import Power
from sdm.testing import withCUDA


@withCUDA
def test_power_standardized_fit_transform_and_inverse_round_trip(
    device: torch.device,
) -> None:
    input = torch.tensor(
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

    processor = Power().fit(TableTensor.from_tensor(input))
    transformed = processor.transform(TableTensor.from_tensor(input)).numerical
    finite_var = transformed[:, :2].var(dim=0, correction=0)

    assert torch.allclose(
        transformed[:, :2].mean(dim=0),
        torch.zeros(2, dtype=input.dtype, device=device),
    )
    assert torch.allclose(
        finite_var,
        torch.ones(2, dtype=input.dtype, device=device),
    )
    assert torch.allclose(
        transformed[:, 2],
        torch.zeros(input.shape[0], dtype=input.dtype, device=device),
    )
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        input,
        atol=1e-8,
    )
    assert transformed.device == device


@withCUDA
def test_power_without_standardization_is_near_identity(
    device: torch.device,
) -> None:
    input = torch.tensor(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = Power(standardize=False).fit(TableTensor.from_tensor(input))
    transformed = processor.transform(TableTensor.from_tensor(input)).numerical

    assert torch.allclose(
        processor.lambdas,
        torch.ones(1, dtype=input.dtype, device=device),
        atol=1e-5,
    )
    assert torch.allclose(transformed, input, atol=1e-5)
    assert transformed.device == device
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        input,
    )


@withCUDA
def test_power_learns_skewed_lambda_regression(device: torch.device) -> None:
    input = torch.tensor(
        [[0.0], [1.0], [2.0], [4.0], [8.0], [16.0], [32.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = Power(standardize=False).fit(TableTensor.from_tensor(input))
    expected = torch.tensor(
        [-0.057856304067531325],
        dtype=input.dtype,
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
    input = torch.full((4, 2), 3.0, device=device)

    processor = Power().fit(TableTensor.from_tensor(input))
    transformed = processor.transform(TableTensor.from_tensor(input)).numerical

    assert torch.equal(processor.lambdas, torch.ones(2, device=device))
    assert torch.equal(transformed, torch.zeros_like(input))
    assert transformed.device == device
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        input,
    )


@withCUDA
def test_power_inverse_overflow_with_positive_lambda_clamps_to_max(
    device: torch.device,
) -> None:
    # This column fits a positive lambda, whose inverse-domain has no finite
    # upper bound, so ``upper_bound`` must be +inf (matching the lambda == 0
    # case). A non-finite model output must fall back to the fitted per-column
    # max.
    input = torch.tensor(
        [[0.0], [1.0], [4.0], [9.0], [16.0], [25.0], [36.0], [49.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = Power().fit(TableTensor.from_tensor(input))
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


@withCUDA
def test_power_is_nan_aware(device: torch.device) -> None:
    input = torch.tensor(
        [
            [1.0, -2.0],
            [torch.nan, -1.0],
            [4.0, torch.nan],
            [16.0, 2.0],
        ],
        dtype=torch.float64,
        device=device,
    )

    processor = Power().fit(TableTensor.from_tensor(input))
    transformed = processor.transform(TableTensor.from_tensor(input)).numerical
    inverse = processor.inverse_transform(
        TableTensor.from_tensor(transformed)
    ).numerical

    # Fitted overflow-guard ceiling is the finite per-column max, not NaN
    # poisoned by the missing entries.
    assert torch.equal(
        processor.max,
        torch.tensor([16.0, 2.0], dtype=torch.float64, device=device),
    )
    # NaN positions are preserved through transform and inverse; finite
    # entries stay finite.
    assert torch.equal(torch.isnan(transformed), torch.isnan(input))
    assert torch.equal(torch.isnan(inverse), torch.isnan(input))
    assert torch.isfinite(transformed[~torch.isnan(transformed)]).all()
    assert transformed.device == device
    assert inverse.device == device
