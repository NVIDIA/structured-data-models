import torch
from sdm.processing import Power


def test_power_standardized_fit_transform_and_inverse_round_trip() -> None:
    input = torch.tensor(
        [
            [-2.0, 1.0, 4.0],
            [-1.0, 2.0, 4.0],
            [0.0, 4.0, 4.0],
            [1.0, 8.0, 4.0],
            [2.0, 16.0, 4.0],
        ],
        dtype=torch.float64,
    )

    processor = Power().fit(input)
    transformed = processor.transform(input)
    finite_var = transformed[:, :2].var(dim=0, correction=0)

    assert torch.allclose(
        transformed[:, :2].mean(dim=0),
        torch.zeros(2, dtype=input.dtype),
    )
    assert torch.allclose(finite_var, torch.ones(2, dtype=input.dtype))
    assert torch.allclose(
        transformed[:, 2],
        torch.zeros(input.shape[0], dtype=input.dtype),
    )
    assert torch.allclose(
        processor.inverse_transform(transformed),
        input,
        atol=1e-8,
    )


def test_power_batched_inference_compile_round_trips_mixed_columns() -> None:
    fit_input = torch.tensor(
        [
            [-5.0, 0.0, 3.0, torch.nan],
            [-2.0, 1.0, 3.0, -10.0],
            [-1.0, 2.0, 3.0, -5.0],
            [0.0, 4.0, 3.0, -2.0],
            [1.0, 8.0, 3.0, 0.0],
            [2.0, 16.0, 3.0, 1.0],
            [5.0, 32.0, 3.0, 2.0],
            [10.0, 64.0, 3.0, 4.0],
        ],
        dtype=torch.float64,
    )
    inference = torch.tensor(
        [
            [-3.0, 3.0, 3.0, -7.0],
            [0.0, 10.0, 3.0, torch.nan],
            [4.0, 40.0, 3.0, 3.0],
        ],
        dtype=torch.float64,
    )

    processor = Power().fit(fit_input)
    transformed = processor.transform(inference)
    compiled = torch.compile(
        processor.transform,
        backend="eager",
        fullgraph=True,
    )
    compiled_transformed = compiled(inference)
    inverse = processor.inverse_transform(transformed)

    assert torch.allclose(compiled_transformed, transformed, equal_nan=True)
    assert torch.equal(torch.isnan(transformed), torch.isnan(inference))
    finite = inference.isfinite()
    assert torch.allclose(inverse[finite], inference[finite], atol=1e-8)
    assert torch.equal(
        transformed[:, 2],
        torch.zeros(3, dtype=inference.dtype),
    )
    assert torch.equal(inverse[:, 2], inference[:, 2])


def test_power_without_standardization_is_near_identity() -> None:
    input = torch.tensor(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]],
        dtype=torch.float64,
    )

    processor = Power(standardize=False).fit(input)
    transformed = processor.transform(input)

    assert torch.allclose(
        processor.lambdas,
        torch.ones(1, dtype=input.dtype),
        atol=1e-5,
    )
    assert torch.allclose(transformed, input, atol=1e-5)
    assert torch.allclose(processor.inverse_transform(transformed), input)


def test_power_learns_skewed_lambda_regression() -> None:
    input = torch.tensor(
        [[0.0], [1.0], [2.0], [4.0], [8.0], [16.0], [32.0]],
        dtype=torch.float64,
    )

    processor = Power(standardize=False).fit(input)
    expected = torch.tensor([-0.057856304067531325], dtype=input.dtype)

    assert torch.allclose(processor.lambdas, expected, atol=1e-4)
    assert not torch.allclose(
        processor.lambdas,
        torch.ones_like(processor.lambdas),
    )


def test_power_constant_columns_use_identity_lambda() -> None:
    input = torch.full((4, 2), 3.0)

    processor = Power().fit(input)
    transformed = processor.transform(input)

    assert torch.equal(processor.lambdas, torch.ones(2))
    assert torch.equal(transformed, torch.zeros_like(input))
    assert torch.equal(processor.inverse_transform(transformed), input)


def test_power_is_nan_aware() -> None:
    input = torch.tensor(
        [
            [1.0, -2.0],
            [torch.nan, -1.0],
            [4.0, torch.nan],
            [16.0, 2.0],
        ],
        dtype=torch.float64,
    )

    processor = Power().fit(input)
    transformed = processor.transform(input)
    inverse = processor.inverse_transform(transformed)

    # Fitted overflow-guard ceiling is the finite per-column max, not NaN
    # poisoned by the missing entries.
    assert torch.equal(
        processor.max, torch.tensor([16.0, 2.0], dtype=torch.float64)
    )
    # NaN positions are preserved through transform and inverse; finite
    # entries stay finite.
    assert torch.equal(torch.isnan(transformed), torch.isnan(input))
    assert torch.equal(torch.isnan(inverse), torch.isnan(input))
    assert torch.isfinite(transformed[~torch.isnan(transformed)]).all()
