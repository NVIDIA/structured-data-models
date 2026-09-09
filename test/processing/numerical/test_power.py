import pytest
import torch

from sdm import TableTensor
from sdm.processing import PowerTransform
from sdm.processing.numerical import power as power_module
from sdm.testing import onlyCUDA, onlyFullTest, withCUDA


@onlyFullTest
@onlyCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_power_transform_accepts_caller_compiled_optimizer(
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inp = torch.linspace(
        -3,
        3,
        steps=2 * 17 * 3,
        device="cuda",
        dtype=dtype,
    ).view(2, 17, 3)
    table = TableTensor.from_tensor(inp)

    with torch.inference_mode():
        eager = PowerTransform().fit_transform(table).numerical
        compiled_optimizer = torch.compile(
            power_module._optimize_lambdas,
            fullgraph=True,
        )
        monkeypatch.setattr(
            power_module,
            "_optimize_lambdas",
            compiled_optimizer,
        )
        compiled = PowerTransform().fit_transform(table).numerical

    assert compiled.shape == inp.shape
    assert torch.isfinite(compiled).all()
    torch.testing.assert_close(compiled, eager, atol=1e-3, rtol=1e-3)


@withCUDA
def test_power_transform_standardized_fit_transform_and_inverse_round_trip(
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

    processor = PowerTransform().fit(TableTensor.from_tensor(inp))
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


@withCUDA
def test_power_transform_wide_inverse_round_trip(device: torch.device) -> None:
    inp = torch.linspace(-3, 3, steps=32 * 40, device=device).view(32, 40)

    processor = PowerTransform().fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    inverse = processor.inverse_transform(
        TableTensor.from_tensor(transformed)
    ).numerical

    assert torch.allclose(inverse, inp, atol=1e-4, rtol=1e-4)


@withCUDA
def test_power_transform_without_standardization_is_near_identity(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[-2.0], [-1.0], [0.0], [1.0], [2.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = PowerTransform(standardize=False).fit(
        TableTensor.from_tensor(inp)
    )
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.allclose(
        processor.lambdas,
        torch.ones((1, 1), dtype=inp.dtype, device=device),
        atol=1e-5,
    )
    assert torch.allclose(transformed, inp, atol=1e-5)
    assert torch.allclose(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_power_transform_learns_skewed_lambda(
    device: torch.device,
) -> None:
    inp = torch.tensor(
        [[0.0], [1.0], [2.0], [4.0], [8.0], [16.0], [32.0]],
        dtype=torch.float64,
        device=device,
    )

    processor = PowerTransform(standardize=False).fit(
        TableTensor.from_tensor(inp)
    )
    expected = torch.tensor(
        [[-0.057856304067531325]],
        dtype=inp.dtype,
        device=device,
    )

    assert torch.allclose(processor.lambdas, expected, atol=1e-4)
    assert not torch.allclose(
        processor.lambdas,
        torch.ones_like(processor.lambdas),
    )


@withCUDA
def test_power_transform_constant_columns_use_identity_lambda(
    device: torch.device,
) -> None:
    inp = torch.full((4, 2), 3.0, device=device)

    processor = PowerTransform().fit(TableTensor.from_tensor(inp))
    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical

    assert torch.equal(processor.lambdas, torch.ones((1, 2), device=device))
    assert torch.equal(transformed, torch.zeros_like(inp))
    assert torch.equal(
        processor.inverse_transform(
            TableTensor.from_tensor(transformed)
        ).numerical,
        inp,
    )


@withCUDA
def test_power_transform_preserves_nan(device: torch.device) -> None:
    inp = torch.tensor(
        [[1.0, float("nan")], [3.0, 10.0], [5.0, 14.0]],
        device=device,
    )
    processor = PowerTransform().fit(TableTensor.from_tensor(inp))

    transformed = processor.transform(TableTensor.from_tensor(inp)).numerical
    restored = processor.inverse_transform(
        TableTensor.from_tensor(transformed)
    ).numerical

    assert torch.equal(transformed.isnan(), inp.isnan())
    torch.testing.assert_close(restored, inp, equal_nan=True)


@withCUDA
def test_power_transform_inverse_overflow_with_positive_lambda_clamps_to_max(
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

    processor = PowerTransform().fit(TableTensor.from_tensor(inp))
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


@withCUDA
def test_power_transform_fits_leading_batches_independently(
    device: torch.device,
) -> None:
    context = torch.tensor(
        [
            [[-2.0], [-1.0], [0.0], [1.0], [4.0]],
            [[0.0], [1.0], [2.0], [8.0], [32.0]],
        ],
        dtype=torch.float64,
        device=device,
    )
    query = torch.tensor([[[-0.5], [2.0]], [[1.5], [16.0]]], device=device)

    processor = PowerTransform().fit(TableTensor.from_tensor(context))
    actual = processor.transform(TableTensor.from_tensor(query)).numerical
    expected = []
    for batch in range(context.size(0)):
        independent = PowerTransform().fit(
            TableTensor.from_tensor(context[batch])
        )
        expected.append(
            independent.transform(
                TableTensor.from_tensor(query[batch])
            ).numerical
        )

    torch.testing.assert_close(
        actual,
        torch.stack(expected),
        rtol=2e-5,
        atol=2e-5,
    )
