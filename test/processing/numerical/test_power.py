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


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_power_transform_logarithmic_limits_preserve_expanded_input(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    values = torch.tensor(
        [-3.0, -1.0, -0.125, 0.0, 0.125, 1.0, 3.0],
        device=device,
        dtype=dtype,
    )
    inp = values.view(1, -1, 1).expand(2, -1, 4)
    original = inp.clone()
    table = TableTensor.from_tensor(inp)
    processor = PowerTransform(standardize=False).fit(table)
    processor.lambdas.copy_(
        inp.new_tensor([0, torch.finfo(dtype).eps / 2, 1, 2])
    )
    state = {name: value.clone() for name, value in processor.named_buffers()}

    zero_lambda = torch.where(
        values >= 0,
        values.log1p(),
        values - values.square() / 2,
    )
    two_lambda = torch.where(
        values >= 0,
        values + values.square() / 2,
        -(-values).log1p(),
    )
    expected = torch.stack(
        [zero_lambda, zero_lambda, values, two_lambda], dim=-1
    ).expand_as(inp)

    transformed = processor.transform(table)
    before_inverse = transformed.numerical.clone()
    inverse = processor.inverse_transform(transformed)
    repeated = processor.transform(table)

    torch.testing.assert_close(transformed.numerical, expected)
    torch.testing.assert_close(inverse.numerical, inp)
    torch.testing.assert_close(repeated.numerical, expected)
    torch.testing.assert_close(transformed.numerical, before_inverse)
    torch.testing.assert_close(inp, original)
    for name, value in processor.named_buffers():
        torch.testing.assert_close(value, state[name])


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_power_transform_inverse_preserves_domain_and_overflow_behavior(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    context = torch.tensor(
        [[-1.0], [0.0], [1.0]], device=device, dtype=dtype
    ).expand(-1, 5)
    processor = PowerTransform(standardize=False).fit(
        TableTensor.from_tensor(context)
    )
    processor.lambdas.copy_(context.new_tensor([-1, 0, 1, 2, 3]))
    processor.upper_bound.copy_(
        context.new_tensor([1, torch.inf, torch.inf, torch.inf, torch.inf])
    )
    query = context.new_tensor(
        [
            [1, torch.inf, torch.inf, -torch.inf, -1],
            [2, -torch.inf, -torch.inf, torch.inf, -2],
        ]
    )
    original = query.clone()
    expected = context.new_tensor(
        [
            [1 / torch.finfo(dtype).eps - 1, 1, 1, -torch.inf, -torch.inf],
            [torch.nan, -torch.inf, -torch.inf, 1, torch.nan],
        ]
    )

    actual = processor.inverse_transform(
        TableTensor.from_tensor(query)
    ).numerical

    torch.testing.assert_close(actual, expected, equal_nan=True)
    torch.testing.assert_close(query, original)
