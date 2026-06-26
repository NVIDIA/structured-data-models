import torch
from torch import Tensor

from sdm.processing import Power, Quantile
from sdm.processing.power import (
    _yeojohnson_inverse_transform_batch,
    _yeojohnson_transform_batch,
)


def _yeojohnson_transform_reference(input: Tensor, lambdas: Tensor) -> Tensor:
    transformed = input.clone()
    eps = torch.finfo(input.dtype).eps

    for i, lmbda in enumerate(lambdas):
        col = input[:, i]
        lam = float(lmbda)
        positive = col >= 0
        out = torch.zeros_like(col)

        if abs(lam) < eps:
            out[positive] = col[positive].log1p()
        else:
            out[positive] = (lam * col[positive].log1p()).expm1() / lam

        if abs(lam - 2) > eps:
            out[~positive] = -(
                (2 - lam) * (-col[~positive]).log1p()
            ).expm1() / (2 - lam)
        else:
            out[~positive] = -(-col[~positive]).log1p()

        transformed[:, i] = out

    return transformed


def _yeojohnson_inverse_transform_reference(
    input: Tensor,
    lambdas: Tensor,
) -> Tensor:
    inverse = input.clone()
    eps = torch.finfo(input.dtype).eps

    for i, lmbda in enumerate(lambdas):
        col = input[:, i]
        lam = float(lmbda)
        positive = col >= 0
        out = torch.zeros_like(col)

        if abs(lam) < eps:
            out[positive] = col[positive].expm1()
        else:
            out[positive] = (
                (col[positive] * lam + 1).log() / lam
            ).expm1()

        if abs(lam - 2) > eps:
            out[~positive] = -(
                (-(2 - lam) * col[~positive] + 1).log() / (2 - lam)
            ).expm1()
        else:
            out[~positive] = -(-col[~positive]).expm1()

        inverse[:, i] = out

    return inverse


def test_power_vectorized_transform_is_bit_identical_to_column_loop() -> None:
    input = torch.tensor(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 4.0],
            [0.0, 4.0, 9.0],
            [1.0, 8.0, 16.0],
            [2.0, 16.0, 25.0],
        ],
        dtype=torch.float64,
    )
    processor = Power(standardize=False).fit(input)

    assert torch.equal(
        _yeojohnson_transform_batch(input, processor.lambdas),
        _yeojohnson_transform_reference(input, processor.lambdas),
    )
    assert torch.equal(
        _yeojohnson_inverse_transform_batch(input, processor.lambdas),
        _yeojohnson_inverse_transform_reference(input, processor.lambdas),
    )


def test_power_forward_torch_compile_smoke() -> None:
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
    compiled = torch.compile(processor.forward, backend="eager")

    expected = processor.forward(input)
    actual = compiled(input)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_quantile_forward_torch_compile_smoke() -> None:
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
    compiled = torch.compile(processor.forward, backend="eager")

    expected = processor.forward(input)
    actual = compiled(input)

    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)
