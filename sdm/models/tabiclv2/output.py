"""Regression output decoding for TabICLv2."""

from typing import Protocol

from torch import Tensor

from sdm import TableTensor

_NUM_REGRESSION_QUANTILES = 999


class _TargetInverseTransform(Protocol):
    def inverse_transform(self, table: TableTensor) -> TableTensor: ...


def reduce_regression_quantiles(quantiles: Tensor) -> Tensor:
    r"""Reduce TabICLv2 quantiles into point predictions.

    Args:
        quantiles: Regression output with shape ``[..., R, 999]``.

    Returns:
        Point predictions with shape ``[..., R]`` in the current target space.

    Raises:
        ValueError: If ``quantiles`` is not floating point or has an invalid
            final dimension.
    """
    if not quantiles.is_floating_point():
        raise ValueError("Expected regression quantiles to be floating point.")
    if quantiles.dim() < 2 or quantiles.size(-1) != _NUM_REGRESSION_QUANTILES:
        raise ValueError(
            "Expected regression quantiles with shape [..., R, 999] "
            f"(got {tuple(quantiles.size())})."
        )
    return quantiles.sort(dim=-1).values.mean(dim=-1)


def decode_regression_quantiles(
    quantiles: Tensor,
    target_transform: _TargetInverseTransform,
) -> Tensor:
    r"""Decode TabICLv2 regression quantiles into point predictions.

    Predicted quantiles are sorted independently for each row to repair
    quantile crossing, reduced to their mean, and then mapped from model target
    space through the fitted SDM target inverse transform.

    Args:
        quantiles: Raw regression output with shape ``[..., R, 999]``.
        target_transform: Fitted SDM processor that inverts the target
            transform.

    Returns:
        A point-prediction tensor with shape ``[..., R]`` on the input device
        and with the input dtype.

    Raises:
        ValueError: If ``quantiles`` is not floating point or has an invalid
            final dimension.
    """
    point_predictions = reduce_regression_quantiles(quantiles).unsqueeze(-1)
    table = TableTensor.from_tensor(point_predictions, columns=("target",))
    predictions = target_transform.inverse_transform(table).numerical
    return predictions.squeeze(-1).to(
        device=quantiles.device,
        dtype=quantiles.dtype,
    )
