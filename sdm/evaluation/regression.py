# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor

from sdm import Stype, TableTensor


def to_quantiles(pred: TableTensor) -> tuple[Tensor, Tensor]:
    r"""Convert quantile predictions to a per-target quantile tensor.

    Numerical columns must be named ``"q<digits>"`` for a single target or
    ``"<target>__q<digits>"`` for multiple targets, where the digits encode the
    quantile level as a decimal fraction, *e.g.*, ``"q500"`` and ``"q50"`` both
    denote the median.
    Columns are expected to be grouped by target, with the same quantile levels
    in the same order for every target.

    Args:
        pred: Prediction table whose numerical columns contain quantiles.

    Returns:
        The quantile predictions with shape ``[..., V, Q]`` for ``V`` targets
        and ``Q`` quantile levels, and the quantile levels with shape ``[Q]``.

    >>> import torch
    >>> import sdm
    >>> from sdm.evaluation import to_quantiles
    >>> pred = sdm.TableTensor.from_tensor(
    ...     torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    ...     columns=["q100", "q400", "q700", "q900"],
    ... )
    >>> values, levels = to_quantiles(pred)
    >>> values
    tensor([[[1., 2., 3., 4.]]])
    >>> levels
    tensor([0.1000, 0.4000, 0.7000, 0.9000])
    >>> pred = sdm.TableTensor.from_tensor(
    ...     torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
    ...     columns=["a__q10", "a__q90", "b__q10", "b__q90"],
    ... )
    >>> values, levels = to_quantiles(pred)
    >>> values
    tensor([[[1., 2.],
             [3., 4.]]])
    >>> levels
    tensor([0.1000, 0.9000])
    """
    columns = pred.columns[Stype.numerical]
    prefixes, _, digits = zip(*(column.rpartition("q") for column in columns))
    targets = dict.fromkeys(prefixes)
    digits = digits[: len(columns) // len(targets)]
    if columns != tuple(f"{t}q{d}" for t in targets for d in digits):
        raise ValueError(
            "Expected quantile columns grouped by target with the same "
            f"quantile levels (got {list(columns)})"
        )

    levels = [int(d) / 10 ** len(d) for d in digits]
    values = pred.numerical.unflatten(-1, (len(targets), -1))  # [..., V, Q]
    return values, values.new_tensor(levels)
