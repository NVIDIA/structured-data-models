# Portions derived from DPM-Solver:
# Copyright (c) 2022 Cheng Lu
# Licensed under the MIT License; see third_party/dpm-solver/LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor


def dpm_solver_update(
    x: Tensor,
    data_start: Tensor,
    *,
    alpha_end: Tensor,
    sigma_start: Tensor,
    sigma_end: Tensor,
    lambda_delta: Tensor,
    data_mid: Tensor | None = None,
    r1: Tensor | float = 0.5,
) -> Tensor:
    """Apply a first- or second-order single-step DPM-Solver++ update.

    This implements Kumo-TS's ``algorithm_type="dpmsolver++"`` and
    ``solver_type="dpmsolver"`` formulas. Predictions are clean data, not
    noise: convert predicted noise ``eps`` using ``(x - sigma * eps) / alpha``
    at the corresponding time before calling this function.

    Without ``data_mid``, take a first-order step. For second order, first
    use this function to advance from the start to the intermediate time,
    then evaluate the model there and pass its clean-data prediction as
    ``data_mid`` together with the original ``x`` and full-interval
    coefficients. Model calls, time selection, and random draws are owned
    by the caller; this function is deterministic and differentiable.

    Args:
        x: Noisy sample at the start time, for example shape ``[B, K, L]``.
        data_start: Clean-data prediction at the start, with ``x``'s shape.
        alpha_end: Signal amplitude at the end time, not cumulative alpha
            squared as stored by ``DiffusionSchedule.alpha``.
        sigma_start: Positive noise standard deviation at the start time.
        sigma_end: Noise standard deviation at the end time.
        lambda_delta: End minus start half-logSNR, where
            ``lambda = log(alpha) - log(sigma)``.
        data_mid: Clean-data prediction at the intermediate sample/time,
            with ``x``'s shape. Omit for first order.
        r1: Intermediate fraction of the half-logSNR interval, strictly
            between zero and one. Ignored for first order. Uniform time
            spacing generally does not imply ``r1 = 0.5``; the caller must
            compute ``(lambda_mid - lambda_start) / lambda_delta``.

    Returns:
        Updated sample with ``x``'s shape. All tensors must be on the same
        device with compatible dtypes; standard PyTorch promotion applies.
        Coefficients and tensor ``r1`` must broadcast to ``x``: use scalars
        for shared times or ``[B, 1, 1]`` for per-window coefficients.
    """
    factor = alpha_end * (-lambda_delta).expm1()
    out = (sigma_end / sigma_start) * x - factor * data_start
    if data_mid is not None:
        out = out - (0.5 / r1) * factor * (data_mid - data_start)
    return out
