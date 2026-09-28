# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing.numerical.quantile import _batched_interp


class RankGaussian(Processor):
    """Map interpolated empirical mid-ranks to standard normal quantiles.

    Each fitted value receives probability ``(L + R) / (2 * N)``, where
    ``L`` and ``R`` count finite fitted values strictly below and at or below
    it, and ``N`` is the number of finite fitted values. Ties share a rank.
    Query probabilities interpolate between fitted values and clamp to the
    endpoint probabilities, keeping finite outputs even outside the range.

    NaN and infinite values are ignored during fitting. NaNs are preserved
    during transformation. Constant columns map to zero; columns without
    finite fitted values produce NaNs. All fitted values are retained.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_values", torch.empty(0))
        self.register_buffer("_probabilities", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        # [..., N, C] -> [..., C, N]; double precision keeps tail ranks open.
        numerical = table.numerical.double().movedim(-1, -2)
        values = numerical.masked_fill(~numerical.isfinite(), torch.inf)
        values = values.sort(dim=-1).values.contiguous()
        finite = values.isfinite()
        count = finite.sum(dim=-1, keepdim=True)
        left = torch.searchsorted(values, values, right=False)
        right = torch.searchsorted(values, values, right=True)
        probabilities = (left + right).double() / (2 * count.clamp_min(1))

        # Pad missing observations with the last finite knot and its rank.
        last = (count - 1).clamp_min(0)
        self._values = torch.where(finite, values, values.gather(-1, last))
        self._probabilities = torch.where(
            finite, probabilities, probabilities.gather(-1, last)
        )
        self._values.masked_fill_(count == 0, torch.nan)
        self._probabilities.masked_fill_(count == 0, torch.nan)

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        columns = numerical.movedim(-1, -2)
        n_rows = columns.size(-1)
        n_fitted = self._values.size(-1)
        probabilities = _batched_interp(
            columns.reshape(-1, n_rows).to(self._values.dtype).contiguous(),
            self._values.reshape(-1, n_fitted),
            self._probabilities.reshape(-1, n_fitted),
        )
        output = torch.special.ndtri(probabilities).reshape(columns.shape)
        output = output.masked_fill(columns.isnan(), torch.nan)
        return table.replace_blocks(
            numerical=output.movedim(-1, -2).to(numerical.dtype)
        )
