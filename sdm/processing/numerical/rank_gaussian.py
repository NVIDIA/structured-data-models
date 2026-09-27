# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm._memory import split_size
from sdm.processing import Processor
from sdm.processing.numerical._stats import _isfinite
from sdm.processing.numerical.quantile import _batched_interp


class RankGaussian(Processor):
    """Map interpolated empirical mid-ranks to standard normal quantiles.

    Each fitted value receives probability ``(L + R) / (2 * N)``, where
    ``L`` and ``R`` count finite fitted values strictly below and at or below
    it, and ``N`` is the number of finite fitted values. Ties share a rank.
    Query probabilities interpolate between the fitted values kept as knots
    and clamp to the endpoint probabilities, keeping finite outputs even
    outside the range.

    Each column keeps at most ``max_knots`` knots. A column with at most
    ``max_knots`` distinct finite fitted values keeps all of them, which
    matches interpolating between all fitted values. A column with more
    distinct values keeps the fitted values whose mid-ranks come closest to
    ``max_knots`` normal quantiles evenly spaced between its extremes.
    Knots then stay dense in the tails, where the normal quantile function
    magnifies rank errors, and outputs stay within about two knot spacings
    in normal scores of interpolating between all fitted values.

    NaN and infinite values are ignored during fitting. NaNs are preserved
    during transformation. Constant columns map to zero; columns without
    finite fitted values produce NaNs.

    Args:
        max_knots: Maximum number of knots per column, at least ``2``.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self, *, max_knots: int = 8192) -> None:
        super().__init__()
        if max_knots < 2:
            raise ValueError("max_knots must be at least 2.")
        self.max_knots = max_knots
        self.register_buffer("_values", torch.empty(0, dtype=torch.float64))
        self.register_buffer(
            "_probabilities",
            torch.empty(0, dtype=torch.float64),
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        # Columns are fitted on their own, so chunks of columns bound the
        # sorting and ranking temporaries of about ten doubles per cell.
        size = split_size(
            num_items=numerical.size(-1),
            item_bytes=numerical[..., :1].numel() * 10 * 8,
            device=numerical.device,
        )
        knots = [
            self._fit_columns(chunk) for chunk in numerical.split(size, dim=-1)
        ]
        values, probabilities = zip(*knots, strict=True)
        self._values = torch.cat(values, dim=-2)
        self._probabilities = torch.cat(probabilities, dim=-2)

    def _fit_columns(
        self,
        numerical: Tensor,  # [..., N, C]
    ) -> tuple[Tensor, Tensor]:  # [..., C, K] knot values and probabilities
        *batch, num_rows, num_columns = numerical.shape
        if num_rows == 0:
            missing = numerical.new_full(
                (*batch, num_columns, 1),
                torch.nan,
                dtype=torch.float64,
            )
            return missing, missing.clone()

        # Double precision keeps tail ranks open.
        columns = numerical.double().movedim(-1, -2)  # [..., C, N]
        finite = _isfinite(columns)
        count = finite.sum(dim=-1, keepdim=True)  # [..., C, 1]
        values = columns.masked_fill(~finite, torch.inf)
        del columns, finite
        values = values.sort(dim=-1).values.contiguous()
        left = torch.searchsorted(values, values, right=False)
        right = torch.searchsorted(values, values, right=True)
        probabilities = (left + right).double() / (2 * count.clamp_min(1))
        del right
        last = (count - 1).clamp_min(0)
        rows = torch.arange(num_rows, device=values.device)

        if num_rows <= self.max_knots:
            # Every fitted value is a knot, padded with the largest one.
            positions = rows.minimum(last)
        else:
            # Index of each sorted value among the column's distinct values;
            # a value starts a new one at its first occurrence.
            distinct = (left == rows).cumsum(dim=-1).sub_(1)
            num_distinct = distinct.gather(-1, last) + 1
            # Knots at the first occurrence of each distinct value, padded
            # with the largest one.
            steps = torch.arange(self.max_knots, device=values.device)
            positions = torch.searchsorted(
                distinct,
                steps.minimum(num_distinct - 1),
            )
            del distinct
            # With more distinct values than knots, the rows whose mid-ranks
            # come closest to normal quantiles evenly spaced from the
            # smallest to the largest value. Mid-ranks never decrease along
            # the sorted rows, padding included.
            lower = torch.special.ndtri(probabilities[..., :1])
            upper = torch.special.ndtri(probabilities.gather(-1, last))
            quantiles = torch.special.ndtr(
                lower.lerp(upper, steps.double() / (self.max_knots - 1))
            )
            above = torch.searchsorted(probabilities, quantiles).minimum(last)
            below = (above - 1).clamp_(min=0)
            closer = (quantiles - probabilities.gather(-1, below)) < (
                probabilities.gather(-1, above) - quantiles
            )
            spaced = torch.where(closer, below, above)
            spaced[..., :1] = 0
            spaced[..., -1:] = last
            positions = torch.where(
                num_distinct <= self.max_knots,
                positions,
                spaced,
            )
        del left

        missing = count == 0
        return (
            values.gather(-1, positions).masked_fill_(missing, torch.nan),
            probabilities.gather(-1, positions).masked_fill_(
                missing,
                torch.nan,
            ),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        numerical = table.numerical
        columns = numerical.movedim(-1, -2)  # [..., C, R]
        values = self._values.flatten(end_dim=-2)
        probabilities = self._probabilities.flatten(end_dim=-2)
        output = torch.empty_like(numerical)
        # Rows transform on their own, so chunks of rows bound the
        # interpolation temporaries of about a dozen doubles per cell.
        size = split_size(
            num_items=columns.size(-1),
            item_bytes=columns[..., :1].numel() * 12 * 8,
            device=numerical.device,
        )
        for rows, out in zip(
            columns.split(size, dim=-1),
            output.split(size, dim=-2),
            strict=True,
        ):
            quantiles = _batched_interp(
                rows.flatten(end_dim=-2).to(values.dtype).contiguous(),
                values,
                probabilities,
            )
            normal = torch.special.ndtri(quantiles).reshape(rows.shape)
            out.copy_(normal.masked_fill_(rows.isnan(), torch.nan).mT)
        return table.replace_blocks(numerical=output)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"max_knots={self.max_knots})"
        )
