# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import TableTensor
from sdm.prior.tabular.transform import Choice, Identity, PriorTransform
from sdm.prior.tabular.transform._table import (
    categorical_mask,
    category_values,
    feature_values,
    mask_cells,
)
from sdm.prior.tabular.transform.missing._count import _sample_count


class MaskRows(PriorTransform):
    """Mask the same random rows in a fraction of feature columns.

    Args:
        rows: Range for the fraction of rows to mask.
        columns: Range for the fraction of feature columns to mask.
    """

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.05, 0.2),
        columns: tuple[float, float] = (0.5, 0.9),
    ) -> None:
        self.rows = rows
        self.columns = columns

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        num_rows, num_columns = x.shape
        missing = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        row_count = _sample_count(
            total=num_rows,
            low=self.rows[0],
            high=self.rows[1],
            device=x.device,
            generator=generator,
        )
        column_count = _sample_count(
            total=num_columns,
            low=self.columns[0],
            high=self.columns[1],
            device=x.device,
            generator=generator,
        )
        rows = torch.randperm(num_rows, device=x.device, generator=generator)[
            :row_count
        ]
        columns = torch.randperm(
            num_columns, device=x.device, generator=generator
        )[:column_count]
        missing[rows[:, None], columns[None, :]] = True
        return mask_cells(x, missing), y


class MaskBlock(PriorTransform):
    """Mask the same random rows in a fixed number of feature columns.

    Args:
        rows: Range for the fraction of rows to mask.
        num_columns: Inclusive range for the number of columns to mask.
    """

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.1, 0.3),
        num_columns: tuple[int, int] = (2, 5),
    ) -> None:
        self.rows = rows
        self.num_columns = num_columns

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        num_rows, num_columns = x.shape
        missing = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        if num_columns >= self.num_columns[0]:
            row_count = _sample_count(
                total=num_rows,
                low=self.rows[0],
                high=self.rows[1],
                device=x.device,
                generator=generator,
            )
            column_count = int(
                torch.randint(
                    low=self.num_columns[0],
                    high=min(self.num_columns[1], num_columns) + 1,
                    size=(),
                    device=x.device,
                    generator=generator,
                )
            )
            rows = torch.randperm(
                num_rows, device=x.device, generator=generator
            )[:row_count]
            columns = torch.randperm(
                num_columns, device=x.device, generator=generator
            )[:column_count]
            missing[rows[:, None], columns[None, :]] = True
        return mask_cells(x, missing), y


class MaskByTarget(PriorTransform):
    """Mask feature rows selected by a target class or numerical tail.

    Args:
        rows: Range for the fraction selected from a numerical target tail.
        num_columns: Inclusive range for the number of columns to mask.
    """

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.1, 0.3),
        num_columns: tuple[int, int] = (1, 3),
    ) -> None:
        self.rows = rows
        self.num_columns = num_columns

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        num_rows, num_columns = x.shape
        missing = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        if num_columns >= self.num_columns[0]:
            column_count = int(
                torch.randint(
                    low=self.num_columns[0],
                    high=min(self.num_columns[1], num_columns) + 1,
                    size=(),
                    device=x.device,
                    generator=generator,
                )
            )
            columns = torch.randperm(
                num_columns, device=x.device, generator=generator
            )[:column_count]

            if y.numerical.size(-1):
                target = y.numerical[:, 0]
                tail_count = _sample_count(
                    total=num_rows,
                    low=self.rows[0],
                    high=self.rows[1],
                    device=x.device,
                    generator=generator,
                )
                order = torch.argsort(target)
                high_tail = (
                    torch.rand((), device=x.device, generator=generator) < 0.5
                )
                rows = order[-tail_count:] if high_tail else order[:tail_count]
                missing[rows[:, None], columns[None, :]] = True
            elif num_rows:
                target = y.categorical.code[:, 0]
                classes = target.unique()
                class_index = int(
                    torch.randint(
                        high=classes.numel(),
                        size=(),
                        device=x.device,
                        generator=generator,
                    )
                )
                rows = (target == classes[class_index]).nonzero().flatten()
                missing[rows[:, None], columns[None, :]] = True
        return mask_cells(x, missing), y


class MaskCategorical(PriorTransform):
    """Mask independent random rows in selected categorical feature columns.

    Args:
        rows: Range for the fraction of rows to mask per column.
        num_columns: Inclusive range for the number of columns to mask.
    """

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.05, 0.25),
        num_columns: tuple[int, int] = (1, 3),
    ) -> None:
        self.rows = rows
        self.num_columns = num_columns

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        num_rows = x.size(-2)
        missing = torch.zeros(x.shape, dtype=torch.bool, device=x.device)
        categorical = categorical_mask(x).nonzero().flatten()
        if categorical.numel() >= max(1, self.num_columns[0]):
            column_count = int(
                torch.randint(
                    low=self.num_columns[0],
                    high=min(self.num_columns[1], categorical.numel()) + 1,
                    size=(),
                    device=x.device,
                    generator=generator,
                )
            )
            selected = categorical[
                torch.randperm(
                    categorical.numel(), device=x.device, generator=generator
                )[:column_count]
            ]
            for column in selected.tolist():
                row_count = _sample_count(
                    total=num_rows,
                    low=self.rows[0],
                    high=self.rows[1],
                    device=x.device,
                    generator=generator,
                )
                rows = torch.randperm(
                    num_rows, device=x.device, generator=generator
                )[:row_count]
                missing[rows, column] = True
        return mask_cells(x, missing), y


class MaskCensoring(PriorTransform):
    """Mask feature tails under MAR or self-masking MNAR censoring.

    Args:
        rows: Range for the fraction selected from each value tail.
        num_columns: Inclusive range for the number of columns to mask.
    """

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.05, 0.25),
        num_columns: tuple[int, int] = (1, 6),
    ) -> None:
        self.rows = rows
        self.num_columns = num_columns

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        values = feature_values(x)
        num_columns = values.size(-1)
        num_numerical = x.numerical.size(-1)
        categorical = categorical_mask(x)
        numerical = (~categorical).nonzero().flatten()
        if num_numerical > 0 and num_columns >= self.num_columns[0]:
            destination_count = int(
                torch.randint(
                    low=self.num_columns[0],
                    high=min(self.num_columns[1], num_columns) + 1,
                    size=(),
                    device=x.device,
                    generator=generator,
                )
            )
            destinations = torch.randperm(
                num_columns, device=x.device, generator=generator
            )[:destination_count]
            # Numerical destinations censor their own tail (MNAR) or the
            # tail of another numerical column (MAR). Categorical
            # destinations always use another column.
            for destination in destinations.tolist():
                sources = numerical[numerical != destination]
                source = destination
                if categorical[destination] or (
                    sources.numel() > 0
                    and torch.rand((), device=x.device, generator=generator)
                    < 0.5
                ):
                    source = int(
                        sources[
                            torch.randint(
                                high=sources.numel(),
                                size=(),
                                device=x.device,
                                generator=generator,
                            )
                        ]
                    )
                finite_rows = values[:, source].isfinite().nonzero().flatten()
                if finite_rows.numel() > 0:
                    count = _sample_count(
                        total=finite_rows.numel(),
                        low=self.rows[0],
                        high=self.rows[1],
                        device=x.device,
                        generator=generator,
                    )
                    order = finite_rows[
                        torch.argsort(values[finite_rows, source])
                    ]
                    high_tail = (
                        torch.rand((), device=x.device, generator=generator)
                        < 0.5
                    )
                    rows = order[-count:] if high_tail else order[:count]
                    values[rows, destination] = torch.nan
        return mask_cells(x, values.isnan()), y


class MaskStructured(PriorTransform):
    """Combine structured feature masking with conditional censoring."""

    def __init__(self) -> None:
        self.mask = Choice(
            MaskRows(),
            MaskBlock(),
            MaskByTarget(),
            MaskCategorical(),
            Identity(),
            weights=(1, 1, 1, 1, 4),
        )
        self.censoring = MaskCensoring()

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        observed = (
            x.numerical.isfinite().sum()
            + category_values(x.categorical, x.numerical.dtype)
            .isfinite()
            .sum()
        )
        result = self.mask(x, y, generator=generator)
        assert result is not None
        x, y = result
        unchanged = (
            x.numerical.isfinite().sum()
            + category_values(x.categorical, x.numerical.dtype)
            .isfinite()
            .sum()
        ) == observed
        if unchanged or (
            torch.rand((), device=x.device, generator=generator) < 0.5
        ):
            x, y = self.censoring(x, y, generator=generator)
        return x, y
