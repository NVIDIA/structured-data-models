# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import Processor
from sdm.processing._random import (
    SeedMixin,
    count,
    rank,
    subset,
)
from sdm.processing.categorical._categorical import _check_categories
from sdm.processing.missing._cells import cell_values, mask_cells

_RANDOM = -1


class _MaskRules(SeedMixin, Processor):
    r"""Mask the rows in a tail of a source column, rule by rule.

    Rule ``k`` masks the rows in a tail of ``source[..., k]`` in the columns
    ``rule_columns[..., k, :]``. The tail size is drawn from ``rows`` over
    the finite source rows, at the high end if ``high[..., k]``. Fit sets the
    rules, and the transform draws the random rows.
    """

    handles_stypes = (Stype.numerical, Stype.categorical)
    requires_fit = True
    _options: tuple[str, ...] = ()

    _source: Tensor
    _count_draw: Tensor
    _high: Tensor
    _rule_columns: Tensor

    def __init__(self, rows: tuple[float, float], target: str | None) -> None:
        super().__init__()
        self.rows = rows
        self.target = target
        self.register_buffer("_source", torch.empty(0, dtype=torch.long))
        self.register_buffer("_count_draw", torch.empty(0))
        self.register_buffer("_high", torch.empty(0, dtype=torch.bool))
        self.register_buffer("_rule_columns", torch.empty(0, dtype=torch.bool))

    def _names(self, table: TableTensor) -> tuple[str, ...]:
        return (
            *table.columns[Stype.numerical],
            *table.columns[Stype.categorical],
        )

    def _layout(self, table: TableTensor) -> tuple[Tensor, Tensor]:
        r"""Return the feature and the categorical mask of the columns."""
        names = self._names(table)
        features = torch.tensor(
            [name != self.target for name in names], device=table.device
        )
        categorical = torch.arange(len(names), device=table.device) >= len(
            table.columns[Stype.numerical]
        )
        return features, categorical

    def _set_rules(
        self,
        source: Tensor,
        rule_columns: Tensor,
        *,
        generator: torch.Generator | None,
    ) -> None:
        u = torch.rand(
            (2, *source.shape),
            device=source.device,
            dtype=torch.float64,
            generator=generator,
        )
        self._set_buffer("_source", source)
        self._set_buffer("_rule_columns", rule_columns)
        self._set_buffer("_count_draw", u[0])
        self._set_buffer("_high", u[1] < 0.5)

    def _sources(self, table: TableTensor, values: Tensor) -> Tensor:
        return values

    def _transform(self, table: TableTensor) -> TableTensor:
        values = cell_values(table)  # [..., R, C]
        *batch, num_rows, _ = values.shape
        index = self._source.clamp(min=0)[..., None, :]
        sources = (
            self._sources(table, values)
            .gather(-1, index.expand(*batch, num_rows, -1))
            .mT
        )  # [..., K, R]
        u = torch.rand(
            sources.shape,
            device="cpu",
            generator=self._transform_generator(),
        )
        u = u.to(sources)
        sources = torch.where((self._source == _RANDOM)[..., None], u, sources)
        finite = sources.isfinite()
        # Shuffle before ranking so tied values do not follow row order.
        order = u.argsort(dim=-1)
        shuffled = sources.masked_fill(~finite, torch.inf).gather(-1, order)
        position = torch.empty_like(order).scatter_(
            -1, order, rank(shuffled, dim=-1)
        )
        num_finite = finite.sum(dim=-1)
        total = num_finite.double()
        lower = (total * self.rows[0]).ceil().clamp(min=1)
        upper = torch.maximum(
            lower, torch.minimum(total, (total * self.rows[1]).floor())
        )
        k = (
            (lower + self._count_draw.double() * (upper - lower + 1))
            .floor()
            .long()
            .minimum(num_finite)
        )
        # A categorical target uses one as a sentinel for its complete class.
        k = torch.where(self._count_draw >= 1, num_finite, k)
        # Non-finite rows rank after the high tail.
        start = torch.where(self._high, num_finite - k, 0)[..., None]
        hit = (position >= start) & (position < start + k[..., None])
        missing = hit.mT.to(values.dtype) @ self._rule_columns.to(values.dtype)
        return mask_cells(table, missing > 0)

    def __repr__(self, *, indent: int = 0) -> str:
        args = ", ".join(
            f"{name}={getattr(self, name)!r}" for name in self._options
        )
        return f"{' ' * indent}{self.__class__.__name__}({args})"


def _rules_per_column(selected: Tensor, num_rules: int) -> Tensor:
    r"""Give each selected column its own rule, in column order."""
    rule = (selected.cumsum(dim=-1) - 1).masked_fill(~selected, -1)
    k = torch.arange(num_rules, device=selected.device)
    return rule[..., None, :] == k[:, None]  # [..., K, C]


def _pick_columns(
    eligible: Tensor,
    num_columns: tuple[int, int],
    batch: tuple[int, ...],
    *,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Select eligible columns, or none if fewer than the minimum."""
    low, high = num_columns
    num_eligible = eligible.sum()
    upper = num_eligible.clamp(max=high).clamp(min=low)
    u = torch.rand(batch, device=eligible.device, generator=generator)
    num_selected = (low + u * (upper - low + 1)).floor().long()
    selected = subset(
        eligible=eligible.expand(*batch, -1),
        count=num_selected,
        generator=generator,
    )
    return selected & (num_eligible >= low)


class MaskRows(_MaskRules):
    r"""Mask the same random rows in many columns under structured MCAR.

    When the processor is fitted, each table selects a fraction of its
    feature columns and a fraction of its rows. The transform draws the
    rows, and every selected column loses the same rows.

    The transform uses all rows of the table together, so apply it to the
    whole dataset in one call.

    Args:
        rows: Range of the fraction of the rows that are masked.
        columns: Range of the fraction of the feature columns that are
            masked.
        target: A column that is never masked.
    """

    _options = ("rows", "columns", "target")

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.05, 0.2),
        columns: tuple[float, float] = (0.5, 0.9),
        target: str | None = None,
    ) -> None:
        super().__init__(rows, target)
        self.columns = columns

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._draw_seed(generator)
        batch = table.size()[:-2]
        features = self._layout(table)[0].expand(*batch, -1)
        num_features = features.sum(dim=-1)
        num_selected = count(num_features, *self.columns, generator=generator)
        selected = subset(features, num_selected, generator=generator)
        self._set_rules(
            source=torch.full((*batch, 1), _RANDOM, device=table.device),
            rule_columns=selected[..., None, :],
            generator=generator,
        )


class MaskBlock(_MaskRules):
    r"""Mask the same random rows in a few columns under structured MCAR.

    When the processor is fitted, each table selects a number of its feature
    columns and a fraction of its rows.

    The transform uses all rows of the table together, so apply it to the
    whole dataset in one call.

    Args:
        rows: Range of the fraction of the rows that are masked.
        num_columns: Range of the number of the masked columns.
        target: A column that is never masked.
    """

    _options = ("rows", "num_columns", "target")

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.1, 0.3),
        num_columns: tuple[int, int] = (2, 5),
        target: str | None = None,
    ) -> None:
        super().__init__(rows, target)
        self.num_columns = num_columns

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._draw_seed(generator)
        batch = table.size()[:-2]
        features = self._layout(table)[0]
        selected = _pick_columns(
            features, self.num_columns, batch, generator=generator
        )
        self._set_rules(
            source=torch.full((*batch, 1), _RANDOM, device=table.device),
            rule_columns=selected[..., None, :],
            generator=generator,
        )


class MaskByTarget(_MaskRules):
    r"""Mask rows selected by a target class or tail under MAR.

    When the processor is fitted, each table selects a number of its feature
    columns. A categorical target selects one of its present classes, and the
    transform masks the rows of this class. A numerical target selects a
    fraction of its rows at the low or the high end, and the transform masks
    the rows in this tail. The target itself is never masked.

    The transform uses all rows of the table together, so apply it to the
    whole dataset in one call.

    Args:
        target: The target column.
        rows: Range of the fraction of the rows in the tail of a numerical
            target.
        num_columns: Range of the number of the masked columns.
    """

    _options = ("target", "rows", "num_columns")

    _target_class: Tensor

    def __init__(
        self,
        target: str,
        *,
        rows: tuple[float, float] = (0.1, 0.3),
        num_columns: tuple[int, int] = (1, 3),
    ) -> None:
        super().__init__(rows, target)
        self.num_columns = num_columns
        self._categories: BufferList[Tensor] = BufferList()
        self.register_buffer("_target_class", torch.empty(0, dtype=torch.long))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        assert self.target is not None
        self._draw_seed(generator)
        batch = table.size()[:-2]
        device = table.device
        features = self._layout(table)[0]
        names = self._names(table)
        selected = _pick_columns(
            features, self.num_columns, batch, generator=generator
        )
        regression = table.stype(self.target) == Stype.numerical
        # A categorical target reads the class rows as the last source.
        source = names.index(self.target) if regression else len(names)
        self._set_rules(
            source=torch.full((*batch, 1), source, device=device),
            rule_columns=selected[..., None, :],
            generator=generator,
        )
        target_class = torch.zeros(batch, dtype=torch.long, device=device)
        if not regression:
            self._categories = BufferList(table.categorical.categories)
            self._set_buffer("_count_draw", torch.ones_like(self._count_draw))
            position = table.columns[Stype.categorical].index(self.target)
            num_classes = len(table.categorical.categories[position])
            code = table.categorical.code[..., position].long()
            if num_classes:
                present = torch.zeros((*batch, num_classes), device=device)
                present.scatter_add_(
                    -1,
                    code.clamp_min(0),
                    (code >= 0).to(present.dtype),
                )
                u = torch.rand(
                    (*batch, num_classes), device=device, generator=generator
                )
                target_class = u.masked_fill(present == 0, -1.0).argmax(dim=-1)
        self._set_buffer("_target_class", target_class)

    def _sources(self, table: TableTensor, values: Tensor) -> Tensor:
        assert self.target is not None
        if table.stype(self.target) == Stype.numerical:
            return values
        _check_categories(table, self._categories)
        position = table.columns[Stype.categorical].index(self.target)
        code = table.categorical.code[..., position]
        # Only the rows of the class are finite.
        is_class = torch.zeros_like(values[..., 0]).masked_fill(
            code != self._target_class[..., None], torch.nan
        )
        return torch.cat([values, is_class[..., None]], dim=-1)


class MaskCategorical(_MaskRules):
    r"""Mask random rows in categorical columns under structured MCAR.

    When the processor is fitted, each table selects a number of its
    categorical feature columns. Each selected column draws its own fraction
    of rows, and the transform draws its own rows.

    The transform uses all rows of the table together, so apply it to the
    whole dataset in one call.

    Args:
        rows: Range of the fraction of the rows that a column loses.
        num_columns: Range of the number of the masked columns.
        target: A column that is never masked.
    """

    _options = ("rows", "num_columns", "target")

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.05, 0.25),
        num_columns: tuple[int, int] = (1, 3),
        target: str | None = None,
    ) -> None:
        super().__init__(rows, target)
        self.num_columns = num_columns

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._draw_seed(generator)
        batch = table.size()[:-2]
        features, categorical = self._layout(table)
        eligible = features & categorical
        high = self.num_columns[1]
        selected = _pick_columns(
            eligible, self.num_columns, batch, generator=generator
        )
        self._set_rules(
            source=torch.full((*batch, high), _RANDOM, device=table.device),
            rule_columns=_rules_per_column(selected, high),
            generator=generator,
        )


class MaskCensoring(_MaskRules):
    r"""Mask one value tail under MAR or self-masking MNAR censoring.

    When the processor is fitted, each table selects a number of its feature
    columns. Each column reads its own values or the values of another
    numerical column, and draws a fraction of the rows at the low or the
    high end. The transform masks the rows in this tail.

    The transform uses all rows of the table together, so apply it to the
    whole dataset in one call.

    Args:
        rows: Range of the fraction of the finite rows in a tail.
        num_columns: Range of the number of the masked columns.
        target: A column that is never masked or read.
    """

    _options = ("rows", "num_columns", "target")

    def __init__(
        self,
        *,
        rows: tuple[float, float] = (0.05, 0.25),
        num_columns: tuple[int, int] = (1, 6),
        target: str | None = None,
    ) -> None:
        super().__init__(rows, target)
        self.num_columns = num_columns

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._draw_seed(generator)
        batch = table.size()[:-2]
        device = table.device
        features, categorical = self._layout(table)
        num_columns = features.numel()
        numerical = features & ~categorical
        selected = _pick_columns(
            features, self.num_columns, batch, generator=generator
        )
        rule_columns = _rules_per_column(selected, self.num_columns[1])
        column = rule_columns.long().argmax(dim=-1)
        others = numerical.expand(*batch, self.num_columns[1], -1) & (
            torch.arange(num_columns, device=device) != column[..., None]
        )
        has_other = others.any(dim=-1)
        u = torch.rand(
            (*batch, self.num_columns[1], num_columns + 1),
            device=device,
            generator=generator,
        )
        cross = categorical[column] | (has_other & (u[..., 0] < 0.5))
        other = u[..., 1:].masked_fill(~others, -1.0).argmax(dim=-1)
        source = torch.where(cross, other, column)
        rule_columns &= (has_other | ~cross)[..., None]
        self._set_rules(
            source=source,
            rule_columns=rule_columns,
            generator=generator,
        )
