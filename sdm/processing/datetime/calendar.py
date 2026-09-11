from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor

from sdm import NaT, Stype, TableTensor
from sdm.processing import Processor

US_PER_MINUTE = 60 * 1_000_000
US_PER_HOUR = 60 * US_PER_MINUTE
US_PER_DAY = 24 * US_PER_HOUR


class AddCalendarFields(Processor):
    r"""Add numerical calendar fields derived from datetime columns.

    Args:
        fields: The calendar fields to add.
        encoding: How to encode each field. ``"raw"`` uses raw integer values.
            ``"cyclic"`` uses sine and cosine values.
    """

    handles_stypes = frozenset({Stype.datetime})
    requires_fit = False

    def __init__(
        self,
        fields: Sequence[
            Literal[
                "minute",
                "hour",
                "weekday",
                "day_of_month",
                "month",
            ]
        ],
        encoding: Literal["raw", "cyclic"] = "raw",
    ) -> None:
        super().__init__()

        if len(fields) != len(set(fields)):
            raise ValueError("Expected datetime fields to be unique")

        self.fields = fields
        self.encoding = encoding

    def _calendar_fields(self, table: TableTensor) -> list[Tensor]:
        datetime = table.datetime
        na_mask = datetime == NaT

        days: Tensor | None = None
        if len({"weekday", "day_of_month", "month"} & set(self.fields)) > 0:
            # Every int64 microsecond timestamp fits within int32 epoch days.
            days = datetime.div(US_PER_DAY, rounding_mode="floor").to(
                torch.int32
            )

        year = month = day = None
        if len({"day_of_month", "month"} & set(self.fields)) > 0:
            assert days is not None
            year, month, day = _civil_from_days(days)

        if self.encoding != "cyclic" or "day_of_month" not in self.fields:
            year = None
        if "day_of_month" not in self.fields:
            day = None
        if "weekday" not in self.fields:
            days = None

        minutes: Tensor | None = None
        if len({"minute", "hour"} & set(self.fields)) > 0:
            minutes = (
                datetime.remainder(US_PER_DAY)
                .div_(US_PER_MINUTE, rounding_mode="floor")
                .to(torch.int32)
            )

        outs: list[Tensor] = []
        for field in self.fields:
            if field == "minute":
                assert minutes is not None
                out = minutes.remainder(60)
            elif field == "hour":
                assert minutes is not None
                out = minutes.div(60, rounding_mode="floor")
            elif field == "weekday":
                assert days is not None
                out = days.add_(3).remainder_(7)
                days = None
            elif field == "month":
                assert month is not None
                out = month - 1
            elif field == "day_of_month":
                assert day is not None
                out = day.sub_(1)
                day = None
            else:
                raise ValueError(
                    f"{self.__class__.__name__!r} received unsupported "
                    f"field {field!r}"
                )

            if self.encoding == "cyclic":
                if field == "minute":
                    out = out / 60.0
                elif field == "hour":
                    out = out / 24.0
                elif field == "weekday":
                    out = out / 7.0
                elif field == "month":
                    out = out / 12.0
                else:
                    assert field == "day_of_month"
                    assert year is not None
                    assert month is not None
                    period = torch.tensor(
                        [-1, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                        device=year.device,
                        dtype=year.dtype,
                    )
                    period = period[month]
                    leap_year = (year.remainder(4) == 0) & (
                        (year.remainder(100) != 0) | (year.remainder(400) == 0)
                    )
                    period.masked_fill_(leap_year & (month == 2), 29)
                    out = out / period
                    del period, leap_year
                    year = None
            out = out.to(table.numerical.dtype)
            outs.append(out.masked_fill_(na_mask, float("nan")))

        return outs

    def _transform(self, table: TableTensor) -> TableTensor:
        if len(self.fields) == 0:
            return table

        # Release calendar intermediates before assembling the output.
        fields = self._calendar_fields(table)
        width = table.numerical.size(-1)
        shape = (table.datetime.size(-1), len(fields))
        encoding_width = 2 if self.encoding == "cyclic" else 1
        numerical = table.numerical.new_empty(
            (
                *table.datetime.shape[:-1],
                width + shape[0] * shape[1] * encoding_width,
            )
        )
        if width > 0:
            numerical[..., :width].copy_(table.numerical)

        if self.encoding == "cyclic":
            out = numerical[..., width:].unflatten(-1, (*shape, 2))
            for index in reversed(range(len(fields))):
                field = fields.pop()
                field.mul_(2 * torch.pi)
                torch.sin(field, out=out[..., index, 0])
                torch.cos(field, out=out[..., index, 1])
                del field

            columns = tuple(
                f"{column}__{field}__{fn}"
                for column in table.columns[Stype.datetime]
                for field in self.fields
                for fn in ("sin", "cos")
            )
        else:
            assert self.encoding == "raw"
            out = numerical[..., width:].unflatten(-1, shape)
            for index in reversed(range(len(fields))):
                out[..., index].copy_(fields.pop())
            columns = tuple(
                f"{column}__{field}"
                for column in table.columns[Stype.datetime]
                for field in self.fields
            )

        return table.__class__(
            columns={
                **table.columns,
                Stype.numerical: (*table.columns[Stype.numerical], *columns),
            },
            numerical=numerical,
            categorical=table.categorical,
            datetime=table.datetime,
            text=table.text,
            id=table.id,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        field_repr = "".join(
            f"{' ' * (indent + 4)}{field!r},\n" for field in self.fields
        )
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{' ' * (indent + 2)}fields=[\n"
            f"{field_repr}"
            f"{' ' * (indent + 2)}],\n"
            f"{' ' * (indent + 2)}encoding={self.encoding!r},\n"
            f"{' ' * indent})"
        )


def _civil_from_days(days: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert days since the Unix epoch to Gregorian year, month, and day."""
    shifted = days + 719_468
    era = shifted.div(146_097, rounding_mode="floor")
    day_of_era = shifted.sub_(era * 146_097)
    year_of_era = day_of_era.div(1_460, rounding_mode="floor")
    torch.sub(day_of_era, year_of_era, out=year_of_era)
    year_of_era.add_(day_of_era.div(36_524, rounding_mode="floor")).sub_(
        day_of_era.div(146_096, rounding_mode="floor")
    ).div_(365, rounding_mode="floor")
    year = era.mul_(400).add_(year_of_era)
    day_of_year = day_of_era.sub_(
        (365 * year_of_era)
        .add_(year_of_era.div(4, rounding_mode="floor"))
        .sub_(year_of_era.div(100, rounding_mode="floor"))
    )
    del year_of_era
    month_prime = (5 * day_of_year).add_(2).div_(153, rounding_mode="floor")
    month_offset = (153 * month_prime).add_(2).div_(5, rounding_mode="floor")
    day = day_of_year.sub_(month_offset).add_(1)
    del month_offset
    month = month_prime.add_(2).remainder_(12).add_(1)
    year.add_(month <= 2)
    return year, month, day
