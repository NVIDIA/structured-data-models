from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor

US_PER_MINUTE = 60 * 1_000_000
US_PER_HOUR = 60 * US_PER_MINUTE
US_PER_DAY = 24 * US_PER_HOUR


class EncodeCalendar(Processor):
    r"""Separate timestamps into numerical calendar features.

    Args:
        features: The datetime features to extract.
    """

    supported_stypes = frozenset({Stype.datetime})
    requires_fit = False

    def __init__(
        self,
        features: Sequence[
            Literal[
                "minute",
                "hour",
                "weekday",
                "day_of_month",
                "month",
            ]
        ],
    ) -> None:
        super().__init__()
        self.features = features

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.datetime.size(-1) == 0 or len(self.features) == 0:
            return table

        datetime = table.datetime
        na_mask = datetime == torch.iinfo(datetime.dtype).min

        outs: list[Tensor] = []
        for feature in self.features:
            if feature in ("minute", "hour"):
                time_of_day = datetime.remainder(US_PER_DAY)
                if feature == "minute":
                    out = time_of_day.div(US_PER_MINUTE, rounding_mode="floor")
                    out = out.remainder(60)
                    outs.append(out)
                else:
                    assert feature == "hour"
                    out = time_of_day.div(US_PER_HOUR, rounding_mode="floor")
                    outs.append(out)
            elif feature in ("weekday", "day_of_month", "month"):
                days = datetime.div(US_PER_DAY, rounding_mode="floor")
                if feature == "weekday":
                    out = (days + 3).remainder(7)
                    outs.append(out)
                else:
                    _, month, day = _civil_from_days(days)
                    if feature == "month":
                        outs.append(month - 1)
                    else:
                        assert feature == "day_of_month"
                        outs.append(day - 1)
            else:
                raise ValueError(
                    f"'{self.__class__.__name__}' received unsupported "
                    f"feature '{feature}'"
                )

        out = torch.stack(outs, dim=-1).to(table.numerical.dtype)
        out = out.masked_fill(na_mask.unsqueeze(-1), float("nan"))
        out = out.flatten(-2, -1)

        columns = tuple(
            f"{column}__{feature}"
            for column in table.columns[Stype.datetime]
            for feature in self.features
        )

        out_table = TableTensor(
            columns={Stype.numerical: columns},
            numerical=out,
        )

        return cast(TableTensor, torch.cat([table, out_table], dim=-1))

    def __repr__(self, *, indent: int = 0) -> str:
        feature_repr = "".join(
            f"{' ' * (indent + 2)}'{feature}',\n" for feature in self.features
        )
        return (
            f"{' ' * indent}{self.__class__.__name__}([\n"
            f"{feature_repr}"
            f"{' ' * indent}])"
        )


def _civil_from_days(days: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Convert days since the Unix epoch to Gregorian year, month, and day."""
    shifted = days + 719_468
    era = shifted.div(146_097, rounding_mode="floor")
    day_of_era = shifted - era * 146_097
    year_of_era = (
        day_of_era
        - day_of_era.div(1_460, rounding_mode="floor")
        + day_of_era.div(36_524, rounding_mode="floor")
        - day_of_era.div(146_096, rounding_mode="floor")
    ).div(365, rounding_mode="floor")
    year = year_of_era + era * 400
    day_of_year = day_of_era - (
        365 * year_of_era
        + year_of_era.div(4, rounding_mode="floor")
        - year_of_era.div(100, rounding_mode="floor")
    )
    month_prime = (5 * day_of_year + 2).div(153, rounding_mode="floor")
    month_offset = (153 * month_prime + 2).div(5, rounding_mode="floor")
    day = day_of_year - month_offset + 1
    month = month_prime + torch.where(month_prime < 10, 3, -9)
    year = year + (month <= 2)
    return year, month, day
