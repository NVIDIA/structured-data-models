import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor

_MICROSECONDS_PER_MINUTE = 60 * 1_000_000
_MICROSECONDS_PER_HOUR = 60 * _MICROSECONDS_PER_MINUTE
_MICROSECONDS_PER_DAY = 24 * _MICROSECONDS_PER_HOUR
_MISSING_TIMESTAMP = torch.iinfo(torch.int64).min
_FEATURE_NAMES = (
    "minute",
    "hour",
    "weekday",
    "day_of_month",
    "day_of_year",
)


class DatetimeFeatures(Processor):
    r"""Expand microsecond timestamps into normalized calendar features.

    Each datetime column produces minute-of-hour, hour-of-day, Monday-based
    weekday, zero-based day-of-month, and zero-based day-of-year channels.
    Missing timestamps produce zeros for every channel.

    Task-relative elapsed time is not generated because it requires relational
    task anchors and row-to-example assignments. Calendar conversion operates
    directly on the full microsecond timestamp range without applying pandas'
    narrower datetime bounds.

    Args:
        preserve_datetime: Whether to retain the source datetime columns in
            addition to the generated numerical columns.
    """

    supported_stypes = frozenset({Stype.datetime})
    requires_fit = False

    def __init__(self, *, preserve_datetime: bool = False) -> None:
        super().__init__()
        self.preserve_datetime = preserve_datetime

    def _transform(self, table: TableTensor) -> TableTensor:
        timestamp = table.datetime
        missing = timestamp == _MISSING_TIMESTAMP
        safe_timestamp = timestamp.masked_fill(missing, 0)
        days = safe_timestamp.div(
            _MICROSECONDS_PER_DAY,
            rounding_mode="floor",
        )
        time_of_day = safe_timestamp.remainder(_MICROSECONDS_PER_DAY)
        minute = time_of_day.div(
            _MICROSECONDS_PER_MINUTE,
            rounding_mode="floor",
        ).remainder(60)
        hour = time_of_day.div(
            _MICROSECONDS_PER_HOUR,
            rounding_mode="floor",
        )

        year, month, day = _civil_from_days(days)
        leap = (year.remainder(4) == 0) & (
            (year.remainder(100) != 0) | (year.remainder(400) == 0)
        )
        month_lengths = timestamp.new_tensor(
            (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
        )
        days_in_month = month_lengths[month - 1]
        days_in_month = days_in_month + (leap & (month == 2))
        month_starts = timestamp.new_tensor(
            (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
        )
        day_of_year = month_starts[month - 1] + day - 1
        day_of_year = day_of_year + (leap & (month > 2))
        days_in_year = 365 + leap
        monday_weekday = (days + 3).remainder(7)

        float_dtype = table.numerical.dtype
        features = torch.stack(
            (
                minute.to(float_dtype) / 60,
                hour.to(float_dtype) / 24,
                monday_weekday.to(float_dtype) / 7,
                (day - 1).to(float_dtype) / days_in_month.to(float_dtype),
                day_of_year.to(float_dtype) / days_in_year.to(float_dtype),
            ),
            dim=-1,
        )
        features = features.masked_fill(missing.unsqueeze(-1), 0)
        numerical = features.flatten(-2)
        numerical_columns = tuple(
            f"{column}__{feature}"
            for column in table.columns[Stype.datetime]
            for feature in _FEATURE_NAMES
        )

        columns: dict[str, tuple[str, ...]] = {
            Stype.numerical.value: numerical_columns,
        }
        if self.preserve_datetime:
            columns[Stype.datetime.value] = table.columns[Stype.datetime]
        return table.__class__(
            columns=columns,
            numerical=numerical,
            datetime=table.datetime if self.preserve_datetime else None,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"preserve_datetime={self.preserve_datetime})"
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
    day = (
        day_of_year - (153 * month_prime + 2).div(5, rounding_mode="floor") + 1
    )
    month = month_prime + torch.where(month_prime < 10, 3, -9)
    year = year + (month <= 2)
    return year, month, day
