from typing import cast

import torch

from sdm import NaT, Stype, TableTensor
from sdm.processing import Processor


class ToNumerical(Processor):
    """Move categorical and datetime columns into the numerical block.

    This is an stype conversion step:
    :class:`~sdm.tensor.CategoricalTensor` already stores ordinal category ids
    and the category vocabulary. ``ToNumerical`` reuses those existing ids,
    casts them to the numerical dtype, and clears the categorical block so
    numerical-only models can consume both original numerical and categorical
    features. Missing categorical values remain ``-1``.

    Datetime values are cast to the numerical dtype. Missing datetimes become
    NaN.

    Unhandled semantic types are preserved unchanged.

    """

    requires_fit = False
    handles_stypes = frozenset(
        {Stype.numerical, Stype.categorical, Stype.datetime}
    )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Return ``table`` with categorical columns moved to ``numerical``."""
        # Already numerical-only: nothing to move.
        if all(
            stype == Stype.numerical or block.size(-1) == 0
            for stype, block in table.items()
        ):
            return table

        # Casting to the (floating-point) numerical dtype also unwraps a
        # CategoricalTensor to its raw ordinal ids as a plain tensor.
        categorical = table.categorical.to(table.numerical.dtype)
        datetime = table.datetime.to(table.numerical.dtype)
        datetime = datetime.masked_fill(table.datetime == NaT, float("nan"))
        columns = (
            *table.columns[Stype.numerical],
            *table.columns[Stype.categorical],
            *table.columns[Stype.datetime],
        )
        numerical = torch.cat(
            (table.numerical, categorical, datetime),
            dim=-1,
        )
        out = table.__class__(
            columns={Stype.numerical: columns},
            numerical=numerical,
        )
        return cast(
            TableTensor,
            torch.cat(
                (
                    table.drop_stypes(
                        (Stype.numerical, Stype.categorical, Stype.datetime)
                    ),
                    out,
                ),
                dim=-1,
            ),
        )
