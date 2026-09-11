from typing import cast

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class ToNumerical(Processor):
    """Move categorical columns into the numerical block.

    This is an stype conversion step:
    :class:`~sdm.tensor.CategoricalTensor` already stores ordinal category ids
    and the category vocabulary. ``ToNumerical`` reuses those existing ids,
    casts them to the numerical dtype, and clears the categorical block so
    numerical-only models can consume both original numerical and categorical
    features. Missing categorical values remain ``-1`` by default, or a
    selected missing code can be converted to NaN.

    Args:
        missing_code: Categorical code to convert to NaN. If ``None``, all
            codes are preserved as numeric values.

    Unhandled semantic types are preserved unchanged.

    """

    requires_fit = False
    handles_stypes = frozenset({Stype.numerical, Stype.categorical})

    def __init__(self, *, missing_code: int | None = None) -> None:
        super().__init__()
        self.missing_code = missing_code

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
        if self.missing_code is not None:
            categorical = categorical.masked_fill(
                table.categorical.code == self.missing_code,
                float("nan"),
            )
        columns = (
            *table.columns[Stype.numerical],
            *table.columns[Stype.categorical],
        )
        numerical = (
            categorical
            if table.numerical.size(-1) == 0
            else torch.cat((table.numerical, categorical), dim=-1)
        )
        out = table.__class__(
            columns={Stype.numerical: columns},
            numerical=numerical,
        )
        return cast(
            TableTensor,
            torch.cat(
                (table.drop_stypes((Stype.numerical, Stype.categorical)), out),
                dim=-1,
            ),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if self.missing_code is None:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"missing_code={self.missing_code!r})"
        )
