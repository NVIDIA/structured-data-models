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
    features. Missing categorical values remain ``-1``.

    Unhandled semantic types are preserved unchanged.

    """

    requires_fit = False
    handles_stypes = frozenset({Stype.numerical, Stype.categorical})

    def _transform(self, table: TableTensor) -> TableTensor:
        """Return ``table`` with categorical columns moved to ``numerical``."""
        if table.categorical.size(-1) == 0:
            return table

        columns = (
            *table.columns[Stype.numerical],
            *table.columns[Stype.categorical],
        )
        if table.numerical.size(-1) == 0:
            numerical = table.categorical.to(table.numerical.dtype)
        else:
            numerical = table.numerical.new_empty(
                (*table.numerical.shape[:-1], len(columns))
            )
            width = table.numerical.size(-1)
            numerical[..., :width].copy_(table.numerical)
            # Copying casts directly into the output, without a full-sized
            # floating-point copy of the categorical codes before joining.
            numerical[..., width:].copy_(table.categorical.code)
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
