import torch

from sdm import Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor


class ToNumerical(Processor):
    """Move categorical columns into the numerical block.

    This is an stype conversion step:
    :class:`~sdm.tensor.CategoricalTensor` already stores ordinal category ids
    and the category vocabulary. ``ToNumerical`` reuses those existing ids,
    casts them to the numerical dtype, and clears the categorical block so
    numerical-only models can consume both original numerical and categorical
    features. Missing categorical values remain ``-1``.

    Only ``numerical`` and ``categorical`` columns are supported. Any other
    semantic type raises an error; drop those columns before this step.

    """

    requires_fit = False
    supported_stypes = frozenset({Stype.numerical, Stype.categorical})

    def _transform(self, table: TableTensor) -> TableTensor:
        """Return ``table`` with categorical columns moved to ``numerical``."""
        # Already numerical-only: nothing to move.
        if all(
            block.size(-1) == 0
            for block in (
                table.categorical,
                table.datetime,
                table.text,
                table.id,
            )
        ):
            return table

        # Cast the raw ordinal ids to the floating-point numerical dtype.
        categorical = table.categorical.code.to(table.numerical.dtype)
        columns = (
            *table.columns[Stype.numerical],
            *table.columns[Stype.categorical],
        )
        numerical = (
            categorical
            if table.numerical.size(-1) == 0
            else torch.cat((table.numerical, categorical), dim=-1)
        )
        return table.__class__(
            columns={Stype.numerical.value: columns},
            numerical=numerical,
        )
