import torch
from torch import Tensor

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

    def _transform(self, input: TableTensor) -> TableTensor:
        """Return ``input`` with categorical columns moved to ``numerical``."""
        # Already numerical-only: nothing to move.
        if all(
            stype == Stype.numerical or block.size(-1) == 0
            for stype, block in input.items()
        ):
            return input

        dtype = input.numerical.dtype
        blocks: list[Tensor] = []
        columns: list[str] = []
        for stype, block in input.items():
            if block.size(-1) == 0:
                continue
            # Casting to the (floating-point) numerical dtype also unwraps a
            # CategoricalTensor to its raw ordinal ids as a plain tensor.
            blocks.append(block.to(dtype))  # [..., C_stype]
            columns.extend(input.columns[stype])

        numerical = (  # [..., sum(C_stype)]
            blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
        )
        return input.__class__(
            columns={Stype.numerical: tuple(columns)},
            numerical=numerical,
        )
