from typing import cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.tensor import CategoricalTensor, StringTensor, TableTensor


class ToCategorical(Processor):
    """Move text columns into the categorical block.

    This is an stype conversion step: each text column is factorized into
    ordinal category ids over its distinct string values and moved into
    the categorical block, so categorical-capable pipelines can consume
    text features. Factorization is per call; pair with
    :class:`~sdm.processing.CategoricalAlign` to reconcile fit- and
    transform-time categories. Missing values are not representable in
    text columns, so no ``-1`` ids are produced by this step.

    Only ``categorical`` and ``text`` columns are supported. Any other
    semantic type raises an error; drop those columns before this step.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.categorical, Stype.text})

    def _transform(self, table: TableTensor) -> TableTensor:
        """Return ``table`` with text columns moved to ``categorical``."""
        if table.text.size(-1) == 0:
            return table

        codes: list[Tensor] = []
        categories: list[Tensor] = []
        for i in range(table.text.size(-1)):
            column = cast(StringTensor, table.text[..., i])
            if column.is_cuda:
                encoded = CategoricalTensor.from_cudf(
                    ser=column.to_cudf(),
                    device=column.device,
                )
            else:
                encoded = CategoricalTensor.from_arrow(column.to_arrow())
            codes.append(encoded.as_tensor().view(*column.size(), 1))
            categories.append(encoded.categories[0])

        text = CategoricalTensor(
            data=torch.cat(codes, dim=-1),
            categories=tuple(categories),
        )
        columns = (
            *table.columns[Stype.categorical],
            *table.columns[Stype.text],
        )
        categorical = (
            text
            if table.categorical.size(-1) == 0
            else torch.cat(
                (table.categorical, text.to(table.categorical.dtype)),
                dim=-1,
            )
        )
        return table.__class__(
            columns={Stype.categorical: columns},
            categorical=cast(CategoricalTensor, categorical),
        )
