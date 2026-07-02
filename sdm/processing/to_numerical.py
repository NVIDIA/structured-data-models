import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor


class ToNumerical(Processor):
    """Move supported table stypes into the numerical block.

    This is an stype conversion step, not a category encoder:
    :class:`~sdm.tensor.CategoricalTensor` already stores ordinal category ids
    and the category vocabulary. ``ToNumerical`` reuses those existing ids,
    casts them to the numerical dtype, and clears the categorical block so
    numerical-only models can consume both original numerical and categorical
    features. Missing categorical values remain ``-1``.

    The current implementation converts all categorical columns because
    ``Pipeline`` routing is table-level here. Future stype-aware dispatch can
    add source-stype selection without changing the model-facing output
    contract. Column order follows the current ``TableTensor`` block order:
    numerical columns first, then converted categorical columns.

    Args:
        dtype: Optional dtype for the returned numerical block. If omitted, the
            input numerical block dtype is used.
    """

    requires_fit = False
    input_scope = "table"

    def __init__(self, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, input: Tensor) -> Tensor:
        """Return ``input`` with all feature columns in ``numerical``."""
        if not isinstance(input, TableTensor):
            raise TypeError(
                "Expected ToNumerical input to be a TableTensor "
                f"(got '{type(input).__name__}')"
            )

        dtype = input.numerical.dtype if self.dtype is None else self.dtype

        blocks: list[Tensor] = []
        columns: list[str] = []
        if input.numerical.size(-1) > 0:
            numerical = input.numerical
            if numerical.dtype != dtype:
                numerical = numerical.to(dtype)
            blocks.append(numerical)
            columns.extend(input.columns[Stype.numerical])

        if input.categorical.size(-1) > 0:
            blocks.append(input.categorical.as_tensor().to(dtype))
            columns.extend(input.columns[Stype.categorical])

        if not blocks:
            return input
        if len(blocks) == 1 and blocks[0] is input.numerical:
            return input

        numerical = (
            blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
        )
        return input.__class__(
            columns={Stype.numerical: tuple(columns)},
            numerical=numerical,
        )
