from __future__ import annotations

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor


class EmbedText(Processor):
    r"""Replace text columns with numerical embeddings.

    ``encoder`` receives the complete :class:`~sdm.tensor.StringTensor` text
    block with shape ``[..., C]`` and must return a floating-point tensor on
    the same device with shape ``[..., C, D]``. Output columns are named
    ``"{column}__{channel}"``.

    This processor does not download or fit an encoder. The caller supplies a
    ready-to-use module in evaluation mode, which keeps text model and
    vocabulary choices outside the core package.

    Deep copies of this stateless processor share the encoder. This avoids
    duplicating encoder weights when recipes are copied for estimators and
    related tables; the copies consequently share its device and training
    state.

    Args:
        encoder: Module that maps one text column to numerical embeddings.
    """

    supported_stypes = frozenset({Stype.text})
    requires_fit = False

    def __init__(self, encoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def __deepcopy__(self, memo: dict[int, object]) -> EmbedText:
        out = self.__class__(self.encoder)
        out.training = self.training
        memo[id(self)] = out
        return out

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.text.size(-1) == 0:
            return table

        output = self.encoder(table.text)
        expected_size = (*table.text.size(), -1)
        if (
            not isinstance(output, Tensor)
            or output.dim() != table.text.dim() + 1
            or output.size()[:-1] != table.text.size()
            or output.size(-1) == 0
            or not output.is_floating_point()
            or output.device != table.text.device
        ):
            actual = (
                f"{tuple(output.size())} with dtype {output.dtype} "
                f"on device {output.device}"
                if isinstance(output, Tensor)
                else type(output).__name__
            )
            raise ValueError(
                f"Expected 'encoder' output with shape {expected_size} "
                f"and a floating-point dtype on device {table.text.device} "
                f"(got {actual})"
            )

        return TableTensor(
            columns={
                Stype.numerical: tuple(
                    f"{column}__{channel}"
                    for column in table.columns[Stype.text]
                    for channel in range(output.size(-1))
                )
            },
            numerical=output.flatten(-2, -1),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        encoder = repr(self.encoder).replace("\n", f"\n{' ' * (indent + 2)}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{' ' * (indent + 2)}{encoder},\n"
            f"{' ' * indent})"
        )
