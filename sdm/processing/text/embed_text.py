from __future__ import annotations

from typing import Any, cast

import torch

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor


class _ModuleReference(torch.nn.Module):
    """Preserve a module reference across deep copies."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def __deepcopy__(self, _memo: dict[int, Any]) -> _ModuleReference:
        return type(self)(self.module)


class EmbedText(Processor):
    r"""Embed text columns with a user-provided embedding model.

    Each text column is embedded cell-by-cell through ``embedding_model``.
    The model must return one embedding per text value as a
    :class:`torch.Tensor` with shape ``[n, embedding_dim]``. Embedding
    Embeddings are concatenated in column order into the numerical output.

    Args:
        embedding_model: Pre-loaded model called on each flattened text column.
            It must return a :class:`torch.Tensor` with shape
            ``[n, embedding_dim]``.
        embedding_dim: Width of each returned embedding.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.text})

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self._embedding_model = _ModuleReference(embedding_model)
        self._embedding_dim: int = embedding_dim

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.device
        dtype = torch.get_default_dtype()
        col_names = table.columns[Stype.text]
        batch_shape = table.text.shape[:-1]
        out_col_names: list[str] = []
        for col_name in col_names:
            out_col_names.extend(
                f"{col_name}_{i}" for i in range(self._embedding_dim)
            )

        if batch_shape.numel() == 0:
            numerical = torch.zeros(
                (*batch_shape, len(out_col_names)),
                dtype=dtype,
                device=device,
            )
        else:
            numerical = torch.empty(
                (*batch_shape, len(out_col_names)),
                dtype=dtype,
                device=device,
            )
            # FIXME: There're currently multiple issues:
            # 1. Even though the same model gets applied to all columns, we run
            #    it once per column.
            # 2. The embedding_model currently must take in a dataframe and not
            #    a Tensor.
            for col_idx in range(len(col_names)):
                col_tensor = cast(
                    StringTensor,
                    table.text[..., col_idx].reshape(-1),
                )
                strings = (
                    col_tensor.to_cudf()
                    if col_tensor.is_cuda
                    else col_tensor.to_arrow()
                )
                block = self._embedding_model(strings)
                embeddings = block.to(device=device, dtype=dtype).reshape(
                    *batch_shape,
                    self._embedding_dim,
                )
                start_idx = col_idx * self._embedding_dim
                end_idx = start_idx + self._embedding_dim
                numerical[..., start_idx:end_idx] = embeddings

        return TableTensor(
            columns={Stype.numerical: tuple(out_col_names)},
            numerical=numerical,
        )
