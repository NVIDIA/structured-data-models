from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

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

    Args:
        embedding_model: Pre-loaded model called on the flattened text values.
            It must return a :class:`torch.Tensor` with shape
            ``[n, embedding_dim]``.
        embedding_dim: Width of each returned embedding.
        chunk_size: Maximum number of strings per model call. When set,
            the flattened strings are split into chunks of this size to
            avoid out-of-memory errors on large tables.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.text})

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        embedding_dim: int,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self._embedding_model = _ModuleReference(embedding_model)
        self._embedding_dim: int = embedding_dim
        self._chunk_size = chunk_size

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

        numerical = torch.empty(
            (*batch_shape, len(out_col_names)),
            dtype=dtype,
            device=device,
        )
        if numerical.numel() != 0:
            # FIXME: The embedding_model currently must take in a
            # dataframe and not a Tensor.
            num_cols = len(col_names)
            all_cols: list[Tensor] = [
                table.text[..., i].reshape(-1) for i in range(num_cols)
            ]
            flat_strings = cast(StringTensor, torch.cat(all_cols))
            chunk_size = self._chunk_size or len(flat_strings)

            chunks: list[Tensor] = []
            for start in range(0, len(flat_strings), chunk_size):
                chunk = cast(
                    StringTensor,
                    flat_strings[start : start + chunk_size],
                )
                strings = (
                    chunk.to_cudf() if chunk.is_cuda else chunk.to_arrow()
                )
                chunks.append(
                    self._embedding_model(strings).to(
                        device=device,
                        dtype=dtype,
                    )
                )
            all_embeddings = torch.cat(chunks)
            # (num_cols * batch_numel, embedding_dim)
            numerical = (
                all_embeddings.reshape(
                    num_cols,
                    *batch_shape,
                    self._embedding_dim,
                )
                .movedim(0, -2)
                .reshape(*batch_shape, len(out_col_names))
            )

        return TableTensor(
            columns={Stype.numerical: tuple(out_col_names)},
            numerical=numerical,
        )
