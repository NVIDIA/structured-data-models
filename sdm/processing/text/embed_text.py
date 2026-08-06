from __future__ import annotations

import copy
from typing import Any, cast

import torch
from torch import Tensor

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor


class _SentenceTransformerRef:
    """Hold a SentenceTransformer.

    Shares it on deepcopy and reloads on unpickle.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, strings: list[str]) -> Tensor:
        return self.model.encode(
            strings,
            convert_to_tensor=True,
        )

    @property
    def embedding_dim(self) -> int:
        dim = self.model.get_embedding_dimension()
        assert isinstance(dim, int)
        return dim

    def __deepcopy__(self, _memo: dict[int, Any]) -> _SentenceTransformerRef:
        clone = copy.copy(self)
        clone.model = self.model
        return clone

    def __getstate__(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["model_name"])  # type: ignore[misc]


class EmbedText(Processor):
    r"""Embed text columns with a sentence-transformer model.

    Args:
        model_name: Name of a ``sentence-transformers`` model to load
            from the HuggingFace Hub.
        chunk_size: Maximum number of strings per model call. When set,
            the flattened strings are split into chunks of this size to
            avoid out-of-memory errors on large tables.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.text})

    def __init__(
        self,
        model_name: str,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self._model_ref = _SentenceTransformerRef(model_name)
        self._chunk_size = chunk_size

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.device
        dtype = torch.get_default_dtype()
        col_names = table.columns[Stype.text]
        batch_shape = table.text.shape[:-1]

        out_col_names: list[str] = []
        for col_name in col_names:
            out_col_names.extend(
                f"{col_name}_{i}" for i in range(self._model_ref.embedding_dim)
            )

        numerical = torch.empty(
            (*batch_shape, len(out_col_names)),
            dtype=dtype,
            device=device,
        )
        if numerical.numel() != 0:
            num_cols = len(col_names)
            all_strings: list[Any] = []
            for col in range(num_cols):
                col_tensor = cast(
                    StringTensor,
                    table.text[..., col].reshape(-1),
                )
                all_strings.extend(col_tensor.to_arrow().to_pylist())

            valid_mask = [s is not None for s in all_strings]
            valid_strings = [s for s in all_strings if s is not None]
            chunk_size = self._chunk_size or max(len(valid_strings), 1)

            embed_dim = self._model_ref.embedding_dim
            if valid_strings:
                chunks: list[Tensor] = []
                for start in range(0, len(valid_strings), chunk_size):
                    chunks.append(
                        self._model_ref.encode(
                            valid_strings[start : start + chunk_size],
                        ).to(
                            device=device,
                            dtype=dtype,
                        )
                    )
                valid_embeddings = torch.cat(chunks)
            else:
                valid_embeddings = torch.empty(
                    0, embed_dim, device=device, dtype=dtype,
                )

            all_embeddings = torch.zeros(
                len(all_strings), embed_dim, device=device, dtype=dtype,
            )
            if valid_strings:
                valid_indices = [
                    i for i, m in enumerate(valid_mask) if m
                ]
                all_embeddings[valid_indices] = valid_embeddings
            # (num_cols * batch_numel, embedding_dim)
            numerical = (
                all_embeddings.reshape(
                    num_cols,
                    *batch_shape,
                    self._model_ref.embedding_dim,
                )
                .movedim(0, -2)
                .reshape(*batch_shape, len(out_col_names))
            )

        return TableTensor(
            columns={Stype.numerical: tuple(out_col_names)},
            numerical=numerical,
        )
