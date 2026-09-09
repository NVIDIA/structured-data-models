"""Frozen document embeddings followed by context-only, capped FP32 PCA."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from examples.kumo.relational._relarena.cache import DocumentCache

import sdm
import sdm.processing as sp

QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def _documents(table: sdm.TableTensor) -> list[str]:
    frame = table.flatten(0, table.dim() - 2).to_pandas()
    return [
        "\x1f".join(
            f"{column}={value}"
            for column, value in zip(frame.columns, row, strict=True)
            if not pd.isna(value) and str(value) != ""
        )
        for row in frame.itertuples(index=False, name=None)
    ]


class QwenDocuments(sp.Processor):
    """Encode canonical table-row documents with the frozen V11 text model."""

    requires_fit = False
    handles_stypes = frozenset({sdm.Stype.text})

    def __init__(
        self,
        device: torch.device,
        *,
        cache_path: Path | None = None,
        max_vector_bytes: int = 32 * 1024**3,
    ) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.encoder = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            revision=QWEN_REVISION,
            device=str(device),
            model_kwargs={
                "torch_dtype": torch.float16,
                "attn_implementation": "eager",
            },
        )
        self.encoder.max_seq_length = 128
        self.encoder.tokenizer.padding_side = "right"
        self.encoder.default_prompt_name = None
        self.encoder.eval().requires_grad_(False)
        # Retain the API supported by sentence-transformers 5.x as well.
        self.embedding_dim = cast(
            int,
            self.encoder.get_sentence_embedding_dimension(),  # ty: ignore[deprecated]
        )
        self.cache = None
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache = DocumentCache(
                cache_path,
                identity=f"qwen:{QWEN_REVISION}:fp16:eager:right:128:no-prompt:fp32-l2:column-value-unit-separator-v1",
                dimension=self.embedding_dim,
                max_vector_bytes=max_vector_bytes,
                device=device,
            )

    def __deepcopy__(self, memo: dict[int, Any]) -> QwenDocuments:
        # Frozen encoder has no fitted state; per-estimator PCA is separate.
        return self

    def _encode(self, documents: Sequence[str]) -> torch.Tensor:
        with torch.inference_mode():
            vectors = self.encoder.encode(
                list(documents),
                batch_size=256,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            with torch.amp.autocast(vectors.device.type, enabled=False):
                return torch.nn.functional.normalize(vectors.float(), dim=-1)

    def precompute(self, table: sdm.TableTensor) -> None:
        """Populate every nonempty document, failing if the cache is full."""
        assert self.cache is not None
        self.cache.encode(
            [document for document in _documents(table) if document],
            self._encode,
            require_all=True,
        )

    def _transform(self, table: sdm.TableTensor) -> sdm.TableTensor:
        shape = table.size()[:-1]
        documents = _documents(table)
        codes, unique = pd.factorize(
            np.asarray(documents, dtype=object), sort=False
        )
        nonempty = [index for index, doc in enumerate(unique) if doc]
        bank = torch.zeros(
            (len(unique), self.embedding_dim), device=table.device
        )
        if nonempty:
            documents = [str(unique[index]) for index in nonempty]
            vectors = (
                self._encode(documents)
                if self.cache is None
                else self.cache.encode(documents, self._encode)
            )
            bank[torch.tensor(nonempty, device=table.device)] = vectors
        numerical = bank[torch.as_tensor(codes, device=table.device)].reshape(
            *shape, self.embedding_dim
        )
        return sdm.TableTensor(
            columns={
                sdm.Stype.numerical: [
                    f"text_{index}" for index in range(self.embedding_dim)
                ]
            },
            numerical=numerical,
        )


class ContextPCA(sp.PCA):
    """V11 PCA: nonempty context only, fixed signs and zero missing text."""

    def _fit(
        self,
        table: sdm.TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        with torch.amp.autocast(table.device.type, enabled=False):
            numerical = table.numerical
            if numerical.dtype != torch.float32:
                raise RuntimeError("Text PCA requires FP32 embeddings")
            eligible = numerical.norm(dim=-1).ne(0).nonzero().flatten().cpu()
            if not len(eligible):
                raise RuntimeError(
                    "Text table has no nonempty context documents"
                )
            if len(eligible) > 50_000:
                selected = torch.randperm(
                    len(eligible),
                    generator=torch.Generator(device="cpu").manual_seed(0),
                )[:50_000]
                eligible = eligible[selected]
            fit = numerical.index_select(-2, eligible.to(numerical.device))
            self.mean = fit.mean(dim=-2, keepdim=True)
            _, _, vh = torch.linalg.svd(fit - self.mean, full_matrices=False)
            count = min(self.num_components, *fit.shape[-2:])
            components = vh[:count].T.contiguous()
            if count < self.num_components:
                components = torch.nn.functional.pad(
                    components, (0, self.num_components - count)
                )
            largest = components.abs().argmax(dim=0)
            signs = components[
                largest,
                torch.arange(self.num_components, device=components.device),
            ].sign()
            signs[signs == 0] = 1
            self.components = components * signs

    def _transform(self, table: sdm.TableTensor) -> sdm.TableTensor:
        with torch.amp.autocast(table.device.type, enabled=False):
            numerical = table.numerical
            if numerical.dtype != torch.float32:
                raise RuntimeError("Text PCA requires FP32 embeddings")
            output = (numerical - self.mean) @ self.components
            empty = numerical.norm(dim=-1).eq(0)
            output = output.masked_fill(empty.unsqueeze(-1), 0)
            return sdm.TableTensor(
                columns={
                    sdm.Stype.numerical: [
                        f"__text_pca_{index}__"
                        for index in range(self.num_components)
                    ]
                },
                numerical=output,
            )
