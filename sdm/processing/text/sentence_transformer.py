# ruff: noqa: D205

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, cast

import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm._warnings import warn_once
from sdm.processing import Processor

if TYPE_CHECKING:
    import sentence_transformers
    from cudf.core.character_normalizer import CharacterNormalizer
    from cudf.core.wordpiece_tokenize import WordPieceVocabulary
    from transformers import PreTrainedTokenizerBase


class _CuDFTokenizer:
    def __init__(
        self,
        vocabulary: WordPieceVocabulary,
        normalizer: CharacterNormalizer,
        cls_token_id: int,
        sep_token_id: int,
        pad_token_id: int,
        max_length: int,
    ) -> None:
        self._vocabulary = vocabulary
        self._normalizer = normalizer
        self._cls_token_id = cls_token_id
        self._sep_token_id = sep_token_id
        self._pad_token_id = pad_token_id
        self._max_length = max_length

    @classmethod
    def build(
        cls,
        tokenizer: PreTrainedTokenizerBase,
    ) -> _CuDFTokenizer | None:
        import tokenizers
        from transformers import PreTrainedTokenizerFast

        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            return None
        backend = tokenizer.backend_tokenizer
        if not isinstance(backend.model, tokenizers.models.WordPiece):
            return None
        if importlib.util.find_spec("cudf") is None:
            warn_once(
                key="on-device-tokenization-available-but-cudf-unavailable",
                message=(
                    "cuDF supports accelerating tokenization of the specified "
                    "model's tokenizer. However, cuDF is not installed. "
                    "To enable on-device tokenization, install cuDF."
                ),
            )
            return None

        import cudf
        from cudf.core.character_normalizer import CharacterNormalizer
        from cudf.core.wordpiece_tokenize import WordPieceVocabulary

        vocab_tokens = tokenizer.convert_ids_to_tokens(
            list(range(tokenizer.vocab_size))
        )
        return cls(
            vocabulary=WordPieceVocabulary(cudf.Series(vocab_tokens)),
            normalizer=CharacterNormalizer(
                do_lower=tokenizer.do_lower_case,
                special_tokens=cudf.Series(tokenizer.all_special_tokens),
            ),
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=tokenizer.model_max_length,
        )

    def tokenize(self, text: StringTensor) -> tuple[Tensor, Tensor]:
        device = text.device
        if device.type != "cuda":
            raise RuntimeError(
                f"Non-CPU tensor is passed to cuDF tokenizer: {device}"
            )

        ser = text.to_cudf()
        ser = ser.fillna("") if text.is_nullable else ser
        ser = self._normalizer.normalize(ser)
        ser = self._vocabulary.tokenize(ser)

        flat_values = torch.from_dlpack(ser.list.leaves.to_cupy())
        lengths = torch.from_dlpack(ser.list.len().to_cupy())

        num_strings = text.numel()
        offsets = torch.zeros(
            num_strings + 1,
            device=device,
            dtype=torch.int32,
        )
        torch.cumsum(lengths, dim=0, out=offsets[1:])
        lengths.clamp_(max=self._max_length - 2)
        input_ids = torch.full(
            (num_strings, self._max_length),
            self._pad_token_id,
            device=device,
            dtype=torch.int32,
        )
        input_ids[:, 0] = self._cls_token_id

        if flat_values.numel() > 0:
            max_content = self._max_length - 2
            col_idx = torch.arange(
                max_content,
                device=device,
                dtype=torch.int32,
            ).unsqueeze(0)
            mask = col_idx < lengths.unsqueeze(1)
            src = col_idx + offsets[:-1].unsqueeze(1)
            safe_src = torch.where(mask, src, torch.zeros_like(src))
            input_ids[:, 1 : max_content + 1] = torch.where(
                mask,
                flat_values[safe_src],
                input_ids[:, 1 : max_content + 1],
            )

        input_ids[torch.arange(num_strings, device=device), lengths + 1] = (
            self._sep_token_id
        )
        attention_mask = (input_ids != self._pad_token_id).to(torch.int32)

        return input_ids, attention_mask


class _Model(torch.nn.Module):
    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        batch_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self._model = model
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim
        self._cudf_tokenizer: _CuDFTokenizer | None = None
        self._cudf_tokenizer_supported: bool | None = None

    def __deepcopy__(self, memo: dict[int, Any]) -> _Model:
        return self

    def _forward_cpu(self, text: StringTensor) -> Tensor:
        array = text.to_arrow()
        if text.is_nullable:
            array = pc.fill_null(array, "")
        emb = self._model.encode(
            array.to_pylist(),
            show_progress_bar=False,
            convert_to_tensor=True,
            device=str(text.device),
            batch_size=self._batch_size,
        )
        assert isinstance(emb, Tensor)
        return emb

    def _forward_cudf(self, text: StringTensor) -> Tensor:
        assert self._cudf_tokenizer is not None
        input_ids, attention_mask = self._cudf_tokenizer.tokenize(text)

        seq_lengths = attention_mask.sum(dim=1)
        sort_idx = seq_lengths.argsort()
        sorted_input_ids = input_ids[sort_idx]
        sorted_attention_mask = attention_mask[sort_idx]
        sorted_seq_lengths = seq_lengths[sort_idx]

        device = text.device
        num_strings = text.numel()
        batch_ends = torch.arange(
            self._batch_size,
            num_strings + self._batch_size,
            self._batch_size,
            device=device,
        )
        batch_ends.clamp_(max=num_strings)
        batch_ends -= 1
        # Triggers a host device sync to get chunk sizes to
        # minimize the padding
        batch_max_lengths = sorted_seq_lengths[batch_ends].tolist()

        embeddings = torch.empty(
            num_strings,
            self._embedding_dim,
            device=device,
            dtype=torch.float32,
        )
        for i, start in enumerate(range(0, num_strings, self._batch_size)):
            end = min(start + self._batch_size, num_strings)
            max_len = batch_max_lengths[i]
            features: dict[str, Tensor] = {
                "input_ids": sorted_input_ids[start:end, :max_len],
                "attention_mask": sorted_attention_mask[start:end, :max_len],
            }
            features = self._model(features)
            embeddings[sort_idx[start:end]] = features["sentence_embedding"]

        return embeddings

    def forward(self, text: StringTensor) -> Tensor:
        if text.device.type == "cpu":
            return self._forward_cpu(text)

        if self._cudf_tokenizer_supported is None:
            self._cudf_tokenizer = _CuDFTokenizer.build(self._model.tokenizer)
            self._cudf_tokenizer_supported = self._cudf_tokenizer is not None

        if self._cudf_tokenizer_supported:
            return self._forward_cudf(text)

        return self._forward_cpu(text)


class SentenceTransformer(Processor):
    r"""Transform text columns with a
    :class:`sentence_transformers.SentenceTransformer` model.

    Args:
        model_name: Model name or local path passed to
            :class:`sentence_transformers.SentenceTransformer`
        batch_size: Batch size passed to
            :meth:`sentence_transformers.SentenceTransformer.encode`.
            Adjusting the batch size can significantly improve processing
            speed. The optimal value depends on your hardware, model size,
            precision, and input length.
    """

    requires_fit = False
    handles_stypes = frozenset({Stype.text})

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        import sentence_transformers

        self.batch_size = batch_size
        model = sentence_transformers.SentenceTransformer(model_name)
        embedding_dim = model.get_embedding_dimension()
        assert isinstance(embedding_dim, int)
        self._embedding_dim = embedding_dim
        self._model = _Model(model, batch_size, embedding_dim)

    def _transform(self, table: TableTensor) -> TableTensor:
        columns = table.columns[Stype.text]
        batch_shape = table.text.shape[:-1]
        output_columns = tuple(
            f"{column}__emb{i}"
            for column in columns
            for i in range(self._embedding_dim)
        )

        if table.text.numel() == 0:
            numerical = torch.empty(
                (*batch_shape, len(output_columns)),
                dtype=table.dtype,
                device=table.device,
            )
        else:
            text = cast(StringTensor, table.text.movedim(-1, 0).reshape(-1))
            emb = self._model(text)
            emb = emb.to(device=table.device, dtype=table.dtype)
            numerical = (
                emb.reshape(len(columns), *batch_shape, self._embedding_dim)
                .movedim(0, -2)
                .reshape(*batch_shape, len(output_columns))
            )

        out = torch.cat(
            [
                table.drop_stypes(Stype.text),
                TableTensor(
                    columns={Stype.numerical: output_columns},
                    numerical=numerical,
                ),
            ],
            dim=-1,
        )
        return cast(TableTensor, out)
