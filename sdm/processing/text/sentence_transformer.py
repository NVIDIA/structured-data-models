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


class _WordPieceTokenizer:
    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        vocabulary: WordPieceVocabulary,
        normalizer: CharacterNormalizer,
        cls_token_id: int,
        sep_token_id: int,
        pad_token_id: int,
        max_length: int,
    ) -> None:
        self._model = model
        self._vocabulary = vocabulary
        self._normalizer = normalizer
        self._cls_token_id = cls_token_id
        self._sep_token_id = sep_token_id
        self._pad_token_id = pad_token_id
        self._max_length = max_length

    def __deepcopy__(self, memo: dict[int, Any]) -> _WordPieceTokenizer:
        return self

    @classmethod
    def build(
        cls,
        model: sentence_transformers.SentenceTransformer,
    ) -> _WordPieceTokenizer | None:

        from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

        tokenizer = model.tokenizer
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            return None
        if tokenizer.backend_tokenizer.model.__class__.__name__ != "WordPiece":
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
            range(tokenizer.vocab_size)
        )
        return cls(
            model=model,
            vocabulary=WordPieceVocabulary(cudf.Series(vocab_tokens)),
            normalizer=CharacterNormalizer(
                do_lower=tokenizer.do_lower_case,
                special_tokens=cudf.Series(tokenizer.all_special_tokens),
            ),
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=model.max_seq_length or tokenizer.model_max_length,
        )

    def tokenize(self, text: StringTensor) -> tuple[Tensor, Tensor]:
        device = text.device
        num_strings = text.numel()

        text_series = text.to_cudf()
        if text.is_nullable:
            text_series = text_series.fillna("")

        normalized = self._normalizer.normalize(text_series)
        token_lists = self._vocabulary.tokenize(normalized)

        flat_values = torch.from_dlpack(token_lists.list.leaves.to_cupy())
        raw_lengths = torch.from_dlpack(token_lists.list.len().to_cupy()).to(
            torch.long
        )

        # Source offsets into the flat token buffer
        offsets = torch.zeros(num_strings + 1, device=device, dtype=torch.long)
        torch.cumsum(raw_lengths, dim=0, out=offsets[1:])

        lengths = raw_lengths.clamp(max=self._max_length - 2)

        input_ids = torch.full(
            (num_strings, self._max_length),
            self._pad_token_id,
            device=device,
            dtype=torch.int32,
        )
        input_ids[:, 0] = self._cls_token_id

        if flat_values.numel() > 0:
            max_content = self._max_length - 2
            col_idx = torch.arange(max_content, device=device)
            mask = col_idx.unsqueeze(0) < lengths.unsqueeze(1)
            src = col_idx.unsqueeze(0) + offsets[:-1].unsqueeze(1)
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


class _Encoder:
    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        batch_size: int,
        embedding_dim: int,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim
        self._tokenizer: _WordPieceTokenizer | None = None
        self._gpu_resolved = False

    def __deepcopy__(self, memo: dict[int, Any]) -> _Encoder:
        return self

    def _encode_gpu(self, text: StringTensor) -> Tensor:
        assert self._tokenizer is not None
        device = text.device
        num_strings = text.numel()
        input_ids, attention_mask = self._tokenizer.tokenize(text)

        seq_lengths = attention_mask.sum(dim=1)
        sort_idx = seq_lengths.argsort()
        sorted_input_ids = input_ids[sort_idx]
        sorted_attention_mask = attention_mask[sort_idx]
        sorted_seq_lengths = seq_lengths[sort_idx]

        batch_ends = (
            torch.arange(
                self._batch_size,
                num_strings + self._batch_size,
                self._batch_size,
                device=device,
            ).clamp(max=num_strings)
            - 1
        )
        batch_max_lengths = sorted_seq_lengths[batch_ends].tolist()

        embeddings = torch.empty(
            num_strings,
            self._embedding_dim,
            device=device,
            dtype=torch.float,
        )
        for i, batch_start in enumerate(
            range(0, num_strings, self._batch_size)
        ):
            batch_end = min(batch_start + self._batch_size, num_strings)
            batch_seq_len = batch_max_lengths[i]
            features: dict[str, Tensor] = {
                "input_ids": sorted_input_ids[
                    batch_start:batch_end, :batch_seq_len
                ],
                "attention_mask": sorted_attention_mask[
                    batch_start:batch_end, :batch_seq_len
                ],
            }
            features = self._model(features)
            embeddings[sort_idx[batch_start:batch_end]] = features[
                "sentence_embedding"
            ]

        return embeddings

    def encode(self, text: StringTensor) -> Tensor:
        """Encode text into embeddings.

        On the first call, lazily checks whether GPU WordPiece tokenization
        is available. Uses the GPU path if so, otherwise falls back to
        CPU-based :meth:`SentenceTransformer.encode`.

        Args:
            text: Flat :class:`~sdm.StringTensor`.
        """
        if not self._gpu_resolved:
            self._gpu_resolved = True
            if text.device.type == "cuda":
                self._tokenizer = _WordPieceTokenizer.build(self._model)
                if self._tokenizer is not None:
                    self._model.eval()

        if self._tokenizer is not None:
            return self._encode_gpu(text)

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


class _ModuleReference(torch.nn.Module):
    def __init__(
        self,
        module: sentence_transformers.SentenceTransformer,
    ) -> None:
        super().__init__()
        self.module = module

    def __deepcopy__(self, memo: dict[int, Any]) -> _ModuleReference:
        return type(self)(self.module)


class SentenceTransformer(Processor):
    r"""Transform text columns with a
    :class:`sentence_transformers.SentenceTransformer
    <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
    model.

    Args:
        model_name: Model name or local path passed to
            :class:`sentence_transformers.SentenceTransformer
            <sentence_transformers.sentence_transformer.model.SentenceTransformer>`.
        batch_size: Batch size passed to
            :meth:`sentence_transformers.SentenceTransformer.encode()
            <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`.
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
        import sentence_transformers  # noqa: PLC0415

        self.batch_size = batch_size
        model = sentence_transformers.SentenceTransformer(model_name)
        embedding_dim = model.get_embedding_dimension()
        assert isinstance(embedding_dim, int)
        self._embedding_dim = embedding_dim
        self._model = _ModuleReference(model)
        self._encoder = _Encoder(model, batch_size, embedding_dim)

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
            emb = self._encoder.encode(text)
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
