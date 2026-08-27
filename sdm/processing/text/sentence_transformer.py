# ruff: noqa: D205

from __future__ import annotations

import importlib.util
from functools import cached_property
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


# Hugging Face treats words with more than 100 Unicode characters as [UNK],
# while cuDF does so for words with 200 or more UTF-8 bytes.
_CUDF_WORDPIECE_MAX_BYTES = 200


class _CuDFTokenizer:
    def __init__(
        self,
        vocabulary: WordPieceVocabulary,
        normalizer: CharacterNormalizer,
        cls_token_id: int,
        sep_token_id: int,
        pad_token_id: int,
        max_length: int,
        max_input_chars_per_word: int,
    ) -> None:
        self._vocabulary = vocabulary
        self._normalizer = normalizer
        self._cls_token_id = cls_token_id
        self._sep_token_id = sep_token_id
        self._pad_token_id = pad_token_id
        self._max_length = max_length
        self._max_input_chars_per_word = max_input_chars_per_word

    @staticmethod
    def _is_supported(
        model: sentence_transformers.SentenceTransformer,
    ) -> bool:
        import tokenizers
        from sentence_transformers.sentence_transformer.modules import (
            Transformer,
        )
        from transformers import PreTrainedTokenizerFast

        module = model[0]
        tokenizer = model.tokenizer

        # Fall back to CPU tokenization unless this is a standard BERT
        # Transformer module with a fast tokenizer.
        if (
            type(module) is not Transformer
            or module.config.model_type != "bert"
            or not isinstance(tokenizer, PreTrainedTokenizerFast)
        ):
            return False

        if (
            model.modalities != ["text"]
            or model.default_prompt_name is not None
            or model.truncate_dim is not None
            or module.transformer_task != "feature-extraction"
            or module.backend != "torch"
            or module.processing_kwargs
            or module.can_flatten_inputs
            or module.unpad_inputs
        ):
            return False

        backend = tokenizer.backend_tokenizer
        wordpiece = backend.model
        normalizer = backend.normalizer

        # cuDF cannot currently construct an equivalent tokenizer directly
        # from the Hugging Face tokenizer configuration. Keep these checks
        # until it can.
        # Require the standard BERT WordPiece pipeline.
        if (
            type(wordpiece) is not tokenizers.models.WordPiece
            or type(normalizer) is not tokenizers.normalizers.BertNormalizer
            or type(backend.pre_tokenizer)
            is not tokenizers.pre_tokenizers.BertPreTokenizer
            or type(backend.post_processor)
            is not tokenizers.processors.TemplateProcessing
            or (
                normalizer.clean_text,
                normalizer.handle_chinese_chars,
                normalizer.strip_accents,
            )
            != (True, True, None)
            or (wordpiece.unk_token, wordpiece.continuing_subword_prefix)
            != ("[UNK]", "##")
            or (tokenizer.padding_side, tokenizer.truncation_side)
            != ("right", "right")
            or tokenizer.model_max_length != module.max_seq_length
        ):
            return False

        vocab = backend.get_vocab(with_added_tokens=False)
        added_tokens = tuple(backend.get_added_tokens_decoder().values())
        special_tokens = (
            (tokenizer.cls_token, tokenizer.cls_token_id),
            (tokenizer.sep_token, tokenizer.sep_token_id),
            (tokenizer.pad_token, tokenizer.pad_token_id),
            (tokenizer.unk_token, tokenizer.unk_token_id),
        )

        # Vocabulary IDs and added-token behavior must match cuDF's lookup.
        if (
            backend.get_vocab_size(with_added_tokens=True) != len(vocab)
            or set(vocab.values()) != set(range(len(vocab)))
            or not all(
                vocab.get(token) == token_id
                for token, token_id in special_tokens
            )
            or {token.content for token in added_tokens}
            != set(tokenizer.all_special_tokens)
            or not all(
                token.special
                and not token.normalized
                and not token.lstrip
                and not token.rstrip
                and not token.single_word
                for token in added_tokens
            )
        ):
            return False

        encoded = backend.encode(tokenizer.unk_token, add_special_tokens=True)
        return (
            backend.encode(
                tokenizer.unk_token,
                add_special_tokens=False,
            ).ids
            == [tokenizer.unk_token_id]
            and encoded.ids
            == [
                tokenizer.cls_token_id,
                tokenizer.unk_token_id,
                tokenizer.sep_token_id,
            ]
            and not any(encoded.type_ids)
        )

    @classmethod
    def build(
        cls,
        model: sentence_transformers.SentenceTransformer,
    ) -> _CuDFTokenizer | None:
        if not cls._is_supported(model):
            return None

        tokenizer = model.tokenizer
        backend = tokenizer.backend_tokenizer
        wordpiece = backend.model
        normalizer = backend.normalizer
        vocab = backend.get_vocab(with_added_tokens=False)

        if importlib.util.find_spec("cudf") is None:
            warn_once(
                key="on-device-tokenization-available-but-cudf-unavailable",
                message=(
                    "Falling back to CPU-based tokenization because cuDF is "
                    "not installed. Install cuDF to enable faster CUDA-based "
                    "tokenization without device synchronization."
                ),
            )
            return None

        import cudf
        from cudf.core.character_normalizer import CharacterNormalizer
        from cudf.core.wordpiece_tokenize import WordPieceVocabulary

        vocab_tokens = [
            token for token, _ in sorted(vocab.items(), key=lambda x: x[1])
        ]
        return cls(
            vocabulary=WordPieceVocabulary(cudf.Series(vocab_tokens)),
            normalizer=CharacterNormalizer(
                do_lower=normalizer.lowercase,
                special_tokens=cudf.Series(tokenizer.all_special_tokens),
            ),
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=tokenizer.model_max_length,
            max_input_chars_per_word=wordpiece.max_input_chars_per_word,
        )

    def tokenize(self, text: StringTensor) -> tuple[Tensor, Tensor]:
        device = text.device
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
        attention_mask = (
            torch.arange(self._max_length, device=device).unsqueeze(0)
            < (lengths + 2).unsqueeze(1)
        ).to(torch.int32)

        return input_ids, attention_mask


class _Encoder(torch.nn.Module):
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

    @cached_property
    def _cudf_tokenizer(self) -> _CuDFTokenizer | None:
        return _CuDFTokenizer.build(self._model)

    def __deepcopy__(self, memo: dict[int, Any]) -> _Encoder:
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
        self._model.eval()
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

        tokenizer = self._cudf_tokenizer
        if tokenizer is None:
            return self._forward_cpu(text)

        series = text.to_cudf()
        series = series.fillna("") if text.is_nullable else series
        words = tokenizer._normalizer.normalize(series).str.tokenize()

        # CPU rejects long words by Unicode characters, while cuDF uses UTF-8
        # bytes. The following lines find rows where those decisions differ.
        # Remove this fallback if cuDF exposes the tokenizer's
        # max_input_chars_per_word setting.
        use_cpu = (words.str.len() > tokenizer._max_input_chars_per_word) != (
            words.str.byte_count() >= _CUDF_WORDPIECE_MAX_BYTES
        )
        cpu_indices = torch.from_dlpack(words[use_cpu].index.unique().values)

        num_strings = text.numel()
        if cpu_indices.numel() == 0:
            return self._forward_cudf(text)
        if cpu_indices.numel() == num_strings:
            return self._forward_cpu(text)

        use_cudf = torch.ones(
            num_strings,
            device=text.device,
            dtype=torch.bool,
        )
        use_cudf[cpu_indices] = False
        cudf_indices = use_cudf.nonzero().flatten()

        # Tokenize each subset with the matching path, then restore the
        # original input order through indexed assignment.
        embeddings = torch.empty(
            num_strings,
            self._embedding_dim,
            device=text.device,
            dtype=torch.float32,
        )
        cudf_text = cast(StringTensor, text[cudf_indices])
        cpu_text = cast(StringTensor, text[cpu_indices])
        embeddings[cudf_indices] = self._forward_cudf(cudf_text)
        embeddings[cpu_indices] = self._forward_cpu(cpu_text)
        return embeddings


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
        self._model = _Encoder(model, batch_size, embedding_dim)

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
