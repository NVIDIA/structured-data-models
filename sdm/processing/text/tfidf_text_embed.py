import math
import re
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor
from sdm.tensor.io import arrow_as_tensor


class TfidfTextEmbed(Processor):
    r"""Encode text columns as character n-gram TF-IDF vectors.

    Each text column is tokenized into word-boundary character n-grams, and a
    separate vocabulary and inverse-document-frequency (idf) weighting is
    fitted per
    column on the context table. Every column expands to a block of numerical
    features (one per fitted n-gram), and the blocks are concatenated into the
    numerical output. The idf smoothing matches scikit-learn's
    smoothing formula (`idf(t) = ln( (1 + n_docs) / (1 + df(t)) ) + 1`)
    default and rows are L2-normalized.

    Args:
        ngram_range: Inclusive ``(min_n, max_n)`` character-window sizes.
        max_features: If set, keep only this many most frequent n-grams per
            column. ``None`` keeps the full vocabulary.
        lowercase: Lowercase each string before tokenizing.
    """

    supported_stypes = frozenset({Stype.text})

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int],
        max_features: int | None = None,
        lowercase: bool = True,
    ) -> None:
        super().__init__()
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.lowercase = lowercase
        self._vocabularies: list[pa.Array] = []
        self._register_load_state_dict_pre_hook(self._recreate_idf_buffers)

        min_n, max_n = self.ngram_range
        if min_n < 1 or max_n < min_n:
            raise ValueError("'ngram_range' must satisfy 1 <= min_n <= max_n.")

        if max_features is not None and max_features < 0:
            raise ValueError("`max_features` must be non-negative or None.")

    def get_extra_state(self) -> dict[str, Any]:
        r""":meta private:"""  # noqa: D415
        """Package the fitted state for :meth:`~torch.nn.Module.state_dict`.

        The fitted state lives outside PyTorch's parameter/buffer registries
        (``self._vocabularies``),
        so it is exported here instead. Vocabularies are
        stored as plain ``(data, offset)`` tensor pairs to keep checkpoints
        loadable under ``torch.load(weights_only=True)``.
        """
        return {
            "vocabularies": [
                StringTensor.from_arrow(vocabulary).data_offset
                for vocabulary in self._vocabularies
            ],
            "fitted": self._fitted,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        r""":meta private:"""  # noqa: D415
        """Restore the fitted state from a checkpoint."""
        self._vocabularies = [
            StringTensor(
                data=data,
                offset=offset,
                size=(offset.numel() - 1,),
            ).to_arrow()
            for data, offset in state["vocabularies"]
        ]
        self._fitted = state["fitted"]

    def _recreate_idf_buffers(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """Restore buffers from a checkpoint."""
        stale_idfs = [
            name for name in self._buffers if name.startswith("idf_")
        ]
        for name in stale_idfs:
            delattr(self, name)

        idf_keys = [
            key for key in state_dict if key.startswith(f"{prefix}idf_")
        ]
        for key in idf_keys:
            self.register_buffer(key[len(prefix) :], state_dict[key])

    def _character_ngrams(
        self,
        tensor: StringTensor,
        ngram_range: tuple[int, int],
        *,
        lowercase: bool = True,
    ) -> tuple[StringTensor, Tensor]:
        r"""Split each string into word-boundary character n-grams.

        Each whitespace-delimited word is padded with a single space on
        both sides before windowing, so a word shorter than ``n`` still
        yields one n-gram.

        Returns a flat :class:`StringTensor` holding every n-gram of every
        document, together with an ``offset`` tensor of length ``numel() + 1``
        where document ``d``'s n-grams are ``flat[offset[d]:offset[d + 1]]``.

        Args:
            tensor: StringTensor.
            ngram_range: Inclusive ``(min_n, max_n)`` window sizes.
            lowercase: Lowercase each string before windowing.
        """
        if tensor.dim() != 1:
            raise NotImplementedError(
                "'character_ngrams' only supports one-dimensional input"
            )

        if tensor.is_cuda:
            return self._character_ngrams_cuda(tensor, ngram_range, lowercase)

        min_n, max_n = ngram_range
        whitespace = re.compile(r"\s\s+")
        ngrams: list[str] = []
        offsets: list[int] = [0]
        for document in tensor.to_arrow().to_pylist():
            if lowercase:
                document = document.lower()
            document = whitespace.sub(" ", document)
            for word in document.split():
                word = " " + word + " "
                word_len = len(word)
                for n in range(min_n, max_n + 1):
                    offset = 0
                    ngrams.append(word[offset : offset + n])
                    while offset + n < word_len:
                        offset += 1
                        ngrams.append(word[offset : offset + n])
                    if offset == 0:  # word shorter than n: count it once
                        break
            offsets.append(len(ngrams))

        flat = tensor.from_arrow(
            pa.array(ngrams, type=pa.large_string()),
            device=tensor.device,
        )
        offset = torch.tensor(offsets, dtype=torch.int64, device=tensor.device)
        return flat, offset

    def _character_ngrams_cuda(
        self,
        tensor: StringTensor,
        ngram_range: tuple[int, int],
        lowercase: bool = True,
    ) -> tuple[StringTensor, Tensor]:
        r"""Split each string into word-boundary character n-grams on GPU.

        Device-native counterpart to the CPU/Arrow implementation in
        :class:`~sdm.processing.text.tfidf_text_embed.TfidfTextEmbed`;
        requires a CUDA tensor and an installed cuDF. Mirrors scikit-learn's
        ``analyzer='char_wb'``: each whitespace-delimited word is padded with a
        single space on both sides before windowing, so a word shorter than
        ``n`` still yields one n-gram.

        Returns a flat :class:`StringTensor` holding every n-gram of every
        document, together with an ``offset`` tensor of length ``numel() + 1``
        where document ``d``'s n-grams are ``flat[offset[d]:offset[d + 1]]``.
        The n-grams within a document are unordered and may differ in order
        from the CPU implementation; only the per-document grouping is stable.

        Args:
            tensor: StringTensor.
            ngram_range: Inclusive ``(min_n, max_n)`` window sizes.
            lowercase: Lowercase each string before windowing.
        """
        import cudf
        import cupy as cp

        min_n, max_n = ngram_range
        n_docs = tensor.numel()
        s = tensor.to_cudf()  # n_docs documents
        if lowercase:
            s = s.str.lower()
        # Collapse every whitespace run to a single space and trim, so each
        # document splits into words on single spaces (matches str.split()).
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()

        words = s.str.split(" ")
        doc_index = cudf.Series(
            cp.repeat(cp.arange(n_docs), words.list.len().to_cupy())
        )
        flat_words = words.explode().reset_index(drop=True)
        keep = flat_words.str.len() > 0
        flat_words = flat_words[keep].reset_index(drop=True)
        doc_index = doc_index[keep].reset_index(drop=True)
        padded = " " + flat_words + " "
        pad_len = padded.str.len()

        parts: list[cudf.DataFrame] = []
        # 'character_ngrams' raises when no word is long enough for 'n'.
        longest = min(max_n, int(pad_len.max())) if len(padded) > 0 else 0
        for n in range(min_n, longest + 1):
            grams = padded.str.character_ngrams(n, as_list=True)
            long = cudf.DataFrame({"doc": doc_index, "gram": grams}).explode(
                "gram"
            )
            parts.append(long.dropna(subset=["gram"]))

        # word shorter than min_n: count it once
        short = pad_len < min_n
        parts.append(
            cudf.DataFrame({"doc": doc_index[short], "gram": padded[short]})
        )

        flat = cudf.concat(parts, ignore_index=True).sort_values("doc")

        counts = (
            flat.groupby("doc").size().reindex(range(n_docs), fill_value=0)
        )
        offset = torch.zeros(
            n_docs + 1,
            dtype=torch.int64,
            device=tensor.device,
        )
        offset[1:] = from_dlpack(counts.cumsum().astype("int64").to_dlpack())

        ngrams = tensor.from_cudf(
            flat["gram"].reset_index(drop=True), device=tensor.device
        )
        return ngrams, offset

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = table.text.device
        self._vocabularies = []

        # clean up stale buffers from previous fit
        stale_idfs = [
            name for name in self._buffers if name.startswith("idf_")
        ]
        for name in stale_idfs:
            delattr(self, name)

        for column in range(table.text.size(-1)):
            column_text = cast(
                StringTensor,
                table.text[..., column].reshape(-1),
            )
            flat, offsets = self._character_ngrams(
                column_text,
                self.ngram_range,
                lowercase=self.lowercase,
            )
            n_docs = offsets.numel() - 1

            # Factorize n-grams into codes + the unique vocabulary
            if flat.is_cuda:
                encoded = flat.to_cudf().astype("category")
                vocabulary = encoded.cat.categories.to_arrow()
                codes = torch.from_dlpack(
                    encoded.cat.codes.astype("int64").to_dlpack()
                ).to(device)
            else:
                encoded = flat.to_arrow().dictionary_encode()
                vocabulary = encoded.dictionary
                codes = arrow_as_tensor(
                    encoded.indices, dtype=torch.int64, device=device
                )  # [n_ngrams]
            vocab_size = len(vocabulary)
            # Map each n-gram back to its document via the offsets.
            doc_ids = torch.repeat_interleave(
                torch.arange(n_docs, device=device),
                offsets.diff().to(device),
            )  # [n_ngrams]

            idf = self._idf(
                doc_ids=doc_ids,
                codes=codes,
                vocab_size=vocab_size,
                n_docs=n_docs,
                device=device,
            )
            term_counts = torch.bincount(
                codes, minlength=vocab_size
            )  # [vocab_size]
            vocabulary, idf = self._prune(vocabulary, idf, term_counts)
            self._vocabularies.append(vocabulary)
            self.register_buffer(f"idf_{column}", idf)

    def _idf(
        self,
        *,
        doc_ids: Tensor,
        codes: Tensor,
        vocab_size: int,
        n_docs: int,
        device: torch.device,
    ) -> Tensor:
        """Smoothed idf per n-gram from its document frequency."""
        if vocab_size == 0:
            return torch.empty(0, device=device)
        # Count each n-gram once per document: dedup (doc, code) pairs, then
        # tally per code.  # [n_unique_pairs]
        unique_codes = (doc_ids * vocab_size + codes).unique() % vocab_size
        document_freq = torch.bincount(unique_codes, minlength=vocab_size)
        return ((1 + n_docs) / (1 + document_freq)).log() + 1.0

    def _prune(
        self,
        vocabulary: pa.Array,
        idf: Tensor,
        term_counts: Tensor,
    ) -> tuple[pa.Array, Tensor]:
        """Keep the ``max_features`` n-grams with the highest term frequency.

        Ranking follows scikit-learn's ``max_features``: n-grams are ordered by
        their total corpus occurrence count. Ties break by vocabulary order via
        a stable sort so the selection is deterministic. ``idf`` is carried
        along only as the weight aligned to each retained n-gram.
        """
        if self.max_features is None or len(vocabulary) <= self.max_features:
            return vocabulary, idf
        ranked = term_counts.argsort(descending=True, stable=True)
        keep = ranked[: self.max_features].sort().values
        return vocabulary.take(pa.array(keep.tolist())), idf[keep]

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.text.device
        dtype = (
            self.get_buffer("idf_0").dtype
            if self._vocabularies
            else torch.get_default_dtype()
        )
        text_names = table.columns[Stype.text]
        leading_shape = table.text.shape[:-1]
        n_rows = math.prod(leading_shape)

        blocks: list[Tensor] = []
        names: list[str] = []
        for column in range(table.text.size(-1)):
            vocabulary = self._vocabularies[column]
            idf = getattr(self, f"idf_{column}")
            vocab_size = len(vocabulary)

            counts = torch.zeros(
                n_rows * vocab_size, dtype=dtype, device=device
            )
            if vocab_size > 0:
                column_text = cast(
                    StringTensor,
                    table.text[..., column].reshape(-1),
                )
                flat, offsets = self._character_ngrams(
                    column_text,
                    self.ngram_range,
                    lowercase=self.lowercase,
                )
                # Map n-grams to fitted vocab indices; unseen -> -1 (dropped).
                if flat.is_cuda:
                    import cudf

                    encoded = (
                        flat.to_cudf()
                        .astype(cudf.CategoricalDtype(categories=vocabulary))
                        .cat.codes
                    )
                    codes = torch.from_dlpack(
                        encoded.astype("int64").to_dlpack()
                    ).to(device)
                else:
                    codes = arrow_as_tensor(
                        pc.call_function(
                            "index_in",
                            [flat.to_arrow()],
                            options=pc.SetLookupOptions(value_set=vocabulary),
                        ).fill_null(-1),
                        dtype=torch.int64,
                        device=device,
                    )  # [n_ngrams]
                doc_ids = torch.repeat_interleave(
                    torch.arange(n_rows, device=device),
                    offsets.diff().to(device),
                )  # [n_ngrams]
                mask = codes >= 0
                flat_index = doc_ids[mask] * vocab_size + codes[mask]
                counts.scatter_add_(
                    0, flat_index, torch.ones_like(flat_index, dtype=dtype)
                )

            tfidf = counts.view(n_rows, vocab_size) * idf  # [rows, vocab]
            norm = tfidf.norm(dim=1, keepdim=True).clamp_min(1e-12)
            blocks.append((tfidf / norm).reshape(*leading_shape, vocab_size))
            names.extend(
                f"{text_names[column]}_{i}" for i in range(vocab_size)
            )

        numerical = (
            torch.cat(blocks, dim=-1)
            if blocks
            else torch.zeros((*leading_shape, 0), dtype=dtype, device=device)
        )
        return table.__class__(
            columns={Stype.numerical: tuple(names)},
            numerical=numerical,
        )
