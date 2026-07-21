from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm.processing.base import Processor, SharedState
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor
from sdm.tensor.io import arrow_as_tensor


class TfidfEncoder(Processor):
    r"""Encode text columns as character ``char_wb`` TF-IDF vectors.

    Each text column is tokenized into word-boundary character n-grams (see
    :meth:`~sdm.tensor.StringTensor.character_ngrams`), and a separate
    vocabulary and inverse-document-frequency (idf) weighting is fitted per
    column on the context table. Every column expands to a block of numerical
    features (one per fitted n-gram), and the blocks are concatenated into the
    numerical output. The idf smoothing matches scikit-learn's
    ``smooth_idf=True`` default, and rows are L2-normalized.

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
        # Learned per text column: the n-gram vocabulary and its aligned idf.
        # Held in `SharedState` so ensemble members share one fitted copy
        # instead of duplicating it through `copy.deepcopy`.
        self._state: SharedState[tuple[list[pa.Array], list[Tensor]]] = (
            SharedState(([], []))
        )

    @property
    def _vocabularies(self) -> list[pa.Array]:
        return self._state.value[0]

    @property
    def _idfs(self) -> list[Tensor]:
        return self._state.value[1]

    def _column_ngrams(
        self,
        table: TableTensor,
        column: int,
    ) -> tuple[StringTensor, Tensor]:
        """Tokenize one text column into ``(flat_ngrams, offsets)``."""
        column_text = cast(StringTensor, table.text[:, column])
        return column_text.character_ngrams(
            self.ngram_range,
            lowercase=self.lowercase,
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = table.numerical.device
        self._state.value = ([], [])
        for column in range(table.text.size(-1)):
            flat, offsets = self._column_ngrams(table, column)
            n_docs = offsets.numel() - 1

            # Factorize n-grams into codes + the unique vocabulary (arrow C++).
            encoded = flat.to_arrow().dictionary_encode()
            vocabulary = encoded.dictionary
            vocab_size = len(vocabulary)
            codes = arrow_as_tensor(
                encoded.indices, dtype=torch.int64, device=device
            )  # [n_ngrams]
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
            vocabulary, idf = self._prune(vocabulary, idf)
            self._vocabularies.append(vocabulary)
            self._idfs.append(idf)

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
    ) -> tuple[pa.Array, Tensor]:
        """Keep the ``max_features`` most frequent n-grams (lowest idf)."""
        if self.max_features is None or len(vocabulary) <= self.max_features:
            return vocabulary, idf
        keep = idf.argsort()[: self.max_features].sort().values
        return vocabulary.take(pa.array(keep.tolist())), idf[keep]

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.numerical.device
        dtype = (
            self._idfs[0].dtype if self._idfs else torch.get_default_dtype()
        )
        text_names = table.columns[Stype.text]
        n_rows = table.text.size(0)

        blocks: list[Tensor] = []
        names: list[str] = []
        for column in range(table.text.size(-1)):
            vocabulary = self._vocabularies[column]
            idf = self._idfs[column]
            vocab_size = len(vocabulary)

            counts = torch.zeros(
                n_rows * vocab_size, dtype=dtype, device=device
            )
            if vocab_size > 0:
                flat, offsets = self._column_ngrams(table, column)
                # Map n-grams to fitted vocab indices; unseen -> -1 (dropped).
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
            blocks.append(tfidf / norm)
            names.extend(
                f"{text_names[column]}_{i}" for i in range(vocab_size)
            )

        numerical = (
            torch.cat(blocks, dim=-1)
            if blocks
            else torch.zeros((n_rows, 0), dtype=dtype, device=device)
        )
        return table.__class__(
            columns={Stype.numerical: tuple(names)},
            numerical=numerical,
        )
