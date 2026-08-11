# GPU WordPiece Tokenization for SentenceTransformer

Replace CPU tokenization with cuDF's `WordPieceVocabulary` for WordPiece-based models.
BPE models fall back to the current `model.encode()` path.

## Implementation items

- [ ] Detect tokenizer type (WordPiece vs BPE) in `__init__` via `transformers.PreTrainedTokenizerFast` backend inspection
- [ ] Build `WordPieceVocabulary` from the HuggingFace tokenizer vocab in `__init__`
- [ ] Build `CharacterNormalizer` with `do_lower` and special tokens from the HuggingFace tokenizer in `__init__`
- [ ] Store special token IDs (`[CLS]`, `[SEP]`, `[PAD]`) and `max_length` from the tokenizer
- [ ] Convert `StringTensor` to `cudf.Series` via `text.to_cudf()` (zero-copy on GPU)
- [ ] Handle nulls on GPU (fill null strings with empty strings via cuDF)
- [ ] Normalize text on GPU via `CharacterNormalizer.normalize()`
- [ ] Tokenize on GPU via `WordPieceVocabulary.tokenize()` (returns `Series` of `list[int32]`)
- [ ] Truncate, prepend `[CLS]`, append `[SEP]`, pad to uniform length, and build `attention_mask` — all on GPU
- [ ] Batch the padded token ID tensors and run the transformer + pooling forward pass directly (bypass `model.encode()`)
- [ ] Gate the GPU path on cuDF availability; fall back to current `model.encode()` path otherwise
