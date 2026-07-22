# TODO

## TabArena follow-ups

- [ ] Keep the fitted feature schema authoritative at prediction time.
  `_align_features` currently re-runs `infer_stypes` on each query batch, so
  an all-null categorical batch can be inferred as an unsupported null type
  or rejected as a semantic-type change even though the fitted column is
  valid.
- [ ] Preserve exact classification-label identity across the pandas/SDM
  boundary. `_class_labels_by_key` stringifies the original pandas scalar,
  while TabICLv2 builds prediction-column keys from tensor values returned by
  `tolist()`. Some `float32` labels therefore produce different strings and
  cannot be mapped back. Add coverage for nontrivial floating-point labels.
