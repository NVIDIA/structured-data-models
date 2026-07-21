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
- [ ] Reconcile PR #371 with the latest PR #359 precision work before merge.
  PR #371 predates #359's precision commit, so either apply the same autocast
  policy to both TabArena modes or explicitly retain and document fp32 for
  both.
- [ ] Add opt-in end-to-end validation for the optional TabArena dependencies
  once the example interface settles. The previous smoke tests were skipped
  by default, so normal CI did not import or execute the integration.
