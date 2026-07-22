# TODO

## TabArena follow-ups

- [ ] Preserve exact classification-label identity across the pandas/SDM
  boundary. `_class_labels_by_key` stringifies the original pandas scalar,
  while TabICLv2 builds prediction-column keys from tensor values returned by
  `tolist()`. Some `float32` labels therefore produce different strings and
  cannot be mapped back. Add coverage for nontrivial floating-point labels.
