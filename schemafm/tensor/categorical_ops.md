# CategoricalTensor Category-Preserving Ops

`CategoricalTensor` stores integer category codes in `data` and one category
vector per logical category column.

The core invariant is:

```python
data.shape == (*batch_dims, C)
len(categories) == C
categories[c] describes data[..., c]
```

Operations can preserve `CategoricalTensor` metadata only when this invariant
remains true. Otherwise, they should fall back to returning a plain
`torch.Tensor`.

## Always Safe

These operations do not change category-column semantics:

- `detach`
- `clone`
- `contiguous`
- `_to_copy` / `.to`
- `_pin_memory`
- `alias`

## Batch-Dimension View Ops

These can preserve categories only when the original last dimension remains the
result's last dimension:

- `view`
- `_unsafe_view`
- `reshape`
- `squeeze`, only if it does not remove the last dimension
- `unsqueeze`, only if the new dimension is inserted before the category
  dimension
- `expand`, only if the last dimension remains `C`
- `transpose`, only if neither swapped dimension is the last dimension
- `permute`, only if `dims[-1] == old_dim - 1`
- `movedim`, only if the old category dimension lands at result dimension `-1`
- `swapdims` / `swapaxes`, only for non-category dimensions

## Basic Indexing

These can preserve categories:

- row or batch slicing: categories unchanged
- `narrow` on batch dimensions: categories unchanged
- `slice` on batch dimensions: categories unchanged
- `slice` on the last dimension: slice categories the same way
- `index_select` on batch dimensions: categories unchanged
- `index_select` on the last dimension: reorder categories, if the index can be
  read safely
- advanced integer indexing on the last dimension: reorder categories, if the
  metadata update is unambiguous

Scalar selection on the category dimension should fall back because it removes
the category dimension:

```python
x[..., 2]    # fallback to Tensor
x[..., 2:3]  # preserves CategoricalTensor
```

## Multi-Tensor Shape Ops

These can preserve categories with compatibility checks:

- `cat` along batch dimensions: require matching categories
- `cat` along the last dimension: concatenate categories
- `stack`: preserve only when the new dimension is inserted before the category
  dimension and all inputs have matching categories
- `split` / `chunk` / `tensor_split` on batch dimensions: categories unchanged
- `split` / `chunk` / `tensor_split` on the last dimension: split categories the
  same way
- `unbind` on batch dimensions: categories unchanged

`unbind` on the last dimension should fall back because it removes the category
dimension.

## Last-Dimension Reordering

These can preserve categories by applying the same logical transform to the
category tuple:

- `x[..., ::2]`: `categories[::2]`
- `flip` on the last dimension: reverse categories
- `roll` on the last dimension: roll categories
- `repeat` / `tile`: preserve if category repeats are reflected in repeated
  categories
- `repeat_interleave` on the last dimension: preserve if repeats are statically
  known; otherwise fall back

## Maybe Later

These are possible but need stricter semantics:

- `where(condition, x, y)`, if `x` and `y` have identical categories
- `copy_`, only from compatible categorical tensors
- `zeros_like` / `ones_like` / `full_like`, if generated codes are accepted as
  categorical data
- `empty_like`, technically possible but probably not useful

## Should Fall Back

These should return plain tensors because category meaning is lost or code
validity is not guaranteed:

- arithmetic: `add`, `sub`, `mul`, etc.
- comparisons: `eq`, `lt`, etc.
- reductions: `sum`, `min`, `max`, `any`, etc.
- `unique`
- `masked_select`
- `take`
- `gather`
- `scatter`
- `flatten`, if it merges the category dimension
- `transpose` / `permute`, if they move the category dimension away from last
- `diagonal`, in most cases
- scalar select on the last dimension

## Suggested Implementation Order

1. `detach`
2. basic `slice` / `select` policy
3. `view` / `reshape`
4. `transpose` / `permute`
5. `cat` / `stack` / `split`
