# Categorical preprocessing compilation investigation

Scope: CPU PyTorch 2.7.1 and 2.14.0, starting at main `842c408fe`. This branch fixes one tensor operation in numeric category alignment. It does not claim that whole Kumo preprocessing recipes compile.

## Fixed: numeric category lookup

`AlignCategories._category_lookup()` aligns independently encoded numeric categories to their fitted vocabulary. Previously it created data-dependent index lengths:

```python
left_index = match.nonzero().view(-1)
right_index = perm[position[left_index]]
lookup[left_index] = right_index.to(codes.dtype)
```

Actual Inductor with `fullgraph=True` rejects `aten.nonzero.default` because its output size depends on the tensor values. With graph breaks allowed, execution can continue in separate regions.

The branch uses the same sorted vocabulary and search positions, then returns one entry per input category:

```python
return torch.where(match, perm[position].to(codes.dtype), -1)
```

Matching categories retain the same fitted code; unmatched categories retain `-1`. The empty-fitted-vocabulary return is unchanged. There is no padding of vocabularies, no precision change, and no compiler-specific execution path.

| Validation | 2.7.1 | 2.14.0 |
|---|---|---|
| Existing alignment tests, CPU | 23 passed | 23 passed |
| Actual Inductor, both graph-break settings, varying category counts | 2 passed | 2 passed |
| Special-value lookup parity, each graph-break setting | 16 passed | 16 passed |
| CUDA cases | 12 skipped | 12 skipped |

Run the existing tests plus the small compilation regression:

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
  python -m pytest test/processing/categorical/test_align.py -q
```

The transform also materializes `tuple(batch_categories)` before constructing its result, rather than passing a `BufferList` module into a traced tensor constructor. With the separately developed container/applicability patches, public numeric `AlignCategories.transform` passed actual 2.14 Inductor in both modes with missing/unseen categories and changing row counts. That integrated result does not mean this standalone branch fixes the container blockers.

Special-value parity includes NaN/infinities, duplicates, fitted category order, Boolean and unsigned integer vocabularies, empty inputs/fitted vocabularies, and int32/int64 output codes.

## Fixed: table iteration during numerical conversion

`ToNumerical` now calls `TableTensor.items(table)` instead of the bound `table.items()`. The former lets Dynamo inline the existing Python iterator. The latter was treated as a custom tensor operation returning a generator, which tracing rejects. The iterator, values, and output schema are unchanged. With the separate container/applicability patches, actual 2.14 Inductor passed both modes for mixed numerical/categorical tables with missing codes and row counts 4/6/3.

## Remaining work

| Processor/path | Remaining compatibility issue |
|---|---|
| Public processor methods | `TableTensor`, `CategoricalTensor`, other nested containers, and semantic-type applicability checks need shared tracing support. These changes are investigated on separate branches. |
| `AlignCategories.fit` | Boolean indexing materializes vocabularies containing only observed categories. Their lengths depend on the training values. Supporting fullgraph requires a deliberate dynamic-output-shape strategy; replacing the output with a fixed vocabulary would change semantics. |
| `AlignCategories.transform`, numeric | This branch removes the numeric lookup's `nonzero`, but does not fix the surrounding container operations. |
| `AlignCategories.transform`, string | `_string_category_lookups()` constructs ID tables and invokes `join_index()`. That routine uses Arrow joins. Container support alone cannot compile this external computation. |
| `AddCategoryCounts` | Default validation raises based on tensor contents (`any`, `equal`), and the processor creates/selects/concatenates table containers. Silently disabling validation is not a compatible fix. |
| `ShuffleCategories.fit` | Deduplicates randomly sampled permutations using `permutation.tolist()` and Python dictionary keys. This is state preparation dependent on sampled values. |
| `ShuffleCategories.transform` | Uses Boolean indexing when remapping valid codes, plus ensemble/container construction. A dense remap could remove the local dynamic index but does not fix the whole method. |
| `ToNumerical`, `AddCalendarFields` | Arithmetic itself is tensor based; public methods also manipulate table schemas and construct/concatenate containers. No arithmetic rewrite is justified by the failures observed so far. |

The public-method probe uses actual Inductor and records the number of nodes in each captured region. A successful graph-break-allowed call does not imply the whole processor compiled. Missing values, unseen categories, and row counts 4/6/3 are included. String alignment is tested separately from numeric alignment.

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. \
  python evidence/categorical-compilation/public_processors.py
```

No GPU performance or memory improvement is claimed. The branch's change removes one compilation blocker; broader compatibility and performance are separate validation steps.

## Integrated public-method results

Combining this branch with the initial container and processor-applicability fixes produces these results. The exact commit list is in `evidence/categorical-compilation/integration-results.jsonl`; it is a snapshot of those patches, not a claim about future revisions.

| Public operation | 2.14, graph breaks allowed | 2.14, fullgraph | 2.7.1 |
|---|---|---|---|
| Numeric alignment, transform after eager fitting | Pass | Pass | Still fails surrounding tracing |
| Numeric/string alignment, fit-transform | Fails container reconstruction after graph break | Fails data-dependent vocabulary size | Still fails surrounding tracing |
| String alignment, transform after eager fitting | Fails variable-length string operations | Fails variable-length string operations | Still fails |
| Category counts, transform after eager fitting | Pass | Fails vocabulary validation using `equal` | Allowed-breaks passes; fullgraph fails |
| Category counts, fit-transform | Fails symbolic table selection | Fails value-dependent input validation | Still fails |
| Category shuffling, fit-transform/transform | Fails container reconstruction | Fails new-container category tuple access | Still fails |
| Conversion to numerical, fit-transform/transform | Pass | Pass | Fails enum/schema guard handling |
| Calendar fields, fit-transform/transform | Pass | Pass | Fails enum/schema guard handling |

Passing results compare outputs, vocabularies, and schemas against eager behavior for row counts 4/6/3. The successful 2.14 fullgraph calls still created one graph per tested row count despite `dynamic=True`; a speedup or reuse of one graph across all shapes is not established. Numeric alignment graphs had 30 nodes, conversion 7 nodes, and calendar fields 78 nodes.

The 2.7.1 fullgraph public calls first encounter shared `handles_stypes` tracing. Bypassing that check by probing the existing internal `_transform` also exposed enum/schema guard failures for conversion and calendar fields. Fixing only the applicability check therefore does not establish full 2.7 support.
