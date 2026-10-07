# Tensor-subclass compiler caches

This branch adds PyTorch's `_stable_hash_for_caching` hook to SDM tensor
wrappers. It changes cache keys, not model calculations or tensor flattening.
The implementation is exercised by PyTorch 2.14; PyTorch 2.7 does not call it.

## What fails

A small compiled operation is enough to reproduce the 2.14 problem:

```python
def forward(table):
    return table[:3].numerical.sin()

compiled = torch.compile(forward, fullgraph=True, dynamic=True)
compiled(table)  # Numeric features plus string ID columns.
```

The calculation succeeds, but AOTAutograd bypasses its disk cache:

```text
Failed to pickle cache key
AttributeError: Can't get local object 'WeakValueDictionary.__init__.<locals>.remove'
```

PyTorch's default wrapper hash pickles symbolic tensor metadata, which reaches
its unpicklable shape environment. The new hash describes the symbolic
expression and hint instead. It includes the wrapper class, schema, tensor
shape/strides/offset, dtype/device, inference/gradient flags, flattened leaf
names, and repeated-leaf aliases. It never reads tensor values. Unknown
metadata types raise an error rather than silently disappear from the key.

## Results

Each cache check below ran in a separate CPU process with actual Inductor.
The two matching runs also used different `PYTHONHASHSEED` values.

| Runtime and case | AOT cache | Result |
|---|---|---|
| 2.14, original hash, symbolic string table | Bypassed | Correct output; pickle warning |
| 2.14, new hash, first process | Miss, saved | Correct output |
| 2.14, new hash, second process | Hit | Correct output; Inductor cache hit |
| 2.14, changed tensor values only | Hit | Correct new output |
| 2.14, changed column schema | Miss | Correct output; existing kernel reusable |
| 2.14, changed numerical strides | Miss | Correct output |
| 2.14, added flatten-protocol leaf | Miss | Correct output |
| 2.7, added flatten leaf, old cache reused | Incorrect old entry reused | Argument-unpacking error |
| 2.7, same added leaf, fresh cache namespace | Miss, saved | Correct output |

The focused metadata tests also pass on both versions. GPU caches and other
PyTorch versions are unvalidated. These results establish correct cache reuse
for the listed cases, not a general inference speed improvement.

## Reproduce

Use an unused cache directory for the first command, then reuse it unchanged
for the second. Run from this checkout with the selected Python environment.

```bash
PYTHONPATH=. PYTHONHASHSEED=1 python experiments/tensor-subclass-cache/check.py \
  --dynamic --cache-dir /tmp/sdm-cache-check-214
PYTHONPATH=. PYTHONHASHSEED=23 python experiments/tensor-subclass-cache/check.py \
  --dynamic --cache-dir /tmp/sdm-cache-check-214
```

Add `--variant schema`, `--variant strides`, or `--variant flatten` to test
invalidation. `--variant values` checks reuse with different tensor data. `--without-hash` reproduces the original 2.14 cache bypass.
On 2.7, use `--numeric-only` without `--dynamic`: run the base case, then
`--variant flatten` using the same directory to reproduce the stale-entry
error. That variant simulates adding an alias leaf between source revisions.

## PyTorch 2.7 workaround

PyTorch 2.7 has no stable wrapper-hash extension. A new helper method cannot
repair that older cache implementation. After changing a wrapper's flattening
protocol, use a cache directory specific to the source revision, for example:

```bash
TORCHINDUCTOR_CACHE_DIR=/tmp/sdm-inductor-<source-revision> python your_script.py
```

This preserves caching within one revision without deleting unrelated caches.
A representation hash cannot identify arbitrary changes to dispatch or
unflatten implementation semantics either; separate cache namespaces remain
appropriate when developing those implementations on either runtime.
