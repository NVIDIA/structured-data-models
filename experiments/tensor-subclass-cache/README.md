# Tensor-subclass compiler caches

**Experimental; keep this cache optimization out of integration.** Review
found an additional PyTorch 2.14 cache correctness blocker: changing storage
aliasing can reuse a compiled mutation graph incorrectly. The failure also
occurs with plain PyTorch tensors and no SDM code. The wrapper hook is never
called in that reproduction, so changing this hash cannot fix it.

The original hash also omitted storage sharing between distinct leaf views.
Revision v2 now includes deterministic storage-group numbers as well as view
offsets/strides. That metadata correction does not establish general cache
safety, especially for aliasing between separate inputs.

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
names, repeated-leaf aliases, and shared-storage groups. It never reads tensor values. Unknown
metadata types raise an error rather than silently disappear from the key.

## Accepted findings and rejected rollout

- Accepted: metadata-only hashing removes the symbolic serialization failure
  in the small read-only example; cache invalidation checks below pass.
- Accepted workaround: on the tested 2.14 CPU mutation reproduction,
  `TORCHINDUCTOR_AUTOGRAD_CACHE=0` restores correct results while leaving
  Inductor kernel caching active. No production compiler setting is changed.
- Rejected for integration: enabling the optional cache optimization while
  these alias/mutation cases remain unresolved.

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

## Separate native PyTorch 2.14 correctness blocker

The standalone `alias_repro.py` imports only PyTorch. It compiles a function
that mutates its first input and clones its second. When the inputs initially
share storage, the expected result is ones. After recompilation with
independent inputs, the expected result is zeros, but the cached graph returns
ones. Instrumented SDM-wrapper controls made zero calls to the hash hook and
failed identically with every hook removed. Independent mutation probes that
do invoke the hook pass with both hash revisions, including separate wrapper
inputs sharing storage. No hook-specific wrong result has been demonstrated.

```bash
# With an unused cache directory: reproduces incorrect output/assertion failure.
TORCHINDUCTOR_CACHE_DIR=/tmp/native-alias-case python \
  experiments/tensor-subclass-cache/alias_repro.py

# Scoped workaround; kernel caching remains enabled.
TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/native-alias-case-no-aot python \
  experiments/tensor-subclass-cache/alias_repro.py
```

On CPU PyTorch 2.14.0 the first command returns ones instead of zeros for the
independent-input case. The workaround returns the correct sequence (ones,
zeros, ones), and the third case records an Inductor FX graph cache hit.
This alias/mutation result has not been established on CUDA or PyTorch 2.7.
The separate 2.7 flatten-ABI failure above remains a different problem.
