# Variable-length selection with compiled preprocessing

Native string selection determines the number of output bytes from input data. The implementation previously converted that total to a Python `int`, branched on whether a symbolic one-dimensional shape was empty, and evaluated an offset-capacity check as Python control flow.

The focused changes preserve those sizes as symbolic integers, use the exact contiguous-vector storage span, and defer the existing capacity check when symbolic. A known empty selection skips packing, avoiding an Inductor empty-index lowering error. Neither byte values nor offset dtype rules change.

CPU Inductor results with dynamic input lengths and bytes, missing/empty strings and no selected entries:

| Input offset dtype | PyTorch 2.14, graph breaks allowed | PyTorch 2.14, fullgraph |
|---|---|---|
| int64 | Pass | Pass |
| int32 | Pass | Remaining data-dependent dtype promotion |

PyTorch 2.7.1 passes both offset types with graph breaks. Fullgraph first requires opt-in capture of dynamic-output/scalar operators. After opting in, int64 reaches a framework `repeat_interleave` shape guard; int32 still reaches dtype promotion. `probe_capture.py` records that diagnostic separately from default settings.

Actual `AlignCategories.fit_transform` on 2.14 passes numerical dictionaries in all sort orders and string dictionaries with code/frequency order in both graph modes, across four different contexts including all missing values. String value sorting is a separate Arrow-boundary change owned by the categorical preprocessing investigation.

Existing eager VarLen/String tests: 48 passed, 15 skipped on 2.14. This branch includes prerequisite categorical fitting and atomic constructor fixes. Focused selection commits are `d10cc007c` and `25b548f27`.

```bash
OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 PYTHONPATH=. python experiments/varlen_selection/probe.py
```

## Remaining work

Int32 packing retains its existing promotion to int64 if selected bytes exceed INT32_MAX. A general `index_select` may repeat strings, so input byte size alone is not a valid bound. Fullgraph cannot choose an output dtype from tensor data; a proven selection bound is being investigated for category filtering.

A rejected alternative replacing two value-repeat operations with one group-index repeat compiles int64 fullgraph on 2.7.1, but CPU eager packing was about 25–55% slower for medium/large examples. The patch is retained for reproducibility and is not applied to production code. It should not be used as a blanket optimization.
