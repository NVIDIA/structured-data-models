# TabFM SDM Integration: Next Steps

Last updated: 2026-07-08

## Current state

The functional TabFM integration is complete, including checkpoint-compatible
attention, embeddings, row interaction, ICL, recipes, the public `TabFM`
wrapper, local checkpoint loading, and projected context caching.

Current validation baseline:

- TabFM tests: `154 passed, 16 skipped`
- Full repository: `367 passed, 90 skipped`
- Ruff, Ruff format, `ty`, dependency, whitespace, and documentation checks:
  passed
- The 16 TabFM skips are 14 CUDA cases and two official-checkpoint cases
- Other skips require unavailable CUDA, CuPy, or cuDF support

TabFM now delegates its final attention kernel and rotary tensor operation to
shared SDM utilities. Its checkpoint-facing projections, normalization,
scaling, buffer names, and transformer equations remain model-specific.

Detailed implementation and test history is recorded in
[`TABFM_SDM_INTEGRATION_LOG.md`](TABFM_SDM_INTEGRATION_LOG.md).

## 1. Add `torch.compile` coverage — complete

Create `test/models/tabfm/test_compile.py` and cover representative execution
paths where the installed PyTorch version supports `torch.compile`.

Required cases:

1. Compile a small uncached `TabFMCore` classification forward.
2. Compile a small uncached `TabFMCore` regression forward.
3. Prefill a projected context cache, then compile or exercise the cached
   query-only path.
4. Cover both unchunked and `col_chunk_size` column execution.
5. Run a second input shape through the compiled callable to detect accidental
   shape specialization.
6. Compare compiled output with eager output using dtype-appropriate
   tolerances.
7. Document any unavoidable graph break with the exact operation and reason.

Prefer a lightweight backend such as `backend="eager"` for the repository test
suite. Use Inductor as an optional smoke test when the environment supports it.
Tests should skip with a precise reason when compilation is unavailable; they
must not silently pass after an unexpected compile failure.

Suggested command:

```bash
/tmp/tabfm-smoke-venv/bin/pytest -q \
  test/models/tabfm/test_compile.py
```

Acceptance criteria:

- Compiled and eager outputs agree: **passed**.
- Cache prefill/replay semantics are unchanged: **passed**.
- No unexpected graph breaks occur in the representative paths: **passed with
  `fullgraph=True`**.
- The complete TabFM and repository test suites still pass: **passed**.

The compile cases run in a short-lived worker process to release compiler
allocations before the rest of the repository suite.

## 2. Validate official checkpoints — harness complete, execution pending

Create `test/models/tabfm/test_official_checkpoint.py` after appropriately
licensed official classification and regression checkpoint files are supplied
locally. Do not download, commit, or redistribute checkpoint artifacts.

The test should take the checkpoint directory from an environment variable,
for example `TABFM_CHECKPOINT_DIR`, and skip clearly when it is unset.

Required cases:

1. Load the official classification SafeTensors through the public SDM loader.
2. Load the official regression SafeTensors through the public SDM loader.
3. Confirm strict state-dict compatibility, including parameter names and
   tensor shapes.
4. Run deterministic eager inference on a small classification table.
5. Run deterministic eager inference on a small regression table.
6. Compare SDM results with the upstream TabFM implementation using the same
   inputs, preprocessing, model dtype, and checkpoint.
7. Exercise `fit()` followed by repeated cached `predict()` calls.
8. Confirm no raw context features or targets are retained by the cache.

Suggested command:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
TABFM_CHECKPOINT_DIR=/path/to/licensed/tabfm/checkpoints \
  /tmp/tabfm-smoke-venv/bin/pytest -q \
  test/models/tabfm/test_official_checkpoint.py
```

Acceptance criteria:

- Both official checkpoint families load strictly without key rewriting.
- SDM and upstream outputs agree within dtype-appropriate tolerances.
- Cached and uncached SDM predictions agree.
- Checkpoint files remain outside the repository and Git status.

`test/models/tabfm/test_official_checkpoint.py` now implements these checks and
skips with a precise reason when `TABFM_CHECKPOINT_DIR` is unset. Actual
execution remains blocked until appropriately licensed local files are
supplied.

## 3. Run CUDA parity and memory measurements — harness complete, execution pending

Run this work on a CUDA-capable environment with a compatible PyTorch build.
Extend the existing cache tests or add a focused CUDA test module.

Required cases:

1. Classification and regression eager parity on CUDA.
2. Cached versus uncached prediction parity.
3. Float32, bfloat16, and float16 where the GPU supports them.
4. Chunked versus unchunked column execution.
5. Repeated prediction with different query-row counts.
6. Padded batches with different context lengths.
7. Peak allocated-memory measurement for uncached inference, cache prefill,
   and cached replay.

Reset and synchronize CUDA memory statistics around each measured region:

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
```

Record the device name, PyTorch/CUDA versions, dtype, table shape, context and
query sizes, chunk size, and peak bytes. Measurements should be documented in
the integration log; avoid assertions against hardware-specific absolute
memory values.

Acceptance criteria:

- CUDA results agree with the eager reference within dtype-appropriate
  tolerances.
- Cached replay does not retain raw training tensors.
- No device transfers or dtype changes occur unexpectedly.
- Peak-memory results demonstrate and quantify the chunking/cache tradeoff.

`test/models/tabfm/test_cuda.py` now implements the parity matrix and records
device-specific peak-memory metadata. All 14 cases skip explicitly in the
current CPU-only environment; run them on a CUDA-capable host and copy the
measurements into the integration log.

## 4. Complete package-facing documentation — complete

After the runtime checks pass:

1. Add a minimal public example showing checkpoint loading, `fit()`, and
   repeated `predict()` calls.
2. Document optional checkpoint requirements and supported task types.
3. Document cache constraints: fixed schema, categorical routing, active
   width, model device/dtype, and RoPE-disabled cached encoders.
4. Verify TabFM exports remain in dependency order in each `__init__.py`.
5. Confirm no upstream-only platform, serving, or configuration abstraction
   has entered the core SDM package.

The public guide is available at `docs/source/tabfm.md`, is linked from the
guides toctree, and passes a warnings-as-errors Sphinx build.

## 5. Final release gate

Run the focused and repository-wide checks from a clean environment:

```bash
TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q test/models/tabfm

TABFM_REFERENCE_DIR=/home/ruthvikak/tabfm-reference \
  /tmp/tabfm-smoke-venv/bin/pytest -q

/tmp/tabfm-smoke-venv/bin/ruff check \
  sdm/models/tabfm test/models/tabfm

/tmp/tabfm-smoke-venv/bin/ruff format --check \
  sdm/models/tabfm test/models/tabfm

/tmp/tabfm-smoke-venv/bin/ty check \
  --python /tmp/tabfm-smoke-venv/bin/python \
  sdm/models/tabfm test/models/tabfm

/tmp/tabfm-smoke-venv/bin/pip check
git diff --check
```

Run `pre-commit run --all-files` when an environment satisfying the
repository's Python requirement is available. The current smoke environment
cannot run pre-commit because Python 3.10 is unavailable.

Final acceptance criteria:

- All available tests and static checks pass.
- Environment-dependent skips contain explicit reasons.
- The integration log includes commands, results, failures, and resolutions.
- No checkpoint artifacts, generated caches, or unrelated files are committed.
- The final diff contains only TabFM integration and intentional shared SDM
  changes.

## Handoff checklist

Before starting a work item:

- Read `TABFM_SDM_INTEGRATION_LOG.md` for implementation decisions and known
  constraints.
- Use `/home/ruthvikak/tabfm-reference` only as the upstream parity reference.
- Preserve released checkpoint parameter names and tensor layout.
- Keep shared components outside the TabFM model family only when they are
  genuinely reusable by other SDM models.

After completing a work item:

- Add every changed file and test command to the integration log.
- Record failed checks and their resolution, not only the final successful run.
- Rerun focused TabFM tests before the full repository suite.
- Update this document if the ordering or remaining work changes.
