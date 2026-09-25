# Batched KumoTabular-Small fair comparison

The portable runner compares KumoTabular-Small, registry-default Causilo and registry-default TabPFN-3.5-Fast through the same native TabArena protocol: `outer_experiments=False`, eight inner folds and one set, `refit_folds=False`, sequential fold fitting, one GPU and eight CPUs. Eight fitted fold children serve predictions; Kumo's eight internal estimators are separate from those children. There is no HPO, context subsampling, extra real-data warm fit or replacement timing loop.

`kumo-batched` explicitly sets `estimator_batch_size="auto"`, `kv_cache=False`, eight internal estimators, full context, native Small weights and default FP16 autocast. In this branch, no-KV auto batches compatible estimators only when context rows plus the current query chunk's rows are at most 3000 and that sum times the context column count is at most 50000; both bounds are inclusive. Explicit overrides and CPU/subsampled-context fallbacks remain in the adapter. Causilo and TabPFN retain their own registry estimator counts, precision and preprocessing; only their ensemble refit setting is overridden to match the protocol.

The runtime combines main `5995e13d1d64e4283f69de9ca44a604201808799` with the batching changes from `66b3b19db6e5c307f3725a55fdf815ef8bafc2c7`. This main revision defaults to Large and selects checkpoint revision v1.0.7, so the benchmark adapter explicitly selects Small for both checkpoint loading and its metadata model. Record the resolved Small checkpoint hashes and verify their relationship to any earlier v1.0.6 baseline; equal values must be established rather than assumed. It differs from the older standalone PR948 synthetic screen and must be measured separately. Batching can change floating-point outputs and labels. Speed alone does not establish acceptable prediction quality.

## Environment and source receipts

Run from this repository's committed branch on a CUDA host. Keep the environment, data, weights, manifests and results outside the checkout. Use a CUDA-compatible PyTorch installation; the reference runtime is Python 3.12, Torch 2.14/CUDA 13.0, AutoGluon 1.6.4b20260924, Causilo 1.0.2 and TabPFN 9.0.0. Different hardware or runtime versions produce a new comparison, not identical timing provenance.

```bash
export SDM_ROOT="$PWD"
export BENCH_ROOT="$PWD/../fair-benchmark-artifacts"
export TABARENA_ROOT="$BENCH_ROOT/tabarena"
export BENCH_PYTHON="$BENCH_ROOT/venv/bin/python"
mkdir -p "$BENCH_ROOT"
uv venv --python 3.12 "$BENCH_ROOT/venv"
uv --no-config pip install --python "$BENCH_PYTHON" --prerelease=allow \
  -e "$SDM_ROOT" "autogluon.tabular==1.6.4b20260924" \
  "tabarena[data-foundry,plot,preprocessing]>=0.1.1.dev20260924104714,<0.2" \
  "causilo==1.0.2" "tabpfn==9.0.0"
git clone https://github.com/autogluon/tabarena.git "$TABARENA_ROOT"
git -C "$TABARENA_ROOT" checkout 290bc25150656c9f04499a975408eec7d0609f82
export PYTHONPATH="$SDM_ROOT:$TABARENA_ROOT/packages/tabarena/src:$TABARENA_ROOT/packages/bencheval/src"
uv pip freeze --python "$BENCH_PYTHON" > "$BENCH_ROOT/packages.txt"
uv run --no-project --python "$BENCH_PYTHON" python \
  -m benchmark.tabular.tabarena.fair manifest \
  --root "$SDM_ROOT" --output "$BENCH_ROOT/sdm-manifest.json"
uv run --no-project --python "$BENCH_PYTHON" python \
  -m benchmark.tabular.tabarena.fair manifest \
  --root "$TABARENA_ROOT" --output "$BENCH_ROOT/tabarena-manifest.json"
```

The manifest command runs in the configured benchmark environment. It records tracked source bytes, symlinks, commit and tracked diff identity; untracked source files are excluded. Generate final manifests after committing all runner/source changes. The runner verifies the supplied files before and after execution and checks source import locations. Keep the manifests, package receipt, checkpoint hashes and accelerator/driver information with the results. Model libraries acquire their normal weights; Record the source-selected Small checkpoint revision; TabPFN must use the registry's `tabpfn-v3.5-fast-20260909.safetensors` checkpoint. Use the libraries' normal authentication when a checkpoint requires it; credentials do not belong in manifests or results.

When extending an existing cohort instead of rerunning every method, reuse its **exact TabArena manifest file** for candidate runs after verifying that the pinned source matches. The evaluator checks commit and manifest SHA256; regenerating a different JSON receipt for identical files does not match an existing cohort's freeze. In that workflow, replace the generated TabArena receipt before any candidate run:

```bash
cp "$ORIGINAL_TABARENA_MANIFEST" "$BENCH_ROOT/tabarena-manifest.json"
```

## One split and the official grid

Set CPU thread limits and select one otherwise idle GPU. The shared arguments contain only host-local paths chosen above:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
COMMON=(
  --source-root "$SDM_ROOT" --source-manifest "$BENCH_ROOT/sdm-manifest.json"
  --tabarena-root "$TABARENA_ROOT" --tabarena-manifest "$BENCH_ROOT/tabarena-manifest.json"
  --openml-cache "$BENCH_ROOT/openml" --tabarena-cache "$BENCH_ROOT/tabarena-cache"
  --num-cpus 8 --refit-folds no
)
CUDA_VISIBLE_DEVICES=0 uv run --no-project --python "$BENCH_PYTHON" python \
  -m benchmark.tabular.tabarena.fair "${COMMON[@]}" \
  --model kumo-batched --dataset blood-transfusion-service-center \
  --repeat 0 --fold 0 --output "$BENCH_ROOT/results/kumo-batched/blood-r0-f0"
```

Substitute `--model causilo` or `--model tabpfn-fast`, with a distinct output directory, to run the competitors on that same split. The command preserves official dummy/shared-weight warmup, fit, pre-predict preparation, prediction, result assembly and cleanup. Native `time_train_s` includes actual fold fitting/preprocessing/validation; `time_infer_s` measures native outer-test prediction and CPU output. One-time warmup and timing-audit fields are retained separately. Test and OOF predictions remain in the native `results.pkl` without reconstruction or averaging by this runner.

Add `--plan-only` to inspect a split without fitting. Export the pinned complete grid without selecting a dataset:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project --python "$BENCH_PYTHON" python \
  -m benchmark.tabular.tabarena.fair "${COMMON[@]}" --model kumo-batched \
  --output "$BENCH_ROOT/grid-plan" --export-grid "$BENCH_ROOT/grid.json"
```

The pinned grid contains 51 datasets and 816 outer splits. Run every exported `(dataset, repeat, fold)` for each selected method, using a fresh output directory per attempt and one process per assigned GPU. Preserve failures separately; never count a failed or duplicate attempt as another split. The runner leaves native predictions/results and removes fitted models through native cleanup. It requires 128 GiB of scratch headroom by default; `--min-free-gib` changes that preflight check without changing model settings.

Post-timer child metadata records configured batching, actual fitted context shapes, internal estimator count and subsampling state. It does not instrument actual query chunks or executed member groups. AutoGluon can split a test set into multiple prediction calls, so outer-test row count alone does not prove which batching path executed.

## Aggregate timings, quality and prediction changes

Aggregate newly measured batched Kumo and the two competitors, even when results live in separate roots:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project --python "$BENCH_PYTHON" python \
  -m benchmark.tabular.tabarena.evaluate_fair \
  --run-root "$BENCH_ROOT/results" --grid "$BENCH_ROOT/grid.json" \
  --models kumo-batched causilo tabpfn-fast \
  --output "$BENCH_ROOT/evaluation" --official-plots
```

Full scoring requires all 816 successful splits for each selected method, matching pinned TabArena metadata and one source/harness identity per model. No missing-method values are imputed. `--partial` instead reports only common completed splits and cannot produce full-suite scoring/plots. Timings use native fractional mean outer train/test row denominators: normalize each split to seconds/1000 rows, average within each dataset, then take the median over datasets. Test and OOF errors are both retained. Any native Elo in the plots is relative to the selected newly measured pool; it is not comparable to a public leaderboard Elo.

To check the candidate against the **original unbatched main cohort**, provide that cohort's untouched result directory and exact source commit. Candidate runs must have reused that cohort's exact TabArena manifest as described above:

```bash
export ORIGINAL_RESULTS=/path/to/original-main-results
export BASELINE_COMMIT=17dfa3de6bb91f4629c0d0f2a9e768cfb52b351d
CUDA_VISIBLE_DEVICES='' uv run --no-project --python "$BENCH_PYTHON" python \
  -m benchmark.tabular.tabarena.evaluate_fair \
  --run-root "$ORIGINAL_RESULTS" "$BENCH_ROOT/results" \
  --grid "$BENCH_ROOT/grid.json" --models kumo-batched kumo \
  --partial --compare-predictions --baseline-source-commit "$BASELINE_COMMIT" \
  --output "$BENCH_ROOT/paired-quality"
```

This matches exact dataset/repeat/fold identities, verifies raw hashes, row/target/class order, and compares preserved test/OOF arrays. It reports dtype/byte equality, maximum and mean absolute differences, changed values and classification labels, plus native test/OOF error deltas. The composed branch's `--model kumo` selects sequential prediction within the composed source; it is **not** the original main baseline. Keep such a diagnostic in another result root and do not relabel it as the historical cohort. No manifests, outputs, environments, credentials or model caches belong in this branch.
