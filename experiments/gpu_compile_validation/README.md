# GPU compilation validation runner

Prepared on integration `515247d3a`. This branch adds only an experiment script and documentation. No GPU result is claimed until the script is run on a CUDA host.

The runner eagerly fits one model, records eager predictions/timings, then compiles that same fitted instance. `--entry inner` compiles each internal module; `--entry predict` compiles the public prediction method. Use a fresh process for each entry point, autocast setting, and fullgraph setting. Both eager and compiled calls use identical FP32 weights and either disabled autocast or explicit BF16 autocast, including during fit. BF16 is compared with eager BF16, never eager FP32.

Each case records its first call separately, then warmups and repeated warm calls. Wall time includes CPU preparation and synchronization; CUDA event time spans GPU work and intervening host submission delays, so it is not a sum of individual kernel durations. Each call records current allocated/reserved bytes, peak allocated/reserved bytes, and peak allocated growth. The allocator is not emptied between calls; reserved bytes reflect retained allocator pools, not additional live tensor memory. The same single model/cache stays resident, avoiding multiple GPU model copies. Inputs are transferred before measurements and references are copied to CPU after measurement.

The unchanged parity tolerance is `atol=1e-5, rtol=1e-4` for both FP32 and BF16. BF16 runs may fail this strict tolerance; failures are recorded rather than silently widening it. Classification agreement and failed-value counts are also reported. `pass` means the tested predictions meet this tolerance, not that every possible input or model configuration works. Partial compilation may contain graph breaks; captured graphs and break reasons are recorded.

## Stage these files

Archive or check out this branch, including `sdm/`, `pyproject.toml`, and this script. For a git archive without `.git`, pass `--source-commit SHA` or write the exact SHA into a `SOURCE_COMMIT` file at the source root. Explicit arguments take precedence over the receipt; a Git checkout is used only as the final fallback. Install the repository dependencies for the selected CUDA/PyTorch environment; record the environment rather than silently upgrading the intended PyTorch version. The relational CUDA path also needs the project's cuDF dependencies.

| Host filename | Existing local source |
|---|---|
| `classification.npz` | `/Users/ardrianw/repositories/sdm-realdata-validation-20261002/evidence/kumo_tabular/cpu/classification/data_train_validation.npz` |
| `regression.npz` | `/Users/ardrianw/repositories/sdm-realdata-validation-20261002/evidence/kumo_tabular/cpu_regression_raw/regression/data_train_validation.npz` |
| `tabular-classifier.pt` | `/Users/ardrianw/.cache/huggingface/hub/models--nvidia--Kumo-Tabular/snapshots/bd7fa122b516c7355583873ffcf38f5e79403ebd/small/classifier.pt` |
| `tabular-regressor.pt` | Same snapshot's `small/regressor.pt` |
| `driver-dnf_bundle.pt` | `/Users/ardrianw/repositories/sdm-realdata-validation-20261002/evidence/gpu-staging-v1/evidence/kumo_relational/driver-dnf_bundle.pt` |
| `relational-classifier.pt` | `/Users/ardrianw/.cache/huggingface/hub/models--nvidia--Kumo-Relational/snapshots/613d8f930f3a9f4d306fadfe80794da8c9a28570/classifier.pt` |

The RelBench bundle contains trusted pickled SDM objects; do not substitute an untrusted pickle. `RelatedTables.to(device)` recursively transfers its table tensors. The regression validation split contains 111 rows, so request 32/111 rather than 128 when comparing actual row counts. Each sample records actual `rows` separately from `requested_rows`, and `actual_query_rows` records the complete sequence.

## Commands

Start with FP32 and one estimator. Run `--entry inner` and `--entry predict` in separate processes. Omit `--fullgraph` to allow graph breaks, then repeat with it to prohibit graph breaks.

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 python experiments/gpu_compile_validation/run.py \
  --model tabular --task classification --entry predict --device cuda \
  --autocast off --estimators 1 --query-rows 32 128 32 \
  --data /data/classification.npz --checkpoint /data/tabular-classifier.pt \
  --output /results/tabular-classification-predict-fp32-partial.json

PYTHONPATH=. OMP_NUM_THREADS=1 python experiments/gpu_compile_validation/run.py \
  --model relational --entry predict --device cuda --autocast off \
  --fullgraph --capture-dynamic-outputs --arm-index 0 \
  --query-indices 1 0 2 1 \
  --data /data/driver-dnf_bundle.pt --checkpoint /data/relational-classifier.pt \
  --output /results/relational-predict-fp32-full.json
```

Repeat with `--autocast bf16` only after the FP32 baseline, and with `--estimators 4` for tabular ensembles. For regression select `--task regression --query-rows 32 111 32` and its matching files. Relational arm 1 exercises a different/two-hop neighborhood. `--capture-dynamic-outputs` explicitly enables Dynamo capture of data-dependent output shapes; record whether it is used for each run. `--recompile-limit` is optional and its use is recorded; default limits are unchanged.

The script defaults to actual Inductor. `--device cpu --backend eager` exists solely to smoke-test runner control flow on a non-CUDA machine; it is not GPU or Inductor validation.
