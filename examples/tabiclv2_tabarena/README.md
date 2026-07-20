# TabICLv2 on TabArena

This optional example evaluates the repository-local `TabICLv2` model through
TabArena and AutoGluon. It has no Ray runtime dependency and runs every
selected job in one local process.

## Installation and requirements

Install the repository test dependencies and the pinned optional integration
dependencies from the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[test]' \
  -r examples/tabiclv2_tabarena/requirements.txt
```

Do not install the `tabarena` PyPI placeholder package: it does not provide the
`tabarena.benchmark` API used by this example. The requirements file pins
compatible TabArena and AutoGluon source revisions; update them together and
rerun the smoke test when upgrading.

The first run needs internet access to clone the optional dependencies, fetch
the TabICLv2 checkpoint from Hugging Face, and download selected TabArena tasks
from OpenML. A GPU run requires CUDA-compatible PyTorch and a CUDA-capable GPU.
Set `--num-gpus 0` to run on CPU instead; CPU runs can take substantially
longer.

## Usage

Run a small outer-evaluation smoke test from the repository root:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-local-smoke \
  --outer \
  --num-estimators 1 \
  --precision auto \
  --num-cpus 1 \
  --num-gpus 1 \
  --subset lite \
  --datasets blood-transfusion-service-center anneal QSAR_fish_toxicity
```

For a CPU-only single-dataset smoke run:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-cpu-smoke \
  --num-estimators 1 \
  --precision fp32 \
  --num-cpus 1 \
  --num-gpus 0 \
  --datasets blood-transfusion-service-center
```

Omit `--datasets` and `--subset` to run the complete TabArena suite locally:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-local-full \
  --num-estimators 1 \
  --precision auto \
  --num-cpus 1 \
  --num-gpus 1
```

| Parameter                    | Required           | Meaning                                                                                                                                                |
| ---------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--output-root PATH`         | Yes                | New or empty directory for TabArena artifacts and `report/results_per_split.csv`.                                                                      |
| `--num-estimators N`         | No; default `8`    | Number of TabICLv2 estimators per task. `N` must be at least `1`; use `1` for a smoke test.                                                            |
| `--precision MODE`           | No; default `auto` | `auto` uses CUDA bf16 only when supported, `bf16` requires supported CUDA hardware, and `fp32` disables autocast.                                      |
| `--num-cpus N`               | No                 | CPU resource request passed to TabArena. `N` must be at least `1`; omit it to let TabArena auto-detect resources.                                      |
| `--num-gpus N`               | No                 | GPU resource request passed to TabArena. `N` must be non-negative; use `0` for CPU-only execution. A positive value requires CUDA while TabICLv2 fits. |
| `--outer`                    | No                 | Builds TabArena outer-evaluation experiments; appropriate for the smoke run.                                                                           |
| `--subset NAME [NAME ...]`   | No                 | TabArena task-subset names, such as `lite`.                                                                                                            |
| `--datasets NAME [NAME ...]` | No                 | Exact TabArena dataset names. Separate multiple names with spaces, not commas.                                                                         |

Omit both `--subset` and `--datasets` to select the complete suite. When both
are supplied, TabArena applies both filters. Each run requires a new or empty
output directory, preserves TabArena's normal output and cache structure,
writes completed SDM result records to `report/results_per_split.csv`, and
propagates job failures.

## Precision

TabICLv2's standalone GPU example uses bfloat16 autocast around both cached
`fit` and `predict` execution. This runner exposes the same choice as a run
parameter so precision is visible in the TabArena configuration and in the
CSV report:

- `auto` is the default: it enables bf16 autocast only on CUDA hardware that
  reports bf16 support, and otherwise uses fp32.
- `bf16` requires CUDA bf16 support and fails early when it is unavailable.
- `fp32` disables autocast, for a precision/parity baseline or CPU execution.

Autocast affects forward-pass activations and intermediate operations, not the
stored checkpoint dtype. On supported GPUs, bf16 typically lowers peak memory
and can improve throughput; it can also cause small numerical differences from
the fp32 baseline.

## Smoke-test command

Run the opt-in end-to-end smoke test with:

```bash
SDM_RUN_TABARENA_SMOKE=1 .venv/bin/python -m pytest \
  test/examples/test_tabiclv2_tabarena.py::test_real_tabarena_smoke -q
```

The test runs the three datasets from the GPU smoke command: one binary,
multiclass, and regression task. It is not the complete TabArena suite.
