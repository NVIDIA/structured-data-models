# TabICLv2 on TabArena

This optional example evaluates the repository-local `TabICLv2` model through
TabArena. It has no Ray runtime dependency and runs every selected job in one
local process.

It offers two intentionally distinct integration modes:

- `autogluon-compatible` (default) routes the model through TabArena's
  AutoGluon wrapper. AutoGluon fits feature generation and classification label
  cleaning before the SDM adapter runs.
- `sdm-native` routes raw features and labels through TabArena's external-system
  interface. SDM owns schema validation, semantic typing, feature recipes, and
  class-label handling.

The modes are useful for comparison, but their scores and timings are not
interchangeable because they transform the same raw table differently.

## Installation and requirements

Install the repository test dependencies and the pinned optional integration
dependencies from the repository root:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[test]' \
  -r examples/tabiclv2_tabarena/requirements.txt
```

If `uv` is unavailable, use the equivalent pip installation:

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]' \
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

If Hugging Face Xet downloads fail or are rate-limited in the current
environment, set `HF_HUB_DISABLE_XET=1` for the command to use the standard
download path instead.

### Optional comparison reports

The local runner does not require plotting dependencies. To generate TabArena
comparison plots and leaderboard reports, install the pinned TabArena `plot`
extra after the core requirements:

```bash
uv pip install --python .venv/bin/python \
  'tabarena[plot] @ git+https://github.com/autogluon/tabarena.git@7fee3bef1670be0bd52b4ecb99f7761e97b06068#subdirectory=packages/tabarena'
```

## Usage

Run a small outer-evaluation smoke test from the repository root:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-local-smoke \
  --outer \
  --num-estimators 1 \
  --num-cpus 1 \
  --num-gpus 1 \
  --subset lite \
  --datasets blood-transfusion-service-center anneal QSAR_fish_toxicity
```

Run the same smoke selection with SDM-native preprocessing:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-native-smoke \
  --mode sdm-native \
  --num-estimators 1 \
  --num-cpus 1 \
  --num-gpus 1 \
  --subset lite \
  --datasets blood-transfusion-service-center anneal QSAR_fish_toxicity
```

### Official-policy SDM-native control

Use this separate, opt-in runner to keep SDM-native preprocessing while
matching TabICLv2's official eight-member ensemble *schedule* (four paired
base configurations, each with `none` and `power` normalization):

```bash
python -m examples.tabiclv2_tabarena.run_official_policy \
  --output-root outputs/tabiclv2-native-official-policy-smoke \
  --num-cpus 1 \
  --num-gpus 1 \
  --subset lite \
  --datasets blood-transfusion-service-center anneal QSAR_fish_toxicity
```

`--random-state` defaults to `42`, TabICLv2's official package default. The
runner always uses exactly eight members and writes a distinct
`SDMTabICLv2OfficialPolicySystem` result, so its artifacts cannot be mixed
with the regular SDM-native runs.

This is a policy-equivalent control, not an official-preprocessing control.
Categorical code ordering, numerical/categorical block ordering, ID and
datetime handling, SDM's tensor Yeo--Johnson implementation, and TabArena's
upstream AutoGluon preprocessing intentionally remain different from the
official package.

For a CPU-only single-dataset smoke run:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-cpu-smoke \
  --num-estimators 1 \
  --num-cpus 1 \
  --num-gpus 0 \
  --datasets blood-transfusion-service-center
```

Omit `--datasets` and `--subset` to run the complete TabArena suite locally:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-local-full \
  --num-estimators 1 \
  --num-cpus 1 \
  --num-gpus 1
```

| Parameter                    | Required                           | Meaning                                                                                                                                                |
| ---------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--output-root PATH`         | Yes                                | New or empty directory for TabArena artifacts and `report/results_per_split.csv`.                                                                      |
| `--num-estimators N`         | No; default `8`                    | Number of TabICLv2 estimators per task. `N` must be at least `1`; use `1` for a smoke test.                                                            |
| `--num-cpus N`               | No                                 | CPU resource request passed to TabArena. `N` must be at least `1`; omit it to let TabArena auto-detect resources.                                      |
| `--num-gpus N`               | No                                 | GPU resource request passed to TabArena. `N` must be non-negative; use `0` for CPU-only execution. A positive value requires CUDA while TabICLv2 fits. |
| `--mode MODE`                | No; default `autogluon-compatible` | `autogluon-compatible` uses AutoGluon feature/label preprocessing; `sdm-native` uses SDM-owned preprocessing.                                          |
| `--outer`                    | No                                 | Builds TabArena outer-evaluation experiments in `autogluon-compatible` mode. `sdm-native` always uses TabArena's full-data system protocol.            |
| `--subset NAME [NAME ...]`   | No                                 | TabArena task-subset names, such as `lite`.                                                                                                            |
| `--datasets NAME [NAME ...]` | No                                 | Exact TabArena dataset names. Separate multiple names with spaces, not commas.                                                                         |

Omit both `--subset` and `--datasets` to select the complete suite. When both
are supplied, TabArena applies both filters. Each run requires a new or empty
output directory, preserves TabArena's normal output and cache structure,
writes completed SDM result records to `report/results_per_split.csv` with its
integration mode, and propagates job failures.

`sdm-native` deliberately drops columns inferred as IDs and rejects datetime
columns before fitting. Its current recipe supports numerical and categorical
columns, including train-fitted handling of missing and unseen categorical
values. Add an SDM datetime recipe before using that mode on datetime tasks;
do not rely on silent column dropping. `autogluon-compatible` is the lower-risk
choice when broad TabArena dtype handling is more important than SDM ownership.

### Class-probability ordering in `sdm-native`

SDM may emit class probabilities in a recipe-specific order because it preserves
the class labels associated with its transformed categorical target. The native
adapter maps those output labels back to the original labels, validates that one
column is present for every fitted class, and reorders the frame to TabArena's
own `LabelCleaner` class order before returning it for scoring.

This is an output-format compatibility step only. The adapter still passes raw
targets to TabICLv2, and neither AutoGluon label cleaning nor AutoGluon feature
generation is used to train or preprocess the SDM model in `sdm-native` mode.
The alignment is necessary because TabArena's scorer consumes multiclass
probability matrices positionally after its label conversion.

## Choosing a mode

| Mode                   | Advantages                                                                                                   | Trade-offs                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `autogluon-compatible` | Established TabArena/AutoGluon integration; lower compatibility risk for raw dataframe types and labels.     | AutoGluon owns part of the feature and label contract. It is not an SDM-only preprocessing result.      |
| `sdm-native`           | Raw tables, categorical fitting, schemas, and labels are owned by SDM and auditable at the adapter boundary. | Datetime tasks fail clearly until SDM support is added; results may differ from the compatibility mode. |

## Smoke-test command

Run the opt-in end-to-end smoke test with:

```bash
SDM_RUN_TABARENA_SMOKE=1 .venv/bin/python -m pytest \
  test/examples/test_tabiclv2_tabarena.py::test_real_tabarena_smoke -q
```

The test runs both modes on the three datasets from the GPU smoke command: one
binary, multiclass, and regression task. It is not the complete TabArena suite.
