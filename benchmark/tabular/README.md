# Tabular Benchmarks

This directory contains benchmarks of `structured-data-models` on [TabArena and BeyondArena](https://tabarena.ai), plus a separate [TALENT integration](talent/).

> [!NOTE]
> Weights of `TabFM` are distributed under the [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TabFM` benchmark, which will download its weights noninteractively.

## TabArena and BeyondArena setup

Run the commands below from the repository root. Install the source revisions of AutoGluon and TabArena used by these benchmarks. The Data Foundry extra downloads BeyondArena datasets on demand:

```bash
pip install structured-data-models \
  "autogluon.common @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=common" \
  "autogluon.core @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=core" \
  "autogluon.features @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=features" \
  "autogluon.tabular @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=tabular" \
  "bencheval @ git+https://github.com/autogluon/tabarena.git@f64c3742f2cb1b734ecbfa6b429cba76afec2c73#subdirectory=packages/bencheval" \
  "tabarena[data-foundry,plot] @ git+https://github.com/autogluon/tabarena.git@f64c3742f2cb1b734ecbfa6b429cba76afec2c73#subdirectory=packages/tabarena"
```

The SDM adapters run as standard AutoGluon models. TabArena uses its native eight-fold bagging protocol and refits one model on all training rows, matching the published model evaluation. BeyondArena fits one model on all training rows through its outer experiment path.

______________________________________________________________________

## TabArena

### Run

- **`TabICLv2`:**

  ```bash
  python -m benchmark.tabular.tabarena.main --model tabiclv2
  ```

- **`KumoTabular`:**

  ```bash
  python -m benchmark.tabular.tabarena.main --model kumo-tabular
  ```

- **`TabFM`:**

  ```bash
  python -m benchmark.tabular.tabarena.main --model tabfm
  ```

Pass a dataset name to run only that TabArena dataset:

```bash
python -m benchmark.tabular.tabarena.main \
  --model kumo-tabular \
  --dataset blood-transfusion-service-center
```

### Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.tabular.tabarena.evaluate
```

______________________________________________________________________

## BeyondArena

### Run

- **`TabICLv2`:**

  ```bash
  python -m benchmark.tabular.beyondarena.main --model tabiclv2
  ```

- **`KumoTabular`:**

  ```bash
  python -m benchmark.tabular.beyondarena.main --model kumo-tabular
  ```

- **`TabFM`:**

  ```bash
  python -m benchmark.tabular.beyondarena.main --model tabfm
  ```

By default, each command evaluates the recommended `core` subset. Repeat `--subset` to combine filters:

```bash
python -m benchmark.tabular.beyondarena.main \
  --model tabiclv2 \
  --subset core \
  --subset grouped
```

Pass a dataset name to run only that BeyondArena dataset. Use `--subset lite` for its first split:

```bash
python -m benchmark.tabular.beyondarena.main \
  --model tabiclv2 \
  --dataset parkinsons_biomedical_voice_measurements \
  --subset lite
```

Available subset filters include problem types (`classification`, `regression`), size buckets (`tiny`, `small`, `medium`, `large`), split regimes (`iid`, `temporal`, `grouped`), feature groups (`low-dim`, `high-dim`, `text`, `high-cardinality`), and split selections (`core`, `lite`, `all`). Prefix a filter with `!` to negate it.

### Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.tabular.beyondarena.evaluate
```
