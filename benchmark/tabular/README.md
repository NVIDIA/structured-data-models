# Tabular Benchmarks

This directory contains benchmarks of `structured-data-models` on TabArena/BeyondArena, [ScoringBench](scoringbench/), and [TALENT](talent/).

## TabArena and BeyondArena setup

Run the commands below from the repository root:

```bash
pip install . \
  "autogluon.tabular>=1.6.4b20260924,<=1.6.4" \
  "tabarena[data-foundry,plot]>=0.1.1.dev20260924104714,<=0.1.1"
```

______________________________________________________________________

> [!NOTE]
> Weights of `TabFM` are distributed under the [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE).
> Review the license before running the `TabFM` benchmark, which will download its weights noninteractively.

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

By default, each command evaluates the recommended `core` subset.
Repeat `--subset` to combine filters:

```bash
python -m benchmark.tabular.beyondarena.main \
  --model tabiclv2 \
  --subset core \
  --subset grouped
```

Pass a dataset name to run only that BeyondArena dataset.
Use `--subset lite` for its first split:

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
