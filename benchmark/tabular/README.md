# Tabular Benchmarks

This directory contains benchmarks of `structured-data-models` on TabArena/BeyondArena, [ScoringBench](scoringbench/), and [TALENT](talent/).

## TabArena and BeyondArena setup

Run the commands below from the repository root:

```bash
pip install . \
  "autogluon.common @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=common" \
  "autogluon.core @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=core" \
  "autogluon.features @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=features" \
  "autogluon.tabular @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=tabular" \
  "bencheval @ git+https://github.com/autogluon/tabarena.git@f64c3742f2cb1b734ecbfa6b429cba76afec2c73#subdirectory=packages/bencheval" \
  "tabarena[data-foundry,plot] @ git+https://github.com/autogluon/tabarena.git@f64c3742f2cb1b734ecbfa6b429cba76afec2c73#subdirectory=packages/tabarena"
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

- **`KumoTabular` (small):**

  ```bash
  python -m benchmark.tabular.tabarena.main --model kumo-small
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

### Fine-tuning

`tabiclv2-ft`, `kumo-small`, and `kumo-small-ft` full fine-tune every parameter of the model on each dataset's training split before evaluating (see `benchmark/tabular/finetune.py`). `--model kumo-small-ft`/`--model tabiclv2-ft` alone is enough to get benchmark-quality fine-tuning; tune it further with `--finetune_epochs`, `--finetune_iters_per_epoch`, `--finetune_lr`, `--finetune_train_size`, `--finetune_context_frac`, and `--finetune_val_frac`:

```bash
python -m benchmark.tabular.tabarena.main \
  --model tabiclv2-ft \
  --finetune_epochs 75 \
  --finetune_lr 1e-6
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

- **`KumoTabular` (small):**

  ```bash
  python -m benchmark.tabular.beyondarena.main --model kumo-small
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

### Fine-tuning

Same `-ft` model variants and `--finetune_*` flags as TabArena above:

```bash
python -m benchmark.tabular.beyondarena.main --model kumo-small-ft --subset lite
```

### Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.tabular.beyondarena.evaluate
```
