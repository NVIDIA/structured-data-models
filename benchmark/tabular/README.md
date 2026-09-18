# Tabular Benchmarks

This directory contains benchmarks of `structured-data-models` on TabArena/BeyondArena, [ScoringBench](scoringbench/), and [TALENT](talent/).

## TabArena and BeyondArena setup

Run the commands below from the repository root:

```bash
pip install structured-data-models \
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

Run a local checkpoint under a run name, with every cache under one directory (`--size small` for a 128-channel checkpoint, `--num_estimators`, `--numerical_missing {nan,dispatch,mix,impute}` and `--max_context_size` change the inference setup):

```bash
python -m benchmark.tabular.tabarena.main \
  --model kumo-tabular \
  --checkpoint /path/to/final.pt \
  --name my-run \
  --output_root /path/to/tabarena_out \
  --cache_root /path/to/cache
```

`--recipe_ensemble` blends several kumo-tabular recipes. Each recipe is fitted on the folds of the training rows (3 by default, stratified for classification), AutoGluon's ensemble selection on the out-of-fold predictions sets the weights, and the weighted members are refit on the full table at prediction time. Recipes are `+`-joined tokens: `round_robin`, `identity`, `power`, `squash` or `quantile` for the numeric transform, `dispatch`, `nan`, `mix` or `impute` to override `--numerical_missing` for that member, and `catshuffle<N>` to permute the codes of categoricals with at most N levels per estimator:

```bash
python -m benchmark.tabular.tabarena.main \
  --model kumo-tabular \
  --checkpoint /path/to/final.pt \
  --recipe_ensemble round_robin,identity,power,squash,quantile,round_robin+catshuffle30,round_robin+impute \
  --name my-blend \
  --output_root /path/to/tabarena_out \
  --cache_root /path/to/cache
```

### Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.tabular.tabarena.evaluate
```

Evaluate named runs of one model instead. Repeat `--name` for one leaderboard row per run; join runs with `,` to score them as one method, for example a classification and a regression run; append `=LABEL` to name that method on the leaderboard:

```bash
python -m benchmark.tabular.tabarena.evaluate --model kumo-tabular --name my-cls-run,my-reg-run=Kumo-Tabular-L --output_root /path/to/tabarena_out
```

For `kumo-tabular`, numeric columns with two or three distinct values are passed to the model as categorical on tables of more than 150 rows, as the reference estimator does.

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

Run a local checkpoint under a run name, with every cache under one directory:

```bash
python -m benchmark.tabular.beyondarena.main \
  --model kumo-tabular \
  --checkpoint /path/to/final.pt \
  --name my-run \
  --output_root /path/to/beyondarena_out \
  --cache_root /path/to/cache
```

Available subset filters include problem types (`classification`, `regression`), size buckets (`tiny`, `small`, `medium`, `large`), split regimes (`iid`, `temporal`, `grouped`), feature groups (`low-dim`, `high-dim`, `text`, `high-cardinality`), and split selections (`core`, `lite`, `all`). Prefix a filter with `!` to negate it.

### Evaluate

Evaluate all available model results with:

```bash
python -m benchmark.tabular.beyondarena.evaluate
```

Evaluate named runs of one model instead. Repeat `--name` for one leaderboard row per run; join runs with `,` to score them as one method, for example a classification and a regression run; append `=LABEL` to name that method on the leaderboard:

```bash
python -m benchmark.tabular.beyondarena.evaluate --model kumo-tabular --name my-cls-run,my-reg-run=Kumo-Tabular-L --output_root /path/to/beyondarena_out
```

`--backend ray` processes the raw results in parallel; on the 138-split regression pool it took 45 s where the default in-process path took 17 min, with identical output.
