# Tabular Benchmarks

This directory contains benchmarks of `structured-data-models` on TabArena/BeyondArena, [ScoringBench](scoringbench/), and [TALENT](talent/).

## TabArena and BeyondArena setup

Run the commands below from the repository root:

```bash
pip install structured-data-models \
  --pre "autogluon.tabular>=1.6.3b20260917,!=1.6.3,<1.7" \
  "bencheval @ git+https://github.com/autogluon/tabarena.git@90c07f317eec9991409398194ddee070a96b8bf1#subdirectory=packages/bencheval" \
  "tabarena[data-foundry,plot] @ git+https://github.com/autogluon/tabarena.git@90c07f317eec9991409398194ddee070a96b8bf1#subdirectory=packages/tabarena"
```

`tabarena` at this commit (2026-09-21) needs an AutoGluon pre-release for its shared-weights API. Weights of the hosted foundation models are resolved locally when `HF_HOME` (Hugging Face hub cache) and `TABPFN_MODEL_CACHE_DIR` (TabPFN checkpoints) point at populated caches; the TabArena data cache is `TABARENA_CACHE` or `--cache_root`.

### Measurement protocol

Every run starts with an untimed warm-up (imports, CUDA context, a one-member dummy fit); `time_train_s` and `time_infer_s` bracket only the fit and the prediction. `KumoTabular` declares its network loader as AutoGluon shared weights, so the checkpoint is read once per process and never inside the timed fit, like the hosted foundation models.

`--registry_model NAME` (e.g. `TabPFN-3.5`, `LimiX-2`, `TabICLv2`) benchmarks an upstream tabarena model with its own wrapper under the same runner, so its times are measured exactly like the SDM models'. Every fit gets one GPU, as the hosted runs did.

`--validation outer` (default) fits once on all training rows. `--validation official` runs the arena's bagged protocol, eight fold fits and one refit on all rows, which is how every hosted method is measured: its train time is that of nine fits, and its result directory is `official_model/`. Score such runs with `evaluate.py --validation official`: they enter the pool as a config method named after the model's registry key (`SDM-KUMO-TABULAR (default)` in the CSV, the `=LABEL` on the website table), so score two bagged runs of one model in separate pools.

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

Run a local checkpoint under a run name, with every cache under one directory (`--size small` for a 128-channel checkpoint, `--num_estimators`, `--numerical_missing {nan,dispatch,mix,impute}`, `--regression_reduction {scalar_trim,quantile_trim}` and `--max_context_size` change the inference setup; `--validation official` runs the bagged protocol):

```bash
python -m benchmark.tabular.tabarena.main \
  --model kumo-tabular \
  --checkpoint /path/to/final.pt \
  --name my-run \
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
