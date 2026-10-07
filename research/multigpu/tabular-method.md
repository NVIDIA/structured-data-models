# KumoTabular scaling experiment

This experiment measures public KumoTabular inference and alternative GPU execution strategies on fixed validation data. Each configuration runs in a fresh process and records the source revision, runtime, exact arguments, input hashes, query identities, prediction columns, raw predictions, targets, and measurements in its own output directory.

## Comparisons

- `native`: current public `KumoTabular.fit/predict`. At more than one estimator, SDM normally offloads fitted member caches to CPU. Estimator batch size is a separate tuning parameter.
- `ensemble`: `EnsembleParallel` with one, two, or four resident replicas. Compare the one-replica executor with native to identify effects of cache residency and scheduling; compare two/four replicas with the one-replica executor for GPU strong scaling.
- `adapter`: an experiment-specific factory receiving the parsed arguments and initialized replicas. Query parallelism schedules the same fixed query batches concurrently, while model placement distributes a single model's layers and preserves its cache placement.

Native and executor runs use the same preprocessing seed. Resident executor comparisons additionally use identical member seeds. Native ECOC random draws can differ from independent member streams for more than ten classes; the resident one-GPU executor is the semantics reference for those comparisons.

## Measurement boundaries

Model loading, input transfer, context fitting, warmup, and steady prediction passes have separate synchronized wall-clock timers. Every timer waits for all participating GPUs at both boundaries. Each configuration warms up and runs at least three prediction passes. Pass throughput includes CPU output transfer and ordered concatenation. Serial/ensemble per-batch latency measures `predict` completion, excluding the subsequent CPU output copy; these latency numbers must not be compared directly with query-parallel aggregate throughput.

Dataset disk reads are timed separately. Context fitting includes preprocessing and cache construction. The steady prediction phase includes query preprocessing and final ensemble reduction. Neither a fit-only nor prediction-only number is called end-to-end dataset execution. An optional additional profiler pass exports a Chrome trace and operator table after the unprofiled measurements.

Peak allocated and reserved bytes are reset before fitting and again before warmed prediction. The harness also records fitted cache storage by device, executor cache bytes, process peak RSS, and 200 ms `nvidia-smi` samples. GPU peak allocation includes the input table and one output batch; CPU outputs are collected for correctness checks. The telemetry sample interval can miss brief peaks, so allocator peaks are the primary memory comparison.

## Data and quality

Covertype classification and California Housing regression use deterministic train/validation partitions supplied with dataset manifests. Contexts are nested prefixes from TRAIN. Queries are fixed ordered prefixes from validation, and every comparison retains identical batch boundaries. Validation targets only enter the quality scorer after model prediction.

Classification reports log loss, accuracy, and AUROC aligned to the explicit prediction class columns. Regression saves all predicted quantiles and reports median RMSE/MAE and quantile monotonicity. Parallel results must additionally be compared against the matched one-GPU saved prediction arrays, with maximum/mean deviation and downstream metric differences. Repeated outputs are checked for within-configuration differences.

## Initial queue

Start with Covertype, large model, E4, 1,024 context rows, 2,048 query rows, batch 256: native, resident one GPU, resident two GPUs, resident four GPUs. Follow with E1/E8, small/large models, 4,096/16,384 context rows, and query-batch tuning guided by these measurements. Exercise both real tasks. Strong scaling keeps total work fixed; weak scaling increases independent query work in proportion to device count and is reported separately. Capacity tests that exceed one-GPU memory are feasibility results, not numeric speedups against an absent baseline.

Example (paths depend on the execution host):

```sh
PYTHONPATH=. python research/multigpu/tabular_bench.py \
  --data /data/covertype --output /results/large-e4-c1024-ep2 \
  --task classification --size large --mode ensemble --gpus 2 \
  --estimators 4 --context 1024 --queries 2048 --batch-size 256 \
  --repeats 3
```

No measured performance claims are established by this method document alone.
