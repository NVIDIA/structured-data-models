# TabICLv2 processing benchmarks

The suite measures SDM Processor and Recipe overhead without timing dataset
creation or correctness assertions. It supports CPU and CUDA, records
synchronized wall-clock latency, and records incremental peak CUDA allocation.

Dataset sizes are:

| Name   |   Rows | Features |
| ------ | -----: | -------: |
| Tiny   |    300 |       10 |
| Small  |  1,000 |       50 |
| Medium | 10,000 |      100 |
| Large  | 50,000 |      100 |

Run a quick validation:

```bash
python -m benchmark.tabiclv2_processing \
  --matrix smoke --sizes tiny --repetitions 5 --include-processors
```

Run the representative all-path 50,000 × 100 workload on CPU and GPU. Power
and Quantile are alternatives in TabICLv2 planning, so the command materializes
one explicit Recipe for each:

```bash
python -m benchmark.tabiclv2_processing \
  --sizes large --matrix large-stress \
  --devices cpu cuda --recipe-variants power quantile \
  --repetitions 5 --include-processors \
  --output benchmark/results/tabiclv2_processing_large_stress.json
```

The stress workload combines constant columns, numerical and categorical
missing values, unseen query categories, 4,096-value categorical vocabularies,
fixed-Clip outliers, and sigma-based outliers.

Run one-factor-at-a-time and combined scenarios:

```bash
python -m benchmark.tabiclv2_processing \
  --matrix oat --repetitions 5 --include-processors
```

Run the seven-factor Cartesian product. Detailed stage and Processor timings
are retained for one-factor cases; remaining combinations measure total Recipe
overhead:

```bash
python -m benchmark.tabiclv2_processing \
  --matrix full --repetitions 3 --include-processors \
  --output benchmark/results/tabiclv2_processing_full.json
```

Each run writes JSON plus a flattened CSV with median, p95, standard
deviation, device, GPU model, dtype, correctness, repetitions, and peak CUDA
allocation. CPU peak memory remains null to avoid perturbing timed operations.

Measure the pinned open-source reference on the equivalent mixed stress data:

```bash
PYTHONPATH=/tmp/tabicl-v2-ref/src \
python -m benchmark.tabiclv2_reference \
  --size large --stress --normalization-methods power quantile \
  --repetitions 5 \
  --output benchmark/results/tabiclv2_reference_large_stress.json
```

The reference checkout must be at
`f719c886a586ed4a29236345e319ac1ea596c478`.

Establish the GPU speed of light for representative Processor operations:

```bash
python -m benchmark.processor_speed_of_light \
  --warmups 5 --repetitions 20 \
  --output benchmark/results/processor_gpu_speed_of_light.json
```

This CUDA-only run compares the current Processors with direct tensor, eager
batched, compiled, layout, batching, algorithm, and dtype candidates. It records host
wall and CUDA-event medians/p95s, incremental peak allocation, throughput,
correctness error, pinned H2D/D2H copies, and idle synchronization overhead.
Compilation, data creation, fitted-state preparation, and correctness checks
remain outside timed regions. See `REPORT.md` for the current timing
summary, speed-of-light tables, and production recommendation.
