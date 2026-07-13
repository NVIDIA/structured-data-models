# TabICLv2 processing benchmarks

The suite measures SDM Processor and Recipe overhead without timing dataset
creation or correctness assertions. All processing currently runs on CPU; the
GPU name is retained in result metadata because the same host runs the pinned
model parity checks.

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

Run one-factor-at-a-time and combined stress scenarios:

```bash
python -m benchmark.tabiclv2_processing \
  --matrix oat --repetitions 5 --include-processors
```

Run the final Cartesian product of task, size, and all six binary data
characteristics. Detailed stage and Processor timings are retained for the
one-factor cases; remaining combinations measure total Recipe overhead:

```bash
python -m benchmark.tabiclv2_processing \
  --matrix full --repetitions 3 --include-processors \
  --output benchmark/results/tabiclv2_processing_full.json
```

Each run writes JSON plus a flattened CSV with median, p95, standard
deviation, device, dtype, correctness, and peak CUDA allocation when
available. CPU peak memory is left null to avoid perturbing the timed region
with an allocation sampler.

The pinned open-source reference can be measured for the same explicit
single Identity member when the `tabicl` checkout at the documented commit is
installed:

```bash
python -m benchmark.tabiclv2_reference \
  --size large --repetitions 10
```
