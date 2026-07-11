# cuGraph relational sampler benchmark

This report compares the host-memory `RelationalSampler` plus `pyg-lib` with
the CUDA-primary `CuGraphRelationalSampler`. It retains the baseline measured
at `99b5765`, then measures the cached single-integer-key CUDA searchsorted
fast path added in `3ae4139` under the same workload and GPU environment.

## Workload and methodology

- **Dataset/task:** RelBench `rel-arxiv` / `paper-citation`, `test` split.
  The database has 2,733,846 rows across six tables and six relationships.
- **Inputs:** The benchmark follows `examples/kumorfm.py`: it creates
  `TableTensor` inputs for every database table, registers every RelBench
  foreign-key relationship, and links task `Paper_ID` rows to `papers`.
  All four timestamped tables use their `Submission_Date`; task `date` is the
  cutoff.
- **Request construction:** Stable task positions are selected with Python's
  `random.Random(20260711)`. Each mode receives the same selected task rows,
  table schema, relationships, fanouts, and temporal cutoff. The comparison
  JSON rejects mismatched workload contracts, random-state seeds, or task-row
  selection hashes.
- **Randomness:** Finite fanout runs use advancing random streams. The host
  path seeds PyTorch once per benchmark invocation; cuGraph receives
  `random_state=20260711` and advances the sampler generator per dispatch.
  Finite output rows are therefore checked by invariants, not by equality.
- **Timing:** Sampler topology construction is measured separately. Requests
  have three warmups and ten measured repetitions. CPU latencies use
  `perf_counter_ns`. CUDA requests synchronize before and after each request;
  CUDA events measure join, sampling, temporal top-k, and output assembly.
  `synchronization_other_ms` is the synchronized wall-clock residual.
- **Invariants:** Every task seed must be retained, timestamped sampled rows
  must not exceed the task cutoff, and a full canonical per-table identity
  digest is recorded. An exhaustive one-hop `[-1]` temporal request with eight
  task rows produced exactly matching CPU and CUDA output digests: 143 rows,
  zero cutoff violations.

Finite fanout samples are intentionally not compared row-for-row: the two
implementations use different seeded random algorithms. This is why their
two-hop sampled row medians may differ slightly.

## Environment

The GPU was a Tesla T4 (15,636,037,632 bytes, capability 7.5) on NVIDIA driver
580.159.03.

| Mode                 | Runtime                                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Host CPU + `pyg-lib` | Python 3.10.12, PyTorch 2.12.1+cu130, `pyg-lib` 0.7.0+pt212cu130, RelBench 2.1.2, PyArrow 24.0.0                                                                    |
| CUDA cuGraph         | `sdm-rapids-cu13:torch`; Python 3.11.15, PyTorch 2.14.0.dev20260710+cu130, cuDF 26.08.00a882, pylibcugraph 26.08.00a39, CuPy 14.1.1, RelBench 2.1.2, PyArrow 24.0.0 |

The CUDA run constructs tables from RelBench pandas data and transfers SDM
tensors with `.to("cuda")`; it does not call `StringTensor.from_cudf`. This
avoids the pre-existing cuDF 26.08 private `StringColumn.children` incompatibility
in that separate conversion path; the benchmark does not fix or exercise it.

## Initialization

From the primary small-batch suite, host sampler CSC setup took **439.7 ms**.
CUDA data transfer took **86.8 ms** (excluded from sampler initialization), and
CUDA topology construction took **907.4 ms**. Reuse the initialized sampler for
request workloads; topology build is not included in the request tables below.

## Steady-state results

Each cell is median / p95 milliseconds. `x` is CPU median divided by CUDA
median; values above one favor CUDA. Row counts are total related-table rows
per request at the median.

| Fanout     | Batch |          CPU ms |          CUDA ms | CPU/CUDA x |   CPU/CUDA rows |
| ---------- | ----: | --------------: | ---------------: | ---------: | --------------: |
| `[16]`     |     1 | 14.655 / 17.525 |  24.101 / 24.724 |       0.61 |           3 / 3 |
| `[16]`     |   128 | 16.064 / 17.606 |  26.701 / 27.160 |       0.60 |   1,662 / 1,662 |
| `[16]`     | 1,024 | 18.163 / 20.298 |  29.812 / 30.854 |       0.61 | 14,049 / 14,049 |
| `[16, 16]` |     1 | 13.833 / 15.501 |  45.185 / 64.102 |       0.31 |           5 / 5 |
| `[16, 16]` |   128 | 15.933 / 17.393 |  42.527 / 71.924 |       0.37 |   3,187 / 3,302 |
| `[16, 16]` | 1,024 | 24.850 / 28.386 | 56.705 / 114.517 |       0.44 | 27,017 / 28,023 |

The one-hop crossover is visible only after the request is sufficiently
amortized:

| Fanout     |  Batch |           CPU ms |           CUDA ms | CPU/CUDA x |     CPU/CUDA rows |
| ---------- | -----: | ---------------: | ----------------: | ---------: | ----------------: |
| `[16]`     |  4,096 |  30.158 / 34.150 |   42.820 / 43.905 |       0.70 |   53,852 / 53,852 |
| `[16]`     |  8,192 | 39.260 / 123.045 |   38.291 / 54.013 |       1.03 | 107,642 / 107,642 |
| `[16]`     | 16,384 |  50.179 / 52.325 |   49.782 / 51.692 |       1.01 | 216,056 / 216,056 |
| `[16]`     | 32,768 | 87.906 / 162.082 |   66.547 / 70.046 |   **1.32** | 432,040 / 432,041 |
| `[16, 16]` |  4,096 |  55.382 / 67.582 |  83.256 / 109.479 |       0.67 | 103,385 / 107,395 |
| `[16, 16]` |  8,192 | 86.426 / 195.734 | 125.121 / 153.055 |       0.69 | 206,621 / 214,641 |

At batch 32,768 the CUDA mode sustains 492,404 task rows/s versus 372,760 for
the host mode. The 8,192 and 16,384-row median differences are within
run-to-run spread; 32,768 is the first clear CUDA advantage in this dataset
and environment.

## CUDA time breakdown

`neighbor sampling` excludes temporal top-k so components do not overlap.

| Fanout / batch     |  Join | Neighbor sampling | Temporal top-k | Assembly | Sync/other |  Total |
| ------------------ | ----: | ----------------: | -------------: | -------: | ---------: | -----: |
| `[16]` / 1         | 8.035 |            13.723 |          0.746 |    1.129 |      0.436 | 24.101 |
| `[16]` / 1,024     | 8.829 |            17.485 |          0.970 |    1.917 |      0.466 | 29.812 |
| `[16, 16]` / 1,024 | 9.067 |            42.373 |          1.991 |    2.671 |      0.460 | 56.705 |
| `[16]` / 32,768    | 8.901 |            49.055 |          5.981 |    2.022 |      0.469 | 66.547 |

The persistent cuDF task-to-seed merge is about 8--9 ms across request sizes.
For two hops, temporal neighbor sampling accounts for roughly three quarters
of request time. Output assembly and synchronization residual are small.

## Post-optimization result

Commit `3ae4139` caches a sorted lookup for a table's single integer task key
and resolves each later request with CUDA `searchsorted`. Composite, string,
and mixed-dtype keys retain the cuDF merge fallback. The cache is built lazily
during the benchmark warmups and reused; these remain steady-state serving
measurements rather than first-request latency.

The optimized run used the same T4, software environment, RelBench cache,
task-row positions, seed, fanouts, three warmups, and ten repetitions. All
invariants remained valid. The exhaustive batch-8 digest still matched the
CPU result exactly with 143 rows and zero temporal cutoff violations.

| Fanout     | Batch | CPU median ms | Baseline CUDA ms | Optimized CUDA ms | CPU/optimized x |
| ---------- | ----: | ------------: | ---------------: | ----------------: | --------------: |
| `[16]`     |     1 |        14.655 |           24.101 |            16.757 |            0.87 |
| `[16]`     |   128 |        16.064 |           26.701 |            18.250 |            0.88 |
| `[16]`     | 1,024 |        18.163 |           29.812 |            21.141 |            0.86 |
| `[16]`     | 4,096 |        30.158 |           42.820 |            24.065 |        **1.25** |
| `[16]`     | 8,192 |        39.260 |           38.291 |            28.955 |        **1.36** |
| `[16, 16]` | 1,024 |        24.850 |           56.705 |            47.524 |            0.52 |
| `[16, 16]` | 4,096 |        55.382 |           83.256 |            71.685 |            0.77 |
| `[16, 16]` | 8,192 |        86.426 |          125.121 |           114.702 |            0.75 |

The task lookup median fell from 8--9 ms to 0.44--0.55 ms, roughly a 94%
reduction. At batch 1,024, one-hop total latency fell 29% and two-hop latency
fell 16%. The clear one-hop crossover moved from batch 32,768 in the baseline
to batch 4,096: the optimized CUDA path is 1.25x faster there and 1.36x faster
at batch 8,192. Small one-hop requests are substantially closer but remain
12--16% slower than the host path.

Two-hop CUDA remains 29--91% slower across the measured sizes. Its batch-1,024
profile spends 41.7 ms in neighbor sampling versus 0.52 ms in task lookup, so
further lookup tuning cannot close that gap. The next useful optimization must
reduce the temporal gather/materialization and the per-hop cuGraph dispatch
cost.

## Interpretation and follow-ups

The cached lookup addresses the dominant fixed request overhead and makes CUDA
advantageous for medium and large one-hop requests. The current T4
implementation is still not a latency win for small batches or the two-hop
temporal workload. The remaining profile supports these follow-ups:

1. Fuse temporal sampling output handling with top-k selection to reduce the
   repeated materialization and sort work per hop.
2. Batch/fuse two-hop temporal C API dispatches and retain frontier metadata in
   a compact device representation; the second dispatch dominates the
   two-hop gap.
3. Consider a size-aware policy that keeps small requests on the host path and
   uses CUDA for one-hop batches at or above the measured crossover. Recheck
   that threshold on the production GPU and workload rather than hard-coding
   the T4 result.

## Reproduction

The full commands, row-selection hashes, environments, timing samples,
invariants, and phase summaries are retained under `benchmarks/results/`.
The primary raw files are:

- `rel_arxiv_cpu_pyg_lib_t4.json`, `rel_arxiv_cuda_cugraph_t4.json`, and
  `rel_arxiv_comparison_t4.json` for small requests and exhaustive parity.
- `rel_arxiv_*_large.json` for batches 4,096 and 8,192.
- `rel_arxiv_*_16384.json` and `rel_arxiv_*_32768.json` for crossover probes.
- `rel_arxiv_cuda_cugraph_t4_optimized.json` and the two optimized comparison
  files for the post-`3ae4139` matrix through batch 8,192.
- `../../benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_timeline.html` and
  `../../benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_trace.json` for the
  baseline self-contained phase timeline and its compact
  Chrome-trace-compatible data. The corresponding `_optimized` artifacts
  contain the post-optimization profile.

Host baseline setup:

```bash
/home/ubuntu/work/kumo/venv/bin/python -m pip install \
  pyg_lib -f https://data.pyg.org/whl/torch-2.12.0+cu130.html
/home/ubuntu/work/kumo/venv/bin/python -m pip install relbench -e .
RELBENCH_CACHE_DIR="$PWD/.cache/relbench" \
  /home/ubuntu/work/kumo/venv/bin/python \
  benchmarks/relational_sampler.py run --mode cpu \
  --dataset rel-arxiv --task paper-citation --split test \
  --batch-size 1,128,1024 --fanout 16 --fanout 16,16 \
  --include-exhaustive --parity-batch-size 8 --warmups 3 --repetitions 10 \
  --seed 20260711 --output benchmarks/results/rel_arxiv_cpu_pyg_lib_t4.json
```

CUDA setup and primary run:

```bash
chmod -R a+rX "$PWD/.cache/relbench"
chmod a+rwX benchmarks/results benchmarks/artifacts
docker run --rm --gpus all --entrypoint bash \
  -v "$PWD":/workspace/structured-data-models \
  -v "$PWD/.cache/relbench":/cache/relbench \
  -w /workspace/structured-data-models sdm-rapids-cu13:torch -lc '
    export PIP_TARGET=/tmp/relbench-deps
    pip install -q --target "$PIP_TARGET" --no-deps \
      relbench==2.1.2 pooch duckdb datasets multiprocess dill xxhash \
      huggingface-hub
    export PYTHONPATH=/workspace/structured-data-models:$PIP_TARGET
    RELBENCH_CACHE_DIR=/cache/relbench python benchmarks/relational_sampler.py \
      run --mode cuda --dataset rel-arxiv --task paper-citation --split test \
      --batch-size 1,128,1024 --fanout 16 --fanout 16,16 \
      --include-exhaustive --parity-batch-size 8 --warmups 3 --repetitions 10 \
      --seed 20260711 --output benchmarks/results/rel_arxiv_cuda_cugraph_t4.json \
      --timeline-html benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_timeline.html \
      --timeline-trace benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_trace.json'
python benchmarks/relational_sampler.py compare \
  --cpu benchmarks/results/rel_arxiv_cpu_pyg_lib_t4.json \
  --cuda benchmarks/results/rel_arxiv_cuda_cugraph_t4.json \
  --output benchmarks/results/rel_arxiv_comparison_t4.json
```

Use the same commands with `--batch-size 4096,8192`, `16384`, or `32768`
and the corresponding result names to reproduce the amortization probes.
For the optimized matrix, run the CUDA command from `3ae4139` or later with
`--batch-size 1,128,1024,4096,8192` and write the result and timeline names
with the `_optimized` suffix. Compare that result once against the primary CPU
file and once against the `large` CPU file to reproduce the checked-in
optimized comparison JSON.
