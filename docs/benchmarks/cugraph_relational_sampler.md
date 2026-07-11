# cuGraph relational sampler benchmark

This report compares the host-memory `RelationalSampler` plus `pyg-lib` with
the CUDA-primary `CuGraphRelationalSampler` on RelBench. It also records the
pre-optimization profile that led to the cached numeric task-key lookup in
`3ae4139`, including the mutation invalidation added in `12b41ed`.

## Verdict

- For one-hop `[16]` sampling, CUDA has a narrow 1.06--1.09x median advantage
  at batches 1--1,024 and a clearer 1.15x and 1.58x advantage at batches 4,096
  and 8,192. Treat the small-batch difference as inconclusive across process
  runs; the two paths use different supported runtime builds.
- Finite two-hop `[16, 16]` CUDA sampling remains 28--63% slower by request
  median. It is also 19--36% slower after normalizing by sampled rows, so the
  conclusion is not explained by its 3--4% larger sampled outputs alone.
- Exact one-hop and two-hop exhaustive requests match CPU output digests: 143
  and 277 rows respectively, with zero temporal cutoff violations.
- The task-key optimization reduced the profiled lookup phase from 8--9 ms to
  0.27--0.33 ms. The remaining two-hop cost is temporal neighbor gathering,
  not task-to-seed resolution.

## Data flow

```text
RelBench pandas tables
  -> TableTensor + RelationalData
  -> CPU: pyg-lib CSC topology -> public RelationalSampler call
  -> CUDA: one data transfer -> persistent pylibcugraph.SGGraph
           -> cached numeric task lookup (cuDF fallback for general keys)
           -> public CuGraphRelationalSampler call
  -> RelationalSamplerOutput
  -> seed retention + temporal cutoff + canonical identity digest checks
  -> raw JSON summaries and aggregate HTML phase profile
```

## Methodology

- **Workload:** RelBench `rel-arxiv` / `paper-citation`, `test` split. The
  database has 2,733,846 rows across six tables and six relationships. Four
  tables use `Submission_Date`; task `date` is the original cutoff at every
  hop.
- **Matched inputs:** Both modes use the same selected task positions, seed,
  fanout, warmups, repetitions, schemas, relationships, and normalized SHA-256
  fingerprints of every sampling-relevant ID/time column and task ID/time
  column. The comparison command rejects any mismatch.
- **Timing:** Each variant has three warmups and ten repetitions. Public
  sampler calls run back-to-back in the latency pass. A separate untimed pass
  validates all ten outputs. CUDA phase events run in a third pass, so profiler
  allocation and method wrapping are excluded from public latency.
- **Scope:** CUDA timing starts with the task `TableTensor` resident on the
  device. The observed 0.37--0.50 ms task transfer is recorded but excluded.
  Sampler topology initialization and the one-time relational data transfer
  are also reported separately.
- **Evidence:** Every latency, row count, output digest, profiled latency, and
  phase duration is retained in the final raw JSON. The HTML and trace files
  arrange independent phase medians in data-flow order; they are aggregate
  flamegraph-like profiles, not single-request execution timelines.

Finite fanout implementations use different random/tie ordering. They are
compared as the same requested serving shape, not as identical concrete rows.
The separate exhaustive cases establish semantic parity through two hops.

## Environment

Both modes ran on one Intel Xeon Platinum 8259CL host with 16 physical cores,
32 logical CPUs, and 16 PyTorch intra-op / interop threads. CPU affinity was
not pinned. The GPU was a Tesla T4 (15,636,037,632 bytes, capability 7.5) on
driver 580.159.03.

| Mode                 | Runtime                                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Host CPU + `pyg-lib` | Python 3.10.12, PyTorch 2.12.1+cu130, `pyg-lib` 0.7.0+pt212cu130, RelBench 2.1.2, PyArrow 24.0.0                                                                    |
| CUDA cuGraph         | `sdm-rapids-cu13:torch`; Python 3.11.15, PyTorch 2.14.0.dev20260710+cu130, cuDF 26.08.00a882, pylibcugraph 26.08.00a39, CuPy 14.1.1, RelBench 2.1.2, PyArrow 24.0.0 |

This is a deployment-path comparison, not an isolated backend experiment.
The available `pyg-lib` and RAPIDS builds require different Python and
PyTorch versions, so the entire delta cannot be attributed to the sampler
libraries alone. The CUDA run constructs tensors from pandas then calls
`.to("cuda")`; it does not exercise the separate cuDF 26.08
`StringColumn.children` compatibility issue.

## Public latency

Each cell is median / p95 milliseconds. `x` is CPU median divided by CUDA
median; values above one favor CUDA.

| Fanout     | Batch |           CPU ms |           CUDA ms | CPU/CUDA x |     CPU/CUDA rows |
| ---------- | ----: | ---------------: | ----------------: | ---------: | ----------------: |
| `[16]`     |     1 |  15.993 / 18.677 |   14.725 / 14.847 |       1.09 |             3 / 3 |
| `[16]`     |   128 |  17.500 / 18.072 |   16.133 / 16.199 |       1.08 |     1,662 / 1,662 |
| `[16]`     | 1,024 |  19.301 / 22.697 |   18.247 / 18.417 |       1.06 |   14,049 / 14,049 |
| `[16]`     | 4,096 |  24.795 / 25.648 |   21.603 / 22.481 |       1.15 |   53,852 / 53,852 |
| `[16]`     | 8,192 |  41.302 / 43.020 |   26.188 / 26.633 |   **1.58** | 107,642 / 107,642 |
| `[16, 16]` |     1 |  18.339 / 53.857 |   27.235 / 27.392 |       0.67 |             5 / 5 |
| `[16, 16]` |   128 |  18.829 / 19.385 |   30.655 / 30.860 |       0.61 |     3,187 / 3,302 |
| `[16, 16]` | 1,024 |  29.085 / 33.713 |   44.023 / 46.388 |       0.66 |   27,017 / 28,023 |
| `[16, 16]` | 4,096 | 52.941 / 136.829 |   67.610 / 67.827 |       0.78 | 103,385 / 107,395 |
| `[16, 16]` | 8,192 |  72.822 / 79.149 | 110.793 / 111.162 |       0.66 | 206,621 / 214,641 |

Ten samples make p95 sensitive to one slow request; the raw arrays are
included so the CPU tails at two-hop batches 1 and 4,096 are auditable. Median
is the primary comparison. For unequal finite two-hop row counts, CUDA/CPU
sampled-row throughput ratios are 0.64, 0.69, 0.81, and 0.68 at batches 128,
1,024, 4,096, and 8,192.

## Correctness

| Fanout    | Batch | Rows | CPU ms | CUDA ms | Exact digest | Cutoff violations |
| --------- | ----: | ---: | -----: | ------: | :----------: | ----------------: |
| `[-1]`    |     8 |  143 | 13.944 |  13.402 |     yes      |                 0 |
| `[-1,-1]` |     8 |  277 | 27.941 |  25.248 |     yes      |                 0 |

All ten validation repetitions in each exhaustive run have the same digest.
This covers the original-cutoff behavior through two hops; finite output
differences therefore reflect selection/tie behavior rather than a different
temporal boundary.

## CUDA profile

The values below are independent medians from the separate profiled pass.
`neighbor` excludes temporal top-k.

| Fanout / batch     | Lookup | Neighbor | Top-k | Assembly | Other | Profiled total |
| ------------------ | -----: | -------: | ----: | -------: | ----: | -------------: |
| `[16]` / 1         |  0.278 |   12.648 | 0.691 |    1.043 | 0.304 |         14.978 |
| `[16]` / 1,024     |  0.319 |   16.635 | 1.078 |    2.167 | 0.373 |         20.585 |
| `[16, 16]` / 1,024 |  0.299 |   38.204 | 1.779 |    2.461 | 0.314 |         43.067 |
| `[16, 16]` / 8,192 |  0.321 |  104.659 | 3.541 |    2.601 | 0.334 |        111.445 |

The pre-optimization baseline used a repeated cuDF task merge and measured an
8--9 ms lookup phase. The final 0.27--0.33 ms lookup is a 96% reduction under
the same CUDA-event instrumentation. Historical total latency is not directly
mixed with the final public-call table because the old run instrumented the
headline call itself.

## Initialization

Host CSC setup took **408.6 ms**. CUDA relational data transfer took **86.8
ms** and persistent cuGraph topology construction took **867.4 ms**. These
costs are excluded from request latency and must be amortized by sampler reuse.

## Interpretation

The numeric task lookup was the useful fixed-cost optimization and is now
small. The remaining finite two-hop bottleneck is the per-hop temporal gather
and materialization. The next work should target fewer cuGraph dispatches or a
native latest-k temporal primitive; further lookup tuning will not close the
gap.

A size-aware CPU/CUDA policy is only practical when a service already keeps
both topologies resident. Copying a CUDA-primary graph to host per request
would erase any small-batch benefit. Re-measure thresholds on the production
GPU and workload rather than hard-coding this T4 result.

## Artifacts

- `benchmarks/results/rel_arxiv_{cpu_pyg_lib,cuda_cugraph}_t4_final.json`:
  final raw runs with all samples, fingerprints, and invariants.
- `benchmarks/results/rel_arxiv_comparison_t4_final.json`: final request and
  sampled-row throughput comparison.
- `benchmarks/results/rel_arxiv_*_two_hop_exhaustive.json` and the matching
  comparison: two-hop semantic parity evidence.
- `benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_final_profile.html` and
  `_trace.json`: self-contained aggregate phase profile and
  Chrome-trace-compatible representation.
- Files without `final` retain the pre-optimization baseline and scaling
  probes that motivated the lookup change.

## Reproduction

Host run:

```bash
RELBENCH_CACHE_DIR="$PWD/.cache/relbench" \
  /home/ubuntu/work/kumo/venv/bin/python \
  benchmarks/relational_sampler.py run --mode cpu \
  --dataset rel-arxiv --task paper-citation --split test \
  --batch-size 1,128,1024,4096,8192 \
  --fanout 16 --fanout 16,16 --include-exhaustive \
  --parity-batch-size 8 --warmups 3 --repetitions 10 --seed 20260711 \
  --output benchmarks/results/rel_arxiv_cpu_pyg_lib_t4_final.json
```

CUDA run:

```bash
docker run --rm --gpus all --entrypoint bash \
  -v "$PWD":/workspace/structured-data-models \
  -v "$PWD/.cache/relbench":/cache/relbench \
  -w /workspace/structured-data-models sdm-rapids-cu13:torch -lc '
    export PIP_TARGET=/tmp/relbench-deps
    pip install -q --target "$PIP_TARGET" --no-deps \
      relbench==2.1.2 pooch duckdb datasets multiprocess dill xxhash \
      huggingface-hub
    export PYTHONPATH=/workspace/structured-data-models:$PIP_TARGET
    RELBENCH_CACHE_DIR=/cache/relbench \
      python benchmarks/relational_sampler.py run --mode cuda \
      --dataset rel-arxiv --task paper-citation --split test \
      --batch-size 1,128,1024,4096,8192 \
      --fanout 16 --fanout 16,16 --include-exhaustive \
      --parity-batch-size 8 --warmups 3 --repetitions 10 --seed 20260711 \
      --output benchmarks/results/rel_arxiv_cuda_cugraph_t4_final.json \
      --timeline-html \
        benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_final_profile.html \
      --timeline-trace \
        benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_final_profile_trace.json'
```

Run each mode again with `--batch-size 8 --fanout=-1,-1` for two-hop parity,
then compare:

```bash
python benchmarks/relational_sampler.py compare \
  --cpu benchmarks/results/rel_arxiv_cpu_pyg_lib_t4_final.json \
  --cuda benchmarks/results/rel_arxiv_cuda_cugraph_t4_final.json \
  --output benchmarks/results/rel_arxiv_comparison_t4_final.json
```
