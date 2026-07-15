# cuGraph relational sampler benchmark

This report compares the host-memory `RelationalSampler` plus `pyg-lib` with
the CUDA-primary `CuGraphRelationalSampler` on RelBench. All retained CPU and
CUDA results use the same stable Python and package environment.

## Verdict

- The refresh was required. The previous evidence mixed PyTorch 2.12 CPU
  results with a PyTorch nightly and RAPIDS 26.08 alpha CUDA environment.
  The comparison command now rejects mismatched runtime versions.
- For one-hop `[16]`, CPU is faster through batch 128, batches 1,024 and 4,096
  are too close to treat as a durable backend threshold, and CUDA is 1.22x
  faster at batch 8,192 in this run.
- For two-hop `[16, 16]`, CPU remains faster at every measured batch. CUDA
  achieves 0.47--0.94x CPU sampled-row throughput after accounting for its
  2--4% larger finite outputs.
- Headline CUDA latency assumes both input and returned tables stay on the
  GPU. Materializing the full sampled output on this host took 220--229 ms
  median and is excluded. A host-consuming caller should not use the headline
  speedups.
- One-hop and two-hop exhaustive outputs match exactly: 143 and 277 rows,
  respectively, with identical canonical digests and no temporal violations.

## Data flow

```text
RelBench pandas tables
  -> TableTensor + RelationalData
  -> CPU: pyg-lib CSC topology -> public RelationalSampler call
  -> CUDA: one data transfer -> persistent pylibcugraph.SGGraph
           -> cached numeric task lookup (cuDF fallback for general keys)
           -> public CuGraphRelationalSampler call
  -> GPU-resident RelationalSamplerOutput (headline latency ends here)
  -> optional host materialization
  -> seed, cutoff, multiplicity, fingerprint, and parity checks
  -> raw JSON summaries and aggregate HTML phase profile
```

## Methodology

- **Workload:** RelBench `rel-arxiv` / `paper-citation`, `test` split. The
  database has 2,733,846 rows across six tables and six relationships. Four
  tables use `Submission_Date`; task `date` is the original cutoff at every
  hop.
- **Matched inputs:** Both modes use the same selected task positions, seed,
  fanout, schemas, relationships, and SHA-256 fingerprints of every
  sampling-relevant ID/time column and task ID/time column. The comparison
  command rejects workload, seed, measurement-count, or runtime mismatches.
- **Reliability protocol:** Each retained shape has 10 warmups and 50 timed
  requests. An earlier 10-request CPU pass was rejected after batch 1,024
  appeared slower than batch 4,096. Three fresh-process probes restored the
  expected ordering; they were diagnostic only and are not retained as
  reference artifacts. The final 10/50 pass is the only headline evidence.
- **Separate passes:** Public latency, output validation, and CUDA profiling
  are independent passes. Validation samples 50 more outputs, materializes
  each on the host, then runs correctness checks and computes canonical digests.
  CUDA profiling samples another 50 outputs with event instrumentation.
- **Timing boundary:** Public CUDA timing is a synchronized sampler call with
  the task `TableTensor` already resident on the GPU. It includes device-side
  output assembly. Task host-to-device transfer, one-time data transfer,
  topology setup, host output materialization, fingerprinting, and profiler
  overhead are recorded separately or explicitly excluded.
- **Finite outputs:** Finite fanout implementations can differ in random/tie
  ordering. They are compared as the same request shape, not as identical
  rows. Exhaustive requests establish semantic parity.

All latency, row-count, output-digest, host-materialization, validation,
profile, phase, and finite-fanout samples are retained in the raw JSON.
The HTML and trace arrange independent phase medians in data-flow order; they
are aggregate profiles, not single-request execution timelines.

## Environment

Both modes ran from one Python 3.12.13 environment on an Intel Xeon Platinum
8259CL host with 16 physical cores, 32 logical CPUs, and 16 PyTorch intra-op
and inter-op threads. Process affinity allowed all 32 logical CPUs and was not
pinned. The CPU and CUDA runs began with 1/5/15-minute host loads of
16.23/16.39/15.80 and 14.63/15.29/15.55, respectively, so small CPU deltas and
tail latency remain sensitive to shared-host contention.

The GPU was a Tesla T4 (15,636,037,632 bytes, capability 7.5) on NVIDIA driver
580.159.03.

| Component            | Version                               |
| -------------------- | ------------------------------------- |
| Python               | 3.12.13                               |
| PyTorch / CUDA build | 2.11.0+cu130 / 13.0                   |
| cuDF                 | 26.06.00 (`cudf-cu13` 26.6.0)         |
| pylibcugraph         | 26.06.00 (`pylibcugraph-cu13` 26.6.0) |
| pyg-lib              | 0.7.0+pt211cu130                      |
| CuPy                 | 14.1.1                                |
| NumPy / pandas       | 2.4.6 / 2.3.3                         |
| PyArrow              | 23.0.1                                |
| RelBench             | 2.1.2                                 |

## Public latency

Each cell is median / p95 milliseconds from 50 requests. `x` is CPU median
divided by CUDA median; values above one favor CUDA.

| Fanout     | Batch |            CPU ms |           CUDA ms | CPU/CUDA x |     CPU/CUDA rows |
| ---------- | ----: | ----------------: | ----------------: | ---------: | ----------------: |
| `[16]`     |     1 |   14.886 / 18.806 |   16.146 / 17.882 |       0.92 |             3 / 3 |
| `[16]`     |   128 |   14.245 / 15.404 |   17.592 / 19.724 |       0.81 |     1,662 / 1,662 |
| `[16]`     | 1,024 |   19.212 / 21.158 |   19.969 / 20.500 |       0.96 |   14,049 / 14,049 |
| `[16]`     | 4,096 |   24.785 / 26.616 |   23.002 / 25.428 |       1.08 |   53,852 / 53,852 |
| `[16]`     | 8,192 |   34.042 / 36.461 |   27.797 / 28.269 |   **1.22** | 107,642 / 107,642 |
| `[16, 16]` |     1 |   13.885 / 15.883 |   29.429 / 30.930 |       0.47 |             5 / 5 |
| `[16, 16]` |   128 |   16.989 / 20.998 |   33.391 / 34.595 |       0.51 |     3,187 / 3,302 |
| `[16, 16]` | 1,024 |   29.378 / 34.893 |   45.560 / 47.352 |       0.64 |   27,017 / 28,023 |
| `[16, 16]` | 4,096 |   54.130 / 58.469 |   71.500 / 75.108 |       0.76 | 103,385 / 107,395 |
| `[16, 16]` | 8,192 | 102.513 / 108.410 | 113.508 / 117.636 |       0.90 | 206,621 / 214,641 |

The one-hop 1,024 and 4,096 differences are 4% and 8%, smaller than is prudent
to operationalize from one shared host. Re-measure any dispatch threshold in
the target service environment.

## Correctness

| Fanout     | Batch | Rows | CPU ms | CUDA ms | Exact digest | Cutoff violations |
| ---------- | ----: | ---: | -----: | ------: | :----------: | ----------------: |
| `[-1]`     |     8 |  143 | 13.490 |  14.516 |     yes      |                 0 |
| `[-1, -1]` |     8 |  277 | 14.591 |  26.201 |     yes      |                 0 |

All 50 validation outputs in each exhaustive run have the same digest. The
one-hop digest is `f6bc02aa...b9c635`; the two-hop digest is
`8bd664af...5c330d`.

Every finite and exhaustive validation output also had zero rows exceeding
the source table's identity-key multiplicity. This distinguishes legitimate
duplicate key tuples already present in the source from sampler-created row
inflation. During the CUDA profile pass, temporal latest-k checks observed up
to 9,947,900 per-example/source/edge-type groups for a shape, selected at most
16 neighbors, and found zero fanout violations. This check does not expose
non-temporal fanout decisions inside cuGraph or pyg-lib's internal CPU groups;
focused sampler tests cover a small observable finite-fanout case.

The finite two-hop CUDA row surplus is therefore not duplicate inflation. It
is consistent with backend random/tie selection differences; exhaustive
parity confirms the same temporal boundary and reachable rows.

## Excluded costs

- Host CSC topology setup: **419.5 ms**.
- Relational data host-to-device transfer: **150.3 ms**.
- Persistent cuGraph topology setup after transfer: **455.9 ms**.
- Per-shape task host-to-device observation: **0.37--0.55 ms**. This is one
  observation per shape, not a latency distribution.
- Returned-output device-to-host materialization: **219.8--228.6 ms median**.
- Canonical validation/fingerprinting after host materialization:
  **3.8--1,000.0 ms median**, scaling with sampled output size.

The large output transfer is why the public-latency table applies only when a
downstream consumer remains on the GPU. Setup costs require sampler reuse.

## CUDA profile

Values are independent medians from the 50-request profiled pass. `Neighbor`
excludes temporal top-k.

| Fanout / batch     | Lookup | Neighbor | Top-k | Assembly | Other | Profiled total |
| ------------------ | -----: | -------: | ----: | -------: | ----: | -------------: |
| `[16]` / 1         |  0.332 |   13.979 | 0.729 |    1.133 | 0.370 |         16.541 |
| `[16]` / 1,024     |  0.308 |   16.636 | 0.920 |    1.836 | 0.351 |         20.078 |
| `[16, 16]` / 1,024 |  0.344 |   40.377 | 1.877 |    2.588 | 0.386 |         45.574 |
| `[16, 16]` / 8,192 |  0.363 |  107.167 | 3.551 |    2.724 | 0.402 |        114.221 |

The difference between independent profiled and public medians ranges from
-0.53 to +1.09 ms. Negative values can occur because these are separate
passes, not paired measurements. The task-key lookup is small; two-hop cost
is dominated by temporal neighbor gathering.

## Opportunities and deferred work

The lowest-cost optimization opportunity is avoiding host materialization
when the next stage can consume `RelationalSamplerOutput` on CUDA. For the
sampler itself, fewer per-hop cuGraph dispatches or a native grouped latest-k
temporal primitive are more promising than further task-key lookup tuning.

The following are deliberately deferred from this PR:

- CPU phase attribution inside the pyg-lib call and a CPU thread/affinity
  sweep.
- Peak CPU/GPU memory measurement, including non-PyTorch RAPIDS allocations.
- A second dataset, GPU architecture, or concurrency study.

## Artifacts

- `benchmarks/results/rel_arxiv_{cpu_pyg_lib,cuda_cugraph}_t4_final.json`:
  final 10/50 raw runs, fingerprints, timing boundaries, and correctness checks.
- `benchmarks/results/rel_arxiv_comparison_t4_final.json`: matched runtime,
  request latency, sampled-row throughput, and one-hop parity comparison.
- `benchmarks/results/rel_arxiv_*_two_hop_exhaustive.json` and
  `rel_arxiv_comparison_t4_two_hop_exhaustive.json`: two-hop parity evidence.
- `benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_final_profile.html` and
  `_trace.json`: self-contained aggregate phase profile and
  Chrome-trace-compatible representation.
- `benchmarks/artifacts/cugraph_relational_sampler_benchmark_walkthrough.html`:
  self-contained explanation of the timing boundary, fairness checks, results,
  review findings, and practical follow-up work.

Obsolete nightly/alpha baselines and diagnostic 10-request runs are not
retained.

## Reproduction

Create the stable environment used by both modes:

```bash
uv venv --python 3.12.13 .venv
uv pip install --python .venv/bin/python \
  --torch-backend=cu130 \
  --extra-index-url=https://pypi.nvidia.com \
  --index-strategy=unsafe-best-match \
  'torch==2.11.0' '.[test]' 'relbench==2.1.2' \
  'cudf-cu13==26.6.0' 'pylibcugraph-cu13==26.6.0' \
  'cupy-cuda13x==14.1.1' 'numpy==2.4.6' 'pandas==2.3.3' \
  'pyarrow==23.0.1'
uv pip install --python .venv/bin/python \
  --find-links=https://data.pyg.org/whl/torch-2.11.0+cu130.html \
  'pyg-lib==0.7.0+pt211cu130'
```

Run the final CPU and CUDA matrix:

```bash
export RELBENCH_CACHE_DIR="$PWD/.cache/relbench"

.venv/bin/python benchmarks/relational_sampler.py run --mode cpu \
  --dataset rel-arxiv --task paper-citation --split test \
  --batch-size 1,128,1024,4096,8192 \
  --fanout 16 --fanout 16,16 --include-exhaustive \
  --parity-batch-size 8 --warmups 10 --repetitions 50 --seed 20260711 \
  --output benchmarks/results/rel_arxiv_cpu_pyg_lib_t4_final.json

.venv/bin/python benchmarks/relational_sampler.py run --mode cuda \
  --dataset rel-arxiv --task paper-citation --split test \
  --batch-size 1,128,1024,4096,8192 \
  --fanout 16 --fanout 16,16 --include-exhaustive \
  --parity-batch-size 8 --warmups 10 --repetitions 50 --seed 20260711 \
  --output benchmarks/results/rel_arxiv_cuda_cugraph_t4_final.json \
  --timeline-html \
    benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_final_profile.html \
  --timeline-trace \
    benchmarks/artifacts/rel_arxiv_cuda_cugraph_t4_final_profile_trace.json

.venv/bin/python benchmarks/relational_sampler.py compare \
  --cpu benchmarks/results/rel_arxiv_cpu_pyg_lib_t4_final.json \
  --cuda benchmarks/results/rel_arxiv_cuda_cugraph_t4_final.json \
  --output benchmarks/results/rel_arxiv_comparison_t4_final.json
```

Run each mode again with `--batch-size 8 --fanout=-1,-1 --warmups 10 --repetitions 50` and compare those files to regenerate the separate two-hop
exhaustive evidence.
