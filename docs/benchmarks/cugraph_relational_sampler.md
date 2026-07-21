# cuGraph relational sampler benchmark

This benchmark compares the host-memory `RelationalSampler` plus `pyg-lib`
with the CUDA-primary `CuGraphRelationalSampler` on RelBench. Both backends use
explicit uniform temporal sampling so they implement the same request
semantics:

```python
TemporalSamplingConfig(
    time_columns=time_columns,
    strategy="uniform",
)
```

Finite fanout remains stochastic and may select different rows across
backends. Exhaustive fanout establishes exact output parity.

## Timing boundary

The public CUDA latency is a synchronized sampler call with the task table
already resident on the GPU. It includes GPU-resident output assembly. The
following costs are measured separately or excluded:

- relational data transfer to CUDA;
- sampler topology construction;
- per-shape task transfer to CUDA;
- output materialization on the host;
- validation and canonical fingerprinting; and
- the separate CUDA phase-profile pass.

The CUDA phase profile reports task-to-seed lookup, neighbor sampling, output
assembly, and synchronization or other overhead. Uniform temporal selection
is performed by cuGraph and is included in neighbor sampling; there is no
separate latest-k phase.

## Environment

The result comparison requires both modes to use the same Python package
versions. Create the pinned CUDA 13 environment from the repository root:

```bash
uv venv --python 3.12.13 .venv
uv pip install --python .venv/bin/python \
  --torch-backend=cu130 \
  --extra-index-url=https://pypi.nvidia.com \
  --index-strategy=unsafe-best-match \
  'torch==2.11.0' '.[test]' 'relbench==2.1.2' \
  'cudf-cu13==26.6.0' 'pylibcugraph-cu13==26.6.0' \
  'cupy-cuda13x==14.1.1' 'numpy==2.4.6' 'pandas==2.3.3' \
  'pyarrow==23.0.1' 'datasets==4.8.5' 'fsspec==2026.2.0'
uv pip install --python .venv/bin/python \
  --find-links=https://data.pyg.org/whl/torch-2.11.0+cu130.html \
  'pyg-lib==0.7.0+pt211cu130'
```

Verify the environment before downloading benchmark data:

```bash
.venv/bin/python - <<'PY'
from importlib.metadata import version

import cudf
import cupy
import datasets
import pyg_lib
import pylibcugraph
import torch

assert torch.cuda.is_available()
assert version("cudf-cu13") == "26.6.0"
assert version("pylibcugraph-cu13") == "26.6.0"
assert hasattr(
    pylibcugraph,
    "heterogeneous_uniform_temporal_neighbor_sample",
)
print("torch", torch.__version__)
print("cudf", cudf.__version__)
print("cupy", cupy.__version__)
print("datasets", datasets.__version__)
print("pylibcugraph", version("pylibcugraph-cu13"))
print("pyg-lib", version("pyg-lib"))
PY
```

## L4 reproduction

RelBench downloads and verifies the requested dataset on first use. Keep its
cache outside the result directories:

```bash
export RELBENCH_CACHE_DIR="$PWD/.cache/relbench"
```

Run a small smoke test before the full matrix:

```bash
.venv/bin/python benchmarks/relational_sampler.py run --mode cuda \
  --dataset rel-arxiv --task paper-citation --split test \
  --batch-size 1 --fanout 16 --warmups 1 --repetitions 1 \
  --seed 20260711 --output /tmp/rel_arxiv_cuda_cugraph_l4_smoke.json
```

Run the retained CPU and CUDA matrix:

```bash
.venv/bin/python benchmarks/relational_sampler.py run --mode cpu \
  --dataset rel-arxiv --task paper-citation --split test \
  --batch-size 1,128,1024,4096,8192 \
  --fanout 16 --fanout 16,16 --include-exhaustive \
  --parity-batch-size 8 --warmups 10 --repetitions 50 --seed 20260711 \
  --output benchmarks/results/rel_arxiv_cpu_pyg_lib_l4_final.json

.venv/bin/python benchmarks/relational_sampler.py run --mode cuda \
  --dataset rel-arxiv --task paper-citation --split test \
  --batch-size 1,128,1024,4096,8192 \
  --fanout 16 --fanout 16,16 --include-exhaustive \
  --parity-batch-size 8 --warmups 10 --repetitions 50 --seed 20260711 \
  --output benchmarks/results/rel_arxiv_cuda_cugraph_l4_final.json \
  --timeline-html \
    benchmarks/artifacts/rel_arxiv_cuda_cugraph_l4_final_profile.html \
  --timeline-trace \
    benchmarks/artifacts/rel_arxiv_cuda_cugraph_l4_final_profile_trace.json

.venv/bin/python benchmarks/relational_sampler.py compare \
  --cpu benchmarks/results/rel_arxiv_cpu_pyg_lib_l4_final.json \
  --cuda benchmarks/results/rel_arxiv_cuda_cugraph_l4_final.json \
  --output benchmarks/results/rel_arxiv_comparison_l4_final.json
```

Run each mode again with `--batch-size 8 --fanout=-1,-1 --warmups 10 --repetitions 50` and compare those files to retain separate two-hop
exhaustive evidence.

The former T4 results and profiles were generated with latest-k temporal
selection and are intentionally not retained as evidence for the uniform
sampling implementation. New L4 artifacts should be checked in only after
the CPU and CUDA environment metadata, request shapes, and exhaustive output
digests have been compared successfully.
