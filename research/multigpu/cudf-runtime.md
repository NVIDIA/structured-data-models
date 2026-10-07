# cuDF runtime experiment

The initial GPU runtime is `/opt/pytorch`, Python 3.12.3, PyTorch 2.9.1+cu130, CUDA runtime 13.0, Ubuntu 24.04, glibc 2.39 and driver 595.91.07. It has no cuDF. On this runtime SDM emits CPU fallback warnings for relational joins and string sorting. This runtime condition is part of the baseline and must be recorded with its results.

## Supported candidate

[`cudf-cu13==26.8.1`](https://pypi.org/project/cudf-cu13/26.8.1/) provides Linux wheels for Python 3.11+. The [official installation guide](https://docs.nvidia.com/datascience/install/) supports CUDA 13 with driver 580.65.06 or newer and glibc 2.28 or newer; the benchmark hosts meet those requirements. SDM declares `cudf-cu13>=26.8` in its Linux test dependency group, not as a core runtime requirement.

The current cuDF dependency metadata requires pandas `>=3.0.0,<3.0.4a0`, while the baseline has pandas 3.0.6. Its libcudf dependency also requires `nvidia-nvjitlink>=13.3,<14`. Therefore install in a separate environment and record the resolved changes; a simple in-place installation would alter the baseline. Pin the toolkit metapackage to the host's 13.0 series initially and verify imports and actual CUDA execution before claiming compatibility.

The intended environment is `/home/ubuntu/kumo-multigpu/venv-cudf26.8`, created by `/opt/pytorch/bin/python -m venv --system-site-packages`. This inherits the existing PyTorch build while package writes occur in the new venv. Run a pip dry-run with a JSON report before installing `cudf-cu13==26.8.1` and `cuda-toolkit==13.0.*`. Capture the final freeze, pip check, resolved Torch/CUDA/cuDF/CuPy/pandas/NumPy versions, and import locations. The baseline `/opt/pytorch` must remain unchanged. Run installation between benchmark windows and GPU smoke only after acquiring the runner slot.

## Exact SDM backend behavior

`sdm/relational/join.py:join_index` checks whether input tables are CUDA tensors and `importlib.util.find_spec("cudf")` succeeds. When cuDF is absent, it converts key columns to Arrow on CPU, performs an Arrow join and copies row indices back to the requested device. This entails device synchronization and transfer on every such join. With cuDF installed, it converts to CUDA-backed cuDF columns, merges on the current CUDA device and returns device row indices.

The same presence check controls string sorting in `sdm/tensor/string.py`. Sorting category vocabularies can therefore synchronize even when model weights, numeric inputs and attention remain on GPU. These are separate paths from the CPU PyG neighborhood sampler: installing cuDF does not automatically move prepared neighborhood sampling to CUDA or require cuGraph.

Each `TaskGraph.from_input` constructs graph relationships and the task-to-entity mapping through `join_index`. Ensemble members repeat this topology construction. GPU parallel execution can overlap model kernels yet be limited by synchronized host joins and Python coordination, so profiler traces should distinguish graph construction, string/categorical processing, row embedding, GNN and final ICL attention.

## Comparison design

First retain the original runtime's completed native/ensemble profiles. Then run the same native and parallel arms, fixed sampled graphs, checkpoints, context, ensemble plan, precision, query batches and repetitions in the candidate runtime. Save predictions and compare quality and numerical parity.

Because the candidate may change pandas/NumPy dependencies, also run a paired control in that same candidate environment with SDM's cuDF availability check forced to report absent. Label this a research harness override and record it explicitly. This separates backend effect from dependency drift. Confirm the intended backend from the executed path, not merely the package list or absence of warnings.

cuDF/RMM allocations can fall outside PyTorch's allocator. Report CUDA allocated/reserved peaks together with whole-process GPU memory sampling or explicit RMM memory accounting. A lower PyTorch peak by itself would not demonstrate a lower total-memory runtime. Keep cold import/first-call costs separate from warm inference.

## Installation outcome

The isolated candidate was installed during a reserved CPU/I/O window on the L40S host. No GPU smoke or timed inference was started during installation. Python resolves Torch from the unchanged `/opt/pytorch` installation and cuDF from the new overlay. A `.pth` entry explicitly adds `/opt/pytorch/lib/python3.12/site-packages`; a nested `venv --system-site-packages` alone inherits system paths rather than the parent venv. The final overlay disables system-site packages and preserves the explicit parent path after its own site-packages.

Resolved versions are Torch 2.9.1+cu130, cuDF 26.8.1, pandas 3.0.3, NumPy 2.4.6 and PyArrow 23.0.1. Installation succeeded but `pip check` **failed**: Torch 2.9.1 pins older CUDA component versions exactly, while cuDF/toolkit extras selected newer patch versions. Conflicts include cuBLAS, NVRTC, CUDA runtime, cuFFT, cuFile, cuSolver, cuSparse and nvJitLink. Most critically, libcudf requires nvJitLink at least 13.3 while this Torch build pins 13.0.39. The candidate is therefore an experimental, dependency-inconsistent environment until replaced or explicitly evaluated with that caveat. Package presence does not establish binary/runtime compatibility.

The base environment was not changed; pip explicitly declined uninstalling packages outside the new environment. Dry-run report, install report/log, freeze and `pip-check.txt` are stored at `.kumo-multigpu-20261008/ops/runtime-cudf/` locally and `/home/ubuntu/kumo-multigpu/runtime-cudf/` on the host. CUDA smoke and paired backend runs remain scheduled after existing baseline work. No cuDF performance improvement is claimed here.

## Dependency-consistent successor candidate

An off-host Linux/Python 3.12 dependency resolution succeeded for `cudf-cu13==26.6.0`, `cuda-toolkit==13.0.0` and `cupy-cuda13x==13.6.0`, explicitly constraining all eight CUDA component versions to the existing Torch 2.9.1 pins. The result resolves 44 packages, including pandas 2.3.3, NumPy 2.4.6 and PyArrow 23.0.1. This avoids upgrading Torch and avoids the incompatible nvJitLink requirement introduced by the 26.8 candidate. Resolution input/output are `.kumo-multigpu-20261008/ops/cudf26.6-candidate.in` and `cudf26.6-resolved.txt`.

Official [cuDF v26.06.00 Series source](https://github.com/NVIDIA/cudf/blob/v26.06.00/python/cudf/cudf/core/series.py) contains `to_pylibcudf` and `from_pylibcudf`; its [pylibcudf Column source](https://github.com/NVIDIA/cudf/blob/v26.06.00/python/pylibcudf/pylibcudf/column.pyx) contains the `from_array` and GPU-memory-view APIs SDM uses. Source compatibility is encouraging, but actual runtime checks remain required. `check_cudf_runtime.py` exercises a nullable composite string join against a CPU reference, CUDA string sorting and a cuDF string roundtrip in an explicitly scheduled GPU smoke slot.

The successor is installed at `/home/ubuntu/kumo-multigpu/venv-cudf26.6`. After constraining `rich==14.3.4` to retain compatibility with the inherited SageMaker dependency, its final `pip check` reports **No broken requirements found**. The final report and package freeze are `runtime-cudf/check26.6-final.txt` and `freeze26.6-final.txt`. This establishes dependency consistency, not successful CUDA execution; GPU interface validation and performance comparisons are separate steps.

For paired comparisons, `run_dataframe_backend.py --backend arrow|cudf --receipt RECEIPT RUNNER [RUNNER_ARGUMENTS]` executes both arms in the same 26.6 environment. The Arrow arm intercepts only `find_spec("cudf")` availability checks, and the receipt records observed calling modules and dependency versions. This research override covers a single process and thread-based ensemble workers, not spawned data-parallel processes. Keep receipts outside the runner's new output directory and compare saved predictions as well as throughput. In particular, differing join order can change floating-point reduction order even when matched row pairs are identical.
