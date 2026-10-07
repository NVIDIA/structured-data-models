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

| Component | Original runtime | Isolated paired-backend runtime |
|---|---|---|
| PyTorch | 2.9.1+cu130 | 2.9.1+cu130, inherited unchanged |
| CUDA runtime / nvJitLink | 13.0.48 / 13.0.39 | 13.0.48 / 13.0.39 |
| cuDF / libcudf / pylibcudf / RMM | Absent | 26.6.0 |
| CuPy | Absent | 13.6.0 |
| pandas | 3.0.6 | 2.3.3 |
| NumPy | 2.5.3 | 2.4.6 |
| PyArrow | 25.0.1 | 23.0.1 |

These versions come from the captured baseline and final overlay freezes. Therefore comparing cuDF directly with the original runtime would confound dataframe-backend selection with three dependency changes. The same-overlay Arrow control is necessary for backend attribution.

The actual CUDA interface smoke subsequently **passed** on the L40S host using source `1686803e4`: nullable composite-key join produced the same row pairs as CPU Arrow (`[(0,2),(1,0),(2,1)]`, dropping the null-key row), CUDA string sorting returned the expected order, and SDM string tensors round-tripped through cuDF. The success record and log are `runtime-cudf/smoke26.6.json` and `smoke26.6.log`; the imported cuDF version string is `26.06.00`. This establishes compatibility for the tested SDM interfaces, not comprehensive cuDF compatibility or a performance advantage.

For paired comparisons, `run_dataframe_backend.py --backend arrow|cudf --receipt RECEIPT RUNNER [RUNNER_ARGUMENTS]` executes both arms in the same 26.6 environment. The Arrow arm intercepts only `find_spec("cudf")` availability checks, and the receipt records observed calling modules and dependency versions. This research override covers a single process and thread-based ensemble workers, not spawned data-parallel processes. Keep receipts outside the runner's new output directory and compare saved predictions as well as throughput. In particular, differing join order can change floating-point reduction order even when matched row pairs are identical.

## Completed paired experiment

All six arms completed on the same four-L40S host, using the same isolated environment, source `1686803e4`, runner `relational_bench-a12b70a75.py`, real rel-hm/user-churn sampled workload, 1,024 TRAIN context rows, 2,000 VAL queries, query batches of 250, exact temporal `[16,16]` neighbors, E4, BF16, seed 1729, eight CPU threads, one warmup and three timed repetitions. Each arm ran in its own process with an exclusive GPU lease. Native execution and ensemble execution use different existing ensemble plans, so backend effects are compared **within each row**, not native versus EP. The phase diagnostic is a separate pass excluded from throughput measurements.

| Execution | Arrow median rows/s | cuDF median rows/s | cuDF / Arrow | Arrow → cuDF fit seconds | Arrow → cuDF logged AUROC (repeat 3) | Largest within-cuDF repeat difference |
|---|---:|---:|---:|---:|---:|---:|
| Native, one GPU | 1,328.0 | 1,082.9 | 0.815× | 1.589 → 2.521 | 0.661948 → 0.662308 | 0.004085 |
| Ensemble, one GPU | 1,347.6 | 1,154.3 | 0.857× | 1.831 → 2.802 | 0.663524 → 0.663760 | 0.003799 |
| Ensemble, four GPUs | 1,364.5 | 1,059.5 | 0.776× | 2.015 → 2.882 | 0.663524 → 0.664215 | 0.004012 |

cuDF was **14–22% slower** in measured warm throughput for this small sampled-graph workload. Fit measurements include each process's first backend/model calls and are not separate steady-state fit estimates. Three within-process repetitions are descriptive, not confidence intervals over independent runs. Four-GPU cuDF times declined from 1.982 to 1.888 to 1.780 seconds, so additional warmup could affect its steady-state estimate; even its fastest observed repetition remained slower than every matched Arrow repetition (1.463–1.484 seconds). No extrapolation to larger join batches or other datasets is justified.

### Correctness and reproducibility

All Arrow arms were prediction-identical across repetitions and their separate phase pass. Arrow EP1 and EP4 saved predictions were also exactly equal. In contrast, all cuDF arms had nonzero repeat differences shown above and failed the phase pass's exact-prediction check. cuDF EP1 versus EP4 differed by up to 0.009577 in probability, although hard class predictions agreed on all 2,000 rows.

Cross-backend maximum probability differences were 0.011796 native, 0.011978 EP1 and 0.011602 EP4; mean absolute differences were approximately 0.00153–0.00157. Hard-prediction agreement was 99.95% native and 100% for both ensemble modes. cuDF native accuracy was 0.8085 versus Arrow 0.8080; all ensemble accuracies were 0.8085. Small observed AUROC differences are not evidence of a quality improvement, particularly given within-backend nondeterminism. This does **not** pass a bitwise or 1e-3 absolute/relative backend-equivalence check.

The runner exposed a provenance inconsistency: `predictions.npy` and `prediction_sha256` refer to **repeat 1**, whereas logged `quality` and `predictions.pt` refer to **repeat 3**. Arrow scores are unaffected because repeats are identical; cuDF scores must retain the repeat distinction. Independent local rescoring of all six saved arrays used the first 2,000 raw VAL `churn` labels, aligned the actual probability column order `[1,0]`, and verified each array's SHA256 against its receipt. First-repeat cuDF AUROCs are 0.662338 (native), 0.663781 (EP1), and 0.664230 (EP4); corresponding log losses are 0.466934, 0.465332, and 0.465203. First-repeat Arrow AUROC/accuracy match the logged scores. All probability-difference comparisons above use saved **first-repeat** arrays; the table explicitly retains original **third-repeat** logged quality. The raw artifacts were not rewritten to conceal this inconsistency.

The simple nullable-key smoke proves matching pairs for that fixture, not deterministic ordering or complete model equivalence. A plausible mechanism is unordered cuDF hash-join output feeding graph edge sorting by destination only, which leaves reduction order among equal destinations unconstrained. CUDA stream interoperability is another possibility and is not excluded by the available evidence. No completed graph-repeat diagnostic exists: `check_graph_repeatability.py` was prepared to compare ordered edges, canonical edge multisets and task/readout assignments under explicit synchronization, but new network restrictions prevented completing/retrieving that diagnostic. **The cause remains unresolved; no core fix is claimed.**

### Phase and memory observations

For the native separate phase pass, summed inclusive `TaskGraph.from_input` host intervals increased from 0.206 seconds (Arrow) to 0.513 seconds (cuDF), across 32 calls. The nested `HomogeneousGraph.from_tables` intervals increased from 0.136 to 0.318 seconds, while recipe transform increased from 0.371 to 0.412 seconds. Whole phase-pass wall time increased from 1.511 to 1.854 seconds. This is consistent with graph construction accounting for much of the slowdown in this case, but these are host intervals including any synchronization, **not exclusive GPU kernel percentages**. Do not add nested graph intervals, and do not divide four-worker summed intervals by wall time: their calls overlap.

Executed backend receipts confirm 572 join and 16 string-backend checks in each native arm; ensemble arms each record 492 join checks and no CUDA string checks, consistent with CPU recipe processing. Package versions match across all six receipts.

| Execution | Arrow → cuDF Torch prediction peak, MiB per GPU | Arrow → cuDF sampled whole-GPU prediction peak, MiB |
|---|---:|---:|
| Native, one GPU | 470.106 → 470.106 | 1,641 → 1,685 |
| Ensemble, one GPU | 507.761 → 508.137 | 1,681 → 1,685 |
| Ensemble, four GPUs | 379.838 → 379.838 on each GPU | 1,549–1,551 → 1,549–1,553 across GPUs |

Torch numbers are measured maximum allocated bytes after warmup, converted using 2²⁰ bytes/MiB. Whole-GPU numbers are the maximum `nvidia-smi memory.used` sample within recorded prediction wall-clock windows, sampled every 200 ms; they include context/runtime/RMM allocations and can miss short peaks. They are not allocator-exact process peaks. These data do not demonstrate a meaningful memory reduction from cuDF.

### Recommendation and evidence

Keep the original Arrow runtime as the consistent baseline for the current multi-GPU results. cuDF 26.6 is installable and passes targeted SDM interface smoke, but this measured workload shows a latency regression and unresolved prediction-repeatability changes. Do not recommend it as a drop-in performance or exactness improvement without a resolved ordering/stream investigation and representative large-join measurements. cuDF 26.8 remains dependency-inconsistent with this Torch build and was not benchmarked.

All six complete result bundles are local at `.kumo-multigpu-20261008/results/relational/relational-backend-{arrow,cudf}-{native1,ensemble1,ensemble4}-e4-c1024/`, each with `result.json`, `backend.json`, `predictions.npy` and `nvidia-smi.csv`. Local environment evidence is `.kumo-multigpu-20261008/ops/runtime-cudf/`, including successful smoke JSON/log, final freeze/dependency check and installation reports. No completed graph-repeat result is available locally or reported by its runner; preserve that missing evidence explicitly rather than inferring its outcome.

The final local consistency audit found one common source commit, one runner SHA256 (`ea202e91bf7a43239caf79f639309d0fc2bcfd623f2a4f4e06ed358117a94590`), identical workload metadata and fixed parameters, and identical five-package runtime receipts across all six arms. Canonically serializing the workload object with Python `json.dumps(workload, sort_keys=True)` gives SHA256 `6c8ddded6ad47b87bb3f373b0bb9faeedd8c5184910b2739849d9d05b9dbd391` in every result. Each saved prediction array independently matches its recorded first-repeat hash.
