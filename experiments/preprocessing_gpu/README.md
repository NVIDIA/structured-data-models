# GPU validation of preprocessing compilation

These are actual CUDA Inductor measurements on one NVIDIA L4, using pretrained Kumo models and held-out real data. They validate experimental source branches, **not unmodified main**. FP32 and BF16 are each compared against eager execution in the same precision. Tolerance is `atol=1e-5, rtol=1e-4`; no tolerance was widened.

## Confirmed FP32 results

Warm wall time is the median of five measured calls after warmup. Memory is peak **PyTorch allocated tensor memory**, including resident model/cache tensors; it excludes CUDA contexts, compiler host memory, and cuDF/RMM allocations. It is not total GPU memory consumption. Compilation overhead is excluded.

| Workload | Eager | Internal model compiled | Public `predict` compiled | Eager / compiled peak allocated | Maximum score error |
|---|---:|---:|---:|---:|---:|
| Tabular, 256 context rows, 128 queries, 1 estimator, FP32 | 17.28 ms | 10.42 ms | 4.54 ms | 182.59 / 174.14 MiB | 1.38e-6 |
| Relational, real driver-DNF one-hop graph, 8 queries, FP32, final integration | 74.15 ms | Not measured | 39.85 ms | 142.07 / 141.74 MiB | 2.33e-6 |
| Tabular, 256 context rows, 128 queries, 2 estimators, FP32, **current-stream source fix** | 33.41 ms | 16.60 ms | 12.02 ms | 195.85 / 169.78 MiB | 2.00e-6 |

For Tabular, compiling public prediction improves this workload beyond compiling only the internal network. The first row uses the same dataset, checkpoint, context/query sizes, source, and dtype across the independent internal/public runs. An additional seven-pair alternating eager/compiled control confirms the public speed benefit without creating new graphs. Reserved allocator memory increased (approximately 244 to 292 MiB), despite lower live tensor peaks.

Both graph-break policies passed the smaller 32-context Tabular FP32 workload. The public fullgraph path was also exercised with query rows 32→128→32, and relational queries 4→1→8→4. These are limited workload validations, not claims that every model configuration or arbitrary table schema compiles.

## Remaining correctness and compiler blockers

| Case | Result | Meaning |
|---|---|---|
| PyTorch 2.14 public prediction, GPU-resident cache, FP32 | Pass after stream-event fix | Original `wait_stream` triggers an Inductor graph-ordering error; explicit event calls preserve eager semantics and avoid that compiler failure. |
| PyTorch 2.14 public prediction, **two CPU-offloaded caches**, FP32, separate transfer stream | **Wrong scores after the first compiled call**, maximum error 0.37245 | Inductor reuses first-cache buffers for second-cache transfers without a compute→transfer dependency. A successful first call is insufficient validation. |
| Same two-cache case, transfers on the current stream | Pass, including source replay `f0a5db1e4` | Eager keeps asynchronous transfers; compiled prediction uses the compute stream to prevent unsafe concurrent buffer reuse. This trades away compiled transfer/compute overlap but still improves this workload over eager and internal-only compilation. |
| PyTorch 2.14 BF16 public prediction | Compiles after empty-container fix, but **fails score parity** | Maximum error 0.03309 at 128 queries; no speedup claim accepted for this failing case. |
| PyTorch 2.14 BF16 internal-only compilation | **Fails score parity** | Maximum error 0.01554 at 128 queries, so preprocessing is not the sole cause. Frontend-only `backend="eager"` is bitwise exact on the matched 32-row case; it is a diagnostic, not an Inductor performance result. `emulate_precision_casts=True` reduces error but still fails 17/64 values (max 0.00194186). |
| PyTorch 2.7.1 public fullgraph prediction | Fails | Generic generator context around preprocessing remains unsupported; independent stream repro also rejects `record_stream` in strict mode. |
| PyTorch 2.7.1 public prediction allowing graph breaks | Pass on the tested FP32 workload | Partial capture works, but is slower: 204.62 ms versus 18.40 ms eager, with 36 graphs. Do not recommend this configuration as an optimization. |

Classification labels happened to agree in some failing score cases. That does **not** establish prediction parity or acceptable quality. Results include individual maximum errors and failed-value counts.

## Exact source and environment

- Frozen integration: `c75ba3af6c231c7bdcd181a50d35cdc7fb53c090`.
- Stream-event source: frozen integration + `18a4d151a`; changes two stream waits to explicit record-event/wait-event calls.
- BF16 container source: stream-event source + `4071789e4`; avoids reading data-dependent offsets for an empty ragged payload during dtype conversion.
- Final relational validation: `e2abde1542f15ce605b33739411a52edbc87d564`. Includes dynamic table/CSR fixes, empty-container fix, and stream events. No experimental native LayerNorm replacement or cache hashing.
- Source current-stream candidate: stream-event source + `f0a5db1e4`; validation recorded separately from the runtime diagnostic.
- Main benchmark runner: `937beb883`; alternating runner: `de6d656c4`. Copied scripts include their diagnostic additions and hashes.
- PyTorch 2.14.0+cu130, cuDF 26.08.01; PyTorch 2.7.1+cu126 separately installed and `pip check` clean. Exact packages are in `receipts/`. The clean 2.7 environment has no cuDF, so no 2.7 relational CUDA claim.
- A rejected installation of cuDF-cu12 conflicted with PyTorch 2.7's pinned CUDA libraries. No reported results use that rejected environment.

Actual Inductor was used except explicitly named frontend-only diagnostics. Both FX and AOT disk caches were disabled after the initial small Tabular cases; subsequent compiler caches were isolated by case. Compilation workers were increased from one to four for later cases; timed GPU jobs were serialized. The source current-stream replay can reuse generated kernels from the equivalent diagnostic, but FX/AOT graph caches remain disabled. No cold-compilation speed claim is made.

## CSR validation

The custom CSR operator originally forwarded strided indices to a native CUDA kernel that assumes contiguous storage. Fresh subprocess isolation and an independent `bincount`/cumulative-sum oracle exposed wrong native results, and opcheck could trigger illegal memory access. `4a77261ce` normalizes input with `.contiguous()` before calling the native operator; normal sorted graph indices already satisfy this layout, so no extra copy is made for them.

The corrected operator passed four opcheck combinations and 48 dynamic CUDA cases on **both** PyTorch 2.7.1 and 2.14, including empty inputs, changing node counts, int32/int64, and strided inputs. The original native strided calls remain failures. Results use the independent oracle; an earlier native-versus-native comparison was invalid and is deliberately excluded.

## Reproduction and artifacts

`results/` contains completed result JSON/JSONL, including failures. Paths are normalized from the temporary host root to `/validation`; numerical values and error text are otherwise preserved. `scripts/` contains exact runner variants and followup commands. Place the manifest-matching data/checkpoints under `/validation/data`, the stated sources under the script's source directories, and the runtimes under `/validation/torch214` and `/validation/torch27`.

Fit runs eagerly before compiling prediction. Public prediction uses `torch.compile(model.predict, fullgraph=...)`; internal-only runs compile each `model.models` member and call ordinary `model.predict`. Compilation therefore includes different amounts of preprocessing/orchestration by design. The raw config records the backend, dtype/autocast, graph policy, context/query sizes, and entry point for every result.

Input hashes identify the held-out classification/regression fixtures and pretrained checkpoint snapshots. The published runs here cover classification and driver-DNF, not regression. Binary weights, private cloud identifiers, and saved prediction tensors are excluded. Raw prediction tensors and generated code were retained locally for diagnosis. Spot cleanup is managed by the parent investigation; no permanent GPU resources are required.
