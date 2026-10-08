# Pretrained process/graph ensemble closure

Plan updated 2026-10-08. Runtime source frozen by the coordinator: `c0fb64fcc`. These are pending comparisons, not results. The tabular owner runs first; relational process runs use the same host only after an explicit idle-GPU handoff. No concurrent performance jobs.

## Fixed configurations

| Family | Data and preparation | Members / query batches | Arms |
|---|---|---|---|
| KumoTabular large | Existing frozen Covertype train/validation split, context 1,024, queries 2,048, seed 1729 | E4, query batch256, local estimator batch1, BF16 | Native1, tuned native batch4, residentEP1, processEP1/2/4, graphEP1/2/4 |
| KumoRelational | Fresh H&M user-churn two-hop `[16,16]` workload, context1,024, queries2,000, seed1729; prepared once and reused | E4, query batch250, native per-member execution, BF16 | Matched residentEP1 and processEP1/2/4 |

All arms use three measured prediction passes after one warmup, matching the existing study protocol. Startup/model loading, context fit, warmup/capture, and prediction are separate. Baseline execution on a replacement host is required; old host timings provide context only. No graph-relational arm is supported.

The historical small H&M workload retained its metadata but not serialized sampled graphs. The closure therefore prepares a new workload from the verified raw data in a fresh directory, records its hashes, compares table/relationship identities with the historical metadata, and reuses its graph file across every new control. Historical timings are not treated as matched fresh-workload controls.

Tabular command template, with `SOURCE`, `DATA`, and a fresh `OUTPUT` replaced by the operator's explicit paths:

```sh
python -m research.multigpu.tabular_bench \
  --data DATA/covertype --output OUTPUT --source-commit c0fb64fcc \
  --task classification --size large --estimators 4 \
  --context 1024 --queries 2048 --batch-size 256 \
  --repeats 3 --warmups 1 --seed 1729 --precision bfloat16 \
  --estimator-batch-size 1 --mode adapter --gpus 4 \
  --adapter research.multigpu.process_ensemble:factory
```

Change the factory to `research.multigpu.graph_ensemble:factory` for graph replay; use `--mode ensemble` without a factory for resident EP. Native uses `--mode native --gpus 1`, with estimator batch1 for semantic parity and batch4 as the tuned practical control. Process and graph adapters execute single members; do not label them estimator-batched.

Relational command template:

```sh
python -m research.multigpu.relational_bench run \
  --workload PREPARED_HM1024 --output OUTPUT --source-commit c0fb64fcc \
  --mode process-ensemble --gpus 4 --estimators 4 \
  --repeats 3 --warmups 1 --threads 8 --dtype bf16 --seed 1729 \
  --include-final-gather
```

Run GPU counts1/2/4 and the same command with `--mode ensemble --gpus 1`. Both use CPU recipe preprocessing and the same member seeds. Native relational uses another recipe-device/RNG policy, so it is not the exact placement reference. Preserve the same allocator, thread, backend, and model-cache environment across this ladder.

## Acceptance and accounting

- Compare every saved prediction repeat to residentEP1 and to the first repeat. Check exact equality first, then the predeclared BF16 gate `atol=0.01, rtol=0.05`; report full-array failures, probability normalization, class-order alignment, quality metrics, targets, and query identities. Do not relax tolerances after seeing results.
- Tabular outer prediction timing includes per-batch CPU output copies and final concatenation. Relational new arms all set `--include-final-gather`; the report explicitly records that boundary. Sampling/preparation remains separate.
- Process startup includes moving supplied replicas to CPU, spawning persistent workers, and child GPU model loading. Parent and child memory are separate. Record child unique cache storage, allocated/reserved/peak memory, and process maximum RSS; reset child peaks before fit and before measured predictions.
- Parent uses eight PyTorch CPU threads while each process worker uses one. Process1-to-process4 is a fixed per-worker policy; differences from threaded/native execution cannot be attributed solely to the GIL.
- The tabular static-shape configuration should create four member graphs during warmup and none during timing, independent of GPU count. `warmup_s` is wall-clock setup; `graph_capture_s` sums overlapping worker durations. Warmup memory captures graph setup peak, while prediction peaks reset afterward. Nondivisible query counts require warming every remainder shape before steady-state timing.
- Tiny graph CUDA tests already exercise changed query values, a second shape, output lifetime, and refit invalidation. Process tests now also cover actual reduced KumoRelational two-hop classification and all999 regression quantiles, with exact CPU replay across two processes. These tests support correctness of those exercised cases, not pretrained throughput claims.

Actual measurements and independent audit paths will be appended only after result files are retrieved. Process/graph GPU startup failures, numerical failures, or memory errors must remain as separate immutable attempts.

## Measured relational closure

Completed on the replacement Ohio host with four L40S GPUs, frozen runtime `c0fb64fcc`, base `/opt/pytorch` environment, no cuDF overlay. All four arms exited successfully; every result, repeat, log, and sampled workload was downloaded before releasing the host. The parent uses eight CPU threads; each process worker uses one. These runs isolate GPU count within the process implementation, but comparing processes with threads also changes CPU execution and IPC.

H&M user-churn uses E4, C1024, Q2000, B250, BF16, seed1729, two-hop `[16,16]` temporal-last sampling. Every arm reads the same fresh `graphs.pt`, SHA256 `072a81055e967d50438f85b1d054ea55fae81c9f7312b93b2512237389059fbd`; validation labels SHA256 `491125dcc0f42a0e12a74850ef9bba5d57fc64c92c8109461e010e11f03f3ca2`. Local and remote file hashes matched. Context and query row IDs and the context graph match the historical metadata, but query graph digests differ: historical throughput is not a matched control.

| Execution | GPUs | Median rows/s | Median prediction s | Speedup / process1 | Load s | Fit s | Warmup s | Prediction peak per GPU, MB | Fit peak per GPU, MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Resident threaded EP | 1 | 1,411.87 | 1.41657 | 1.146× | 0.796 | 6.862 | 1.565 | 532.43 | 960.88 |
| Process EP | 1 | 1,232.49 | 1.62273 | 1.000× | 3.526 | 2.122 | 1.663 | 531.87 | 960.33 |
| Process EP | 2 | 1,916.98 | 1.04331 | 1.555× | 6.183 | 1.965 | 1.092 | 466.87 | 895.47 |
| Process EP | 4 | 2,396.15 | 0.83467 | 1.944× | 11.310 | 1.912 | 0.896 | 398.69 | 826.37 |

Prediction is the median of three full passes after one complete warmup pass. It includes parent preprocessing, child IPC, GPU inference, per-batch CPU outputs, and final CPU concatenation; graph sampling is excluded. Load, fit, and warmup are separate single observations, not replicated setup benchmarks. Resident EP ran first, so its longer first fit may include cold runtime effects: do not interpret the fit column as a causal fit-speed comparison.

Process4 is 1.697× the resident one-GPU baseline and 1.944× process1 (48.6% four-GPU scaling efficiency), with diminishing returns from two to four GPUs. One process on one GPU is slower than the resident threaded control. Higher startup cost and extra model/process memory make this a repeated-inference option, not an automatic win for one-off requests.

The table reports decimal MB of PyTorch allocated-memory peaks, not total board memory. Process entries are child measurements, reset separately before fit and timed prediction; parent allocator data is retained separately. Each process4 worker has a 398.69MB prediction peak, but summing four child peaks gives 1,594.77MB, versus 532.43MB for resident1; these sums are not necessarily simultaneous host-wide peaks. Ensemble cache storage totals 126,631,968 bytes in every arm, divided evenly as 126.63/63.32/31.66MB per worker for process1/2/4. Process4 child maximum RSS ranges from 1.76–2.12GB, plus parent maximum RSS of about 2.67GB; high-water RSS can include shared pages and must not be summed as unique physical RAM.

All four first prediction arrays have SHA256 `58f2b4f62c2048c2b7ab2c4adead44b0a2f2f14f79130e63f42b8176ad1a24e4`. Independent auditing confirmed all three repeats in all four arms are byte-exact to resident1, with accuracy0.8085, AUROC0.66351639, and log-loss0.46535898. The audit also verified graph/label hashes, labels against the original first2,000 validation rows, member seeds, graph metadata, runner identity, and positive-class column alignment. The fixed BF16 gate passes without needing its tolerance margin; per-arm `quality-independent-audit.json` sidecars retain the checks.

Raw artifacts are under `.kumo-multigpu-20261008/results/relational-process-closure/` in sibling directories `relational-closure-v2-resident1`, `relational-closure-v2-process1`, `relational-closure-v2-process2`, `relational-closure-v2-process4`, and `workload-hm-c1024-b250-closure-v2`. The runner records source/runner hashes, exact arguments, runtime environment, ordered outputs, targets, identities, per-repeat timing, telemetry, and parent/child allocator snapshots. No relational CUDA-graph support or large-context relational ProcessEP claim follows from this small-workload result.
