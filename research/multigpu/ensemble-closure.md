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
