# Independent correctness and measurement audit

Audit scope: current SDM `842c408fe`; KumoTabular and KumoRelational multi-GPU research. This report distinguishes implemented checks from executed GPU evidence. No GPU result has been independently validated yet.

## Fair comparisons

The existing `ICLModel.fit` copies CUDA caches to pinned CPU whenever the number of contexts exceeds one (`sdm/models/base.py`). Native E8 is therefore an offloaded-cache baseline. The ensemble executor retains caches on its assigned devices. Report two separate comparisons: native versus resident one-GPU (execution/cache policy), and resident one-GPU versus two/four GPUs (parallel scaling). Never attribute the offload improvement solely to extra GPUs.

Every comparison fixes checkpoint bytes, model size/task, context observation IDs and order, feature/target schema, member count/order/seeds, fitted recipe, query observation IDs/order and microbatch boundaries, autocast dtype, attention backend, estimator batching, and final output semantics. Report any deliberate changes as separate experiments. Fit and query preprocessing must learn from TRAIN/context only; validation labels are for scoring after predictions. No TEST access is authorized by this experiment.

Native and resident-executor model RNG are distinct contracts: native shares a generator across sequential execution; the executor gives member i seed `member_seed+i`. KumoTabular's ECOC draws randomness only beyond ten classes. Native accuracy is still useful, but matched stochastic scaling requires resident one-GPU versus multi-GPU with the same member seeds. Seven-class Covertype cannot establish ECOC RNG correctness; a separate greater-than-ten-class semantic test is required.

KumoTabular's default classification recipe averages member logits before softmax. Averaging final member probabilities changes the model. Regression inverse target transforms apply per member before aggregation; compare all 999 quantiles, original-unit median RMSE/MAE, pinball loss and quantile crossings. A median-only comparison can miss corruption of the predictive distribution.

## Relational constraints

`TaskGraph.from_input` requires one distinct entity row per task row and explicitly assumes disjoint sampled neighborhoods. Materialize two-hop neighborhoods once, preserving task observation IDs, temporal boundaries, duplicated entity copies, edges, tables and ordering. Whole prepared query microbatches are indivisible scheduling units. Splitting task rows while reusing an unmodified merged graph can alter task assignment and relative-time features. Query data parallelism must schedule complete microbatches or implement and separately validate graph-aware slicing. Repeated entity observations require distinct observation IDs.

The complete relational context and sample are shared across arms. Graph preprocessing, table embeddings, message passing and final ICL attention are distinct phases: context parallelism that only shards final ICL KV does not distribute the graph or full context fit.

## Numerical and quality checks

`quality.py` supplies NumPy-only comparison/scoring helpers. Check finite values and identical shapes/row identities/class labels before comparing values. Save raw member logits where practical plus final predictions. Classification columns must be aligned by class value before AUROC/logloss/accuracy calculations; do not assume numeric labels are contiguous or alphabetical.

| Compute dtype | Screening absolute tolerance | Relative tolerance |
|---|---:|---:|
| FP64 | 1e-9 | 1e-7 |
| FP32 | 1e-5 | 1e-4 |
| FP16 | 2e-3 | 1e-2 |
| BF16 | 1e-2 | 5e-2 |

These are predeclared numerical screening thresholds, not application quality guarantees. Report max/mean/p99 absolute error, relative L2, exact equality, class disagreement and metric deltas regardless of pass/fail. Identical scheduling/math should seek bitwise parity. Context-parallel reductions can change rounding; compare native attention, one-rank partial/LSE attention, and multi-rank partial/LSE attention to separate kernel and distributed-reduction differences. Never loosen a threshold after seeing a failed run without reporting the original failure.

For quality uncertainty, use paired loss differences on identical validation rows. `paired_loss_interval` supports an entity-cluster bootstrap: resample entities, retaining all their observations. An independent-row bootstrap understates uncertainty when entities recur. AUROC is undefined if a resample has only one class; report this explicitly. Numeric equality implies exact metric equality but does not establish accuracy on unseen data.

## Timing and memory

Run each arm in a fresh process. Synchronize every participating GPU before and after wall-clock timing, and aggregate distributed elapsed time by the slowest rank. CUDA-event timings measure GPU stream intervals, not whole-process preprocessing or transfer. Report load, fit/cache, warmup and repeated steady-state query passes separately. Profile in a separate pass because profiler overhead changes latency.

Strong scaling holds total context/member/query work fixed. Weak scaling grows total queries while retaining per-device work and must show that total work grew. Query-DP throughput requires concurrent scheduling of several complete microbatches: synchronizing after each single submitted batch serializes the experiment. Include at least three measured passes, randomized/interleaved arm order where feasible, and distribution/variability. Cold elapsed time includes checkpoint replica loading and cache construction.

Record per-device allocated/reserved peaks, model/cache bytes, CPU RSS/pinned-cache implications, and aggregate memory. Reset peaks at phase boundaries. Retaining every output on GPU through a pass inflates apparent inference memory; document output retention or measure it separately. GPU utilization samples are phase-sensitive and whole-process averages must not be described as kernel occupancy. Failed/OOM/timeout arms remain rows in the result table with workload and failure stage.

## Implementation review status

| Area | Source review finding | Remaining evidence |
|---|---|---|
| Ensemble executor | One shared RecipeExecution, ordered member aggregation, per-member regression inversion, persistent per-replica workers and explicit stream boundaries preserve the intended recipe structure. Member RNG deliberately differs from native. | Actual multi-GPU Kumo predictions, >10-class ECOC, repeated-fit/failure tests, caller-stream tests. |
| Query parallel executor | Schedules whole CPU-prepared graph microbatches, returns results in submission order, serializes each replica with one persistent worker, includes CPU output completion. | Real multi-GPU quality/throughput, exact fitted-plan equality across replicas, graph identity evidence. |
| Context parallel | Source uses global-length query scaling, stable MAX then weighted numerator/denominator SUM, cloned KV shards, rejects unsupported masks/valid lengths and topology changes. Full fit remains replicated. | Real distributed runs, uneven/empty shards, current CUDA efficient-attention backend, full-model quality and memory. |

Review alone is not a GPU correctness claim. Implementation owners are adding real distributed tests; runner results will be independently checked here when available.

## Findings resolved before cloud execution

- Ensemble source initially waited only for input readiness. Parameters initialized on another GPU's nondefault caller stream could race its worker. Follow-up `f74f2cc30` adds per-replica construction stream dependencies and records returned CUDA tensors on the consumer stream; two CUDA tests exercise initialization, different fit/predict streams and CPU outputs. Actual execution on multiple GPUs remains pending.
- The relational runner initially imported the former ensemble module name and omitted the preprocessing generator for the resident executor. Owner follow-ups fix both and retain fixed CPU preprocessing for resident scaling. Native GPU preprocessing remains a separately disclosed baseline.
- The query adapter now propagates estimator batch size, invalidates its executor before refitting, and drains sibling futures before propagating a worker failure. These avoid an unequal batching baseline and stale state after failed fits.
- The ensemble tests now exercise actual reduced-size KumoTabular ECOC with twelve classes, shuffled class vocabularies, exact cached codebooks and predictions across one/two/four CPU replicas and repeated fits. CPU coverage cannot validate CUDA stream behavior.

## Independently checked workload identities

The raw numeric arrays were opened read-only to verify IDs and first-context coverage. Covertype has 464,809 TRAIN and 116,203 validation observations, zero ID overlap and 581,012 unique total IDs. Its first 1,024 TRAIN rows contain all seven classes (counts: 367, 506, 59, 11, 22, 22, 37 for labels 1 through 7). California housing has 16,512 TRAIN and 4,128 validation observations, zero overlap and 20,640 unique total IDs. These are custom deterministic splits, not published benchmark split reproductions.

Rel-hm's 3,832,692 stored TRAIN indices are unique and its 76,556 validation indices are ordered. Rel-f1's 7,453 TRAIN indices are unique and its 499 validation indices are ordered. TRAIN and validation task indices belong to different parquet files, so numeric equality across split-local indices does not imply observation overlap. Validation inference graph sampling is still subject to the timestamp and target-exclusion checks above.

## Actual GPU test evidence

The first four-L40S host execution of `test/models/test_ensemble_parallel.py` completed with eight passing CPU tests and two failing CUDA tests in 33.82 seconds. Both CUDA failures occurred at the newly added `output._tensors()` call: `TableTensor` does not expose the `Cache` tensor iterator. This was an implementation failure, not numerical disagreement or an unsupported hardware result. The independent auditor read the raw `results/executor-tests.log` and immediately notified the implementation owner and coordinator.

Fix `c4ddd7f15` replaces that call with `output.record_stream(stream)`, the existing public operation dispatched by `sdm/tensor/table.py`. A separate rerun saved as `ensemble-cuda-fixed.log` independently confirms both previously failing CUDA tests pass (two passed, eleven deselected, 0.80 seconds). The original failed log remains evidence.

The coordinator also executed the existing base/KumoTabular/KumoRelational/TabICLv2 CPU regression suite: 88 passed, 19 CUDA-only skipped, 27.82 seconds. The integrated new research suites returned 43 passed and 22 CUDA-only skipped in approximately 10 seconds. These support CPU API and model regression coverage; skipped tests are not counted as GPU evidence.

## First measured strong-scaling audit: Covertype

Raw host result directories `tabular-{native,ep1,ep2,ep4}-large-e4-c1024-q2048` were copied independently under `.kumo-multigpu-20261008/results/quality/`. The native source was `9f0d6c765`; the corrected executor source was `6c3f1e22e`. All arms use the pretrained large KumoTabular classifier, four members, 1,024 fixed TRAIN context rows, 2,048 fixed validation queries, microbatch 256, BF16 autocast and three measured passes on the same four-L40S Spot host. Total work remains fixed, so this is strong scaling.

| Arm | Median pass seconds | Recomputed rows/s | Speedup over resident EP1 | Per-device prediction peak allocated GiB |
|---|---:|---:|---:|---|
| Native one GPU, CPU-offloaded caches | 1.125935 | 1,818.93 | Separate policy baseline | 1.284 |
| Resident EP1 | 1.183971 | 1,729.77 | 1.000x | 1.598 |
| Resident EP2 | 1.087597 | 1,883.05 | 1.089x | 1.358, 1.356 |
| Resident EP4 | 1.245131 | 1,644.81 | 0.951x | 1.176, 1.174, 1.174, 1.174 |

The small workload does not scale well: two-GPU efficiency is 54.4% and four-GPU efficiency is 23.8%; four GPUs are slower than one. The resident single-GPU executor itself is slower than the native baseline here. This result must remain in the final account even if larger workloads later improve.

Independent checks found all four saved FP32 prediction arrays `[2048,7]` bitwise identical, finite, and matching their recorded SHA256. Max/mean/p99 absolute differences, relative L2 differences and class disagreement are all zero. All context/query content hashes, query observation IDs, target arrays and class-column orders match. Query IDs and targets additionally match the original fixed Covertype validation prefix. Thus all paired quality deltas are exactly zero on these observations.

Independent NumPy scoring gives accuracy 0.76416015625, macro one-vs-rest AUROC 0.9474149901961447, multiclass Brier score 0.3375792160360528 and FP64 logloss 0.5760446019729857. The harness logloss 0.5760445594787598 differs by 4.25e-8 from FP32 scoring roundoff. Native timing variation across three passes is 0.39% coefficient of variation. This is repeated timing of one context/query cohort, not repeated independent training/context selection or population-level accuracy evidence.

Native enumerated model-cache tensors occupy 358,614,196 CPU bytes (342.001 MiB). Resident EP reports the same logical tensor bytes but 512,754,868 aggregate unique GPU storage bytes, illustrating that logical tensor sizes can underestimate memory retained by views. Both counters omit fitted preprocessing objects that are not traversed by `Cache._tensors()`. Peak allocated memory is the stronger capacity measurement; aggregate parameter replication must also be counted.

The first native fit took 122.75 seconds, whereas later executor fits took roughly 1.1-1.3 seconds. Cold initialization/JIT/cache effects have not yet been isolated; do not attribute that difference to ensemble placement. A fresh matched native control and estimator-batching controls are being run separately.

## Tuned single-GPU control changes the comparison

The subsequent native controls keep the same context, query rows, four members and BF16 settings while grouping one, two or four members per forward. All raw IDs/input hashes/targets/output class orders and prediction file hashes passed independent checks.

| Estimator batch size | Median pass seconds | Recomputed rows/s | Prediction peak allocated GiB | Max probability difference from native E-batch1 | Mean absolute difference | Class changes |
|---|---:|---:|---:|---:|---:|---:|
| 1, repeated warm control | 1.164541 | 1,758.63 | 1.284 | 0 | 0 | 0 |
| 2 | 0.686325 | 2,984.01 | 1.519 | 0.00880432 | 0.00026306 | 0 |
| 4 | 0.511088 | 4,007.14 | 1.646 | 0.00560689 | 0.00025861 | 0 |

The tuned single GPU is 2.13 times faster than the best initial two-GPU ensemble result. Therefore the small-workload study does not establish a multi-GPU performance benefit over a well-batched single GPU. Any later headline must include this stronger baseline.

Batch sizes two and four change floating-point computation and are not bitwise identical, although both meet the predeclared BF16 screening tolerance. All 2,048 predicted classes and accuracy remain unchanged. Batch four's FP64 logloss is 0.5759778337511707, a paired change of -0.0000667682; its paired-row bootstrap 95% interval is [-0.000186384, +0.0000527502]. Macro AUROC changes by -0.00000277055. Batch two's logloss delta is -0.0000488099 with interval [-0.000176049, +0.0000781554]. These bootstrap intervals describe numerical loss differences on the available cohort, not independence of spatial forest observations or performance on another dataset.

The repeated native fit is 1.3268 seconds. This confirms that the earlier 122.75-second first fit contains a large cold-start effect; it is unsuitable as a placement speedup claim.

## Native relational ladder audit

All seven rel-hm arms reuse the exact stored `[16,16]` temporal-last graphs, context 1,024, four members and first 2,000 ordered validation rows in microbatches of 250. Each stored workload object, graph hash, class-column sequence and prediction content hash passed independent comparison. Targets were reconstructed for scoring from the original ordered validation parquet.

| Arm | Recomputed median rows/s | Matched one-GPU speedup | Largest per-device prediction peak GiB |
|---|---:|---:|---:|
| Native | 1,294.66 | Separate baseline | 0.459 |
| EP1 | 1,361.95 | 1.000x | 0.496 |
| EP2 | 1,465.28 | 1.076x | 0.435 |
| EP4 | 1,577.98 | 1.159x | 0.371 |
| Query DP1 | 1,350.21 | 1.000x | 0.468 |
| Query DP2 | 1,465.94 | 1.086x | 0.468 |
| Query DP4 | 1,211.90 | 0.898x | 0.468 |

EP1/2/4 predictions are bitwise identical to each other; query DP1/2/4 predictions are bitwise identical to native. However, EP versus native is not the same stochastic execution: the resident path preprocesses on CPU and uses separate model-member seeds, including sampled GNN edge-type embeddings. Native-to-EP max probability difference is 0.0795122 and mean difference 0.00943745, failing the predeclared BF16 numerical screen. One of 2,000 predicted classes changes. This cannot be described as native-parity or a GPU-placement accuracy benefit.

Native/query-DP accuracy is 0.808, logloss 0.4668492853 and churn-class-1 AUROC 0.6619477129. EP accuracy is 0.8085, logloss 0.4653579016 and churn-class-1 AUROC 0.6635244651. The model's output columns are `['1','0']`; the original harness index-positive AUROC selects class 0 and reports 0.6635422268 for EP. Complementary FP32 probability ties account for the small AUROC difference; always name the positive class. The class-aligned paired logloss delta is -0.00149138, with entity-bootstrap 95% interval [-0.00302336, +0.0000309842]. GPU placement within either execution policy has zero quality delta.

The rel-f1 E4 arms share 1,024 context rows and all 499 validation rows, in four graph microbatches (125/125/125/124). EP1/2/4 arrays `[499,999]` are bitwise identical in all quantiles, finite, correctly ordered `q001` through `q999`, and have zero crossings. EP1/2/4 achieve 384.04/379.23/356.47 rows/s: two and four GPUs are slower than the matched one-GPU executor (0.987x/0.928x). Largest per-GPU peak allocations fall from 0.388 to 0.326 to 0.259 GiB. Native E4 achieves 371.72 rows/s.

F1 EP median RMSE/MAE are 4.09476157/3.32554260, mean pinball loss is 1.17491292 and q050-q950 coverage is 0.95991984. Native E4 RMSE/MAE are 4.21103292/3.41042852. Native-to-EP full-quantile max/mean differences are 4.275915/0.353979 and fail the numerical screen because of the changed stochastic execution; matched EP placement differences remain exactly zero. The paired MAE difference is -0.0848859, with a 47-driver cluster bootstrap interval [-0.139039, -0.0375763]. This is a comparison of one particular pair of random-member plans, not evidence that using more GPUs improves prediction quality.

An additional F1 native E1 control independently validated `[499,999]` outputs with no crossings, RMSE 4.38872041, MAE 3.56287834, pinball loss 1.24091992 and coverage 0.96192385. Its throughput is 1,123.93 rows/s (median of three passes); E1 and E4 have different amounts of model work and should not be called parallel scaling.

F1 query DP1/2/4 additionally produces full-quantile predictions bitwise identical to native E4. Median throughput is 357.41/377.07/296.78 rows/s, giving 1.055x and 0.830x matched speedup at two and four GPUs. All fourteen E4 relational arm directories contain an independent `quality-independent-audit.json` sidecar preserving the original logged quality, corrected positive-class scoring, raw prediction hash, graph/workload checks, matched/native numerical differences and recomputed throughput. The original `result.json` files are unchanged.

## Sequential model placement: capacity rather than throughput

The four-arm placement study runs on the second host's L4 GPUs, source `55dba6bb5`, with the same Covertype context 1,024/query 2,048/member E4/microbatch 256 settings. It compares a local EP1 baseline with placing all ICL layers on a second GPU, two contiguous layer partitions, or four contiguous partitions. This executes one pipeline serially, so it does not demonstrate overlapped pipeline throughput.

| Placement | Recomputed rows/s | Speedup over same-host EP1 | Largest fit peak GiB | Largest prediction peak GiB |
|---|---:|---:|---:|---:|
| EP1 | 1,855.27 | 1.000x | 1.769 | 1.598 |
| Entire ICL on GPU2 | 1,763.91 | 0.951x | 1.074 | 1.060 |
| Two layer partitions | 1,830.08 | 0.986x | 1.248 | 1.071 |
| Four layer partitions | 1,787.74 | 0.964x | 0.994 | 0.813 |

All four raw arrays, input hashes, query IDs, targets, class order, member seeds (1729 through 1732) and recorded output hashes pass independent checks. Predictions are bitwise identical on this host, with zero max/mean error and class changes. Accuracy is 0.76416015625, FP64 logloss 0.5758563016 and macro AUROC 0.9474312942. The L4 baseline differs slightly from the L40S baseline, so it must remain a same-host comparison. Four partitions cut the largest prediction allocation by about 49%, while reducing throughput by about 3.6%. The single cold-fit timing per arm is insufficient for a fit-speedup claim.

The raw placement test log independently confirms 36 passing tests in 5.88 seconds, including 22 tests that required GPUs and had been skipped locally. Four per-arm `quality-independent-audit.json` sidecars store the recomputed results.

At context 16,384 on the same L4 host, all five EP1/compact-cache/stage2/layers2/layers4 arms remain bitwise identical (accuracy 0.91455078, logloss 0.225671257). Their throughputs are 1,865.11/1,799.68/1,830.66/1,754.70/1,809.19 rows/s. Four layer partitions reduce the largest prediction allocation from 2.1873 to 0.8765 GiB (59.9%) and fit peak from 3.0489 to 1.8760 GiB (38.5%), with 0.970x throughput. Cache compaction is a no-op for this workload: its largest allocated peak is unchanged. All five raw directories contain independent audit sidecars.

## Batched ensemble: larger workload changes the outcome

The next Covertype ladder uses eight members, context 4,096, 8,192 fixed validation queries and microbatch 1,024 on L40S, with a maximum estimator batch size of eight. The native baseline batches eight members. The resident adapters cap local batch width to eight/four/two on one/two/four GPUs. Thus the multi-GPU comparison changes local arithmetic batch shape, while preserving total work and member semantics.

| Arm | Recomputed rows/s | Speedup over resident EP1 | Largest prediction peak GiB |
|---|---:|---:|---:|
| Native, estimator batch 8 | 3,219.44 | Separate residency policy | 3.743 |
| Resident batched EP1 | 4,013.81 | 1.000x | 3.503 |
| Resident batched EP2 | 7,368.45 | 1.836x | 2.405 |
| Resident batched EP4 | 5,981.13 | 1.490x | 1.612 |

The two-GPU result is bitwise identical to both native and resident EP1 across `[8192,7]` predictions, with zero class or quality changes. Its 1.8358x resident scaling corresponds to 91.8% efficiency; the larger 2.2887x comparison against native includes the cache-residency effect. Accuracy is 0.841796875, FP64 logloss 0.4138842210 and macro AUROC 0.9749343473.

Four GPUs are slower than two and are not numerically identical: maximum probability difference is 0.0322531, mean 0.000526267 and p99 0.00704364. One of 57,344 probability entries fails the original BF16 tolerance, and six of 8,192 predicted classes change. Accuracy rises by one correct observation, logloss changes by +0.0000402448 (paired-row 95% interval [-0.000119914, +0.000197948]), and macro AUROC changes by -0.00000792784. The threshold is not relaxed. Local batch-width differences are a plausible numerical cause; the result is reported with its observed error rather than called exact parity.

All four input hashes, IDs, labels, columns and output hashes independently match, and per-arm audit sidecars preserve these checks. The larger query cohort strengthens the measured quality comparison but still represents one context selection and member plan.

## Context-parallel pretrained relational regression

The F1 Flash-attention study compares native attention, the one-rank LSE implementation, and two/four context ranks on L4. All arms use the same stored graphs, 1,024 context rows, E4 and all 499 validation observations; native/LSE1 outputs are bitwise identical. Every saved repeat and every rank's reported output hash agrees within its arm. Slowest-rank timing was independently recomputed from individual rank arrays and exactly matches the aggregate report.

| Arm | Unique rows/s | Native speedup | Median RMSE | MAE | Largest prediction peak GiB |
|---|---:|---:|---:|---:|---:|
| Native | 364.23 | 1.000x | 4.21499280 | 3.41471108 | 0.331 |
| One-rank LSE | 362.35 | 0.995x | 4.21499280 | 3.41471108 | 0.330 |
| CP2 | 361.15 | 0.992x | 4.21911206 | 3.41806200 | 0.307 |
| CP4 | 376.00 | 1.032x | 4.21744741 | 3.41670209 | 0.295 |

The full `[499,999]` arrays are finite, correctly ordered and have zero quantile crossings. However, CP2/CP4 fail the predeclared full-quantile BF16 screen: maximum differences are 0.2022419/0.2022400, mean differences 0.0170425/0.0142532 and relative L2 errors 0.0019955/0.0016981. Failures affect 938/915 of 498,501 entries, confined to lower quantiles q001-q049/q052 near zero. Median-only comparisons pass and would miss these tail differences.

CP2/CP4 increase MAE by 0.00335093/0.00199101. The paired 47-driver cluster bootstrap 95% intervals are [0.0013071, 0.0054785] and [0.0003818, 0.0037831]. The deterioration is small in absolute target units but is measured, not claimed away as equality. Per-rank ICL cache bytes fall from 100,663,296 to 50,331,648/25,165,824; the much smaller change in total GPU allocation reflects unsharded components and native cache offload. A 3.2% three-repeat throughput increase is insufficient for a robust speedup conclusion. Four sidecars beside the raw CP outputs record the independent numerical and timing audit.
