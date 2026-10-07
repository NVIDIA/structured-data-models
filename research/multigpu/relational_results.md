# Native KumoRelational GPU results

The first real-data ladder found exact prediction preservation across GPU counts but limited throughput scaling at a 1,024-row context. On HM, four-GPU ensemble placement was 1.159 times faster than the same one-GPU executor. On F1 it was slower. Thread-based query parallelism improved throughput slightly with two GPUs and regressed with four. These measurements support further work on larger workloads and CPU scheduling; they do not support a general near-linear relational scaling claim.

## Setup

| Item | Configuration |
|---|---|
| Capacity | One EC2 Spot host, four NVIDIA L40S GPUs |
| Runtime | Python 3.12, PyTorch 2.9.1, CUDA 13.0, pyg-lib 0.7.0 |
| Model | KumoRelational, pretrained `nvidia/Kumo-Relational` v1.0.1, classification or regression checkpoint only |
| Representation | Native temporal two-hop `[16,16]` sampled `RelatedTables`, disjoint observation graphs |
| Context | Same 1,024 TRAIN observations and complete sampled graphs across placements |
| HM query | First 2,000 VAL observations, eight batches of 250 |
| F1 query | All 499 VAL observations, batches 125/125/125/124 |
| Ensemble | Four members; E1 F1 used only as initial smoke |
| Precision | BF16 autocast, FP32 parameters |
| CPU parallelism | PyTorch threads set to eight |
| Measurement | One warmup pass, three synchronized timed passes per arm |
| I/O boundary | Prepared CPU graphs through GPU inference to CPU predictions; sampling measured separately |
| Source | Coordinator source `6c3f1e22e`; initial E1 smoke used `9f0d6c765` |
| Runner | `relational_bench-a71b1992a.py`, exact content SHA saved in each result |

The runtime lacked cuDF. SDM emitted explicit CPU fallback warnings for GPU string sorting and relational joins. All arms below share that runtime. A separately measured cuDF runtime and process-based scheduling are follow-up comparisons, not explanations proven by this ladder alone.

Native fit preprocesses on CUDA and uses the public multi-estimator CPU cache offload policy. The resident ensemble executor preprocesses once on CPU and uses a placement-independent per-member RNG plan. Therefore the correct ensemble scaling reference is Ensemble 1, and the query-parallel scaling reference is Data 1. Native-to-ensemble quality differences also include that recipe/RNG policy change.

## Inference throughput

Values are median rows per second across the three complete timed passes. The range column retains the minimum and maximum measured pass rather than reporting only the fastest pass.

| Workload | Arm | GPUs | Median rows/s | Three-pass min–max | Speedup over same executor at one GPU |
|---|---|---:|---:|---:|---:|
| HM | Native | 1 | 1,294.66 | 1,196.96–1,295.84 | — |
| HM | Ensemble | 1 | 1,361.95 | 1,361.29–1,363.15 | 1.000 |
| HM | Ensemble | 2 | 1,465.28 | 1,460.73–1,467.12 | 1.076 |
| HM | Ensemble | 4 | 1,577.98 | 1,577.70–1,585.86 | 1.159 |
| HM | Data, threads | 1 | 1,350.21 | 1,243.80–1,350.68 | 1.000 |
| HM | Data, threads | 2 | 1,465.94 | 1,439.54–1,551.96 | 1.086 |
| HM | Data, threads | 4 | 1,211.90 | 1,193.81–1,257.93 | 0.898 |
| F1 | Native | 1 | 371.72 | 344.00–379.08 | — |
| F1 | Ensemble | 1 | 384.04 | 353.36–386.29 | 1.000 |
| F1 | Ensemble | 2 | 379.23 | 350.02–387.83 | 0.987 |
| F1 | Ensemble | 4 | 356.47 | 324.53–356.81 | 0.928 |
| F1 | Data, threads | 1 | 357.41 | 329.82–367.10 | 1.000 |
| F1 | Data, threads | 2 | 377.07 | 338.12–379.07 | 1.055 |
| F1 | Data, threads | 4 | 296.78 | 292.91–314.94 | 0.830 |

The F1 E1 smoke reached 1,123.93 median rows/s, but it does different work from every E4 arm and is not a multi-GPU speedup reference. Its first fit also included cold runtime effects.

## Fit and GPU memory

Fit includes recipe preprocessing, graph processing, context inference, and cache creation. Peak allocator measurements reset separately before fit and before measured predictions. Multi-GPU columns show the maximum across the participating GPUs, not their sum.

| Workload | Arm | Fit seconds | Peak fit GiB/GPU | Peak prediction GiB/GPU |
|---|---|---:|---:|---:|
| HM | Native 1 | 1.576 | 0.830 | 0.459 |
| HM | Ensemble 1 | 2.216 | 0.895 | 0.496 |
| HM | Ensemble 2 | 1.946 | 0.834 | 0.435 |
| HM | Ensemble 4 | 1.985 | 0.770 | 0.371 |
| HM | Data 1 | 1.595 | 0.830 | 0.468 |
| HM | Data 2 | 2.515 | 0.830 | 0.468 |
| HM | Data 4 | 4.025 | 0.830 | 0.468 |
| F1 | Native 1 | 1.783 | 0.759 | 0.326 |
| F1 | Ensemble 1 | 1.669 | 0.849 | 0.388 |
| F1 | Ensemble 2 | 1.730 | 0.786 | 0.326 |
| F1 | Ensemble 4 | 1.773 | 0.720 | 0.259 |
| F1 | Data 1 | 1.785 | 0.759 | 0.335 |
| F1 | Data 2 | 2.890 | 0.759 | 0.335 |
| F1 | Data 4 | 4.861 | 0.759 | 0.335 |

This is a small memory workload relative to a 48 GB L40S. Ensemble placement reduces each device's member-cache footprint while repeating model weights and graph execution. Query parallelism repeats the entire fitted model/cache on each worker and currently fits replicas serially, explaining its longer setup time. Larger contexts are necessary to test memory capacity benefits.

## Prediction quality and placement correctness

| Workload and reference policy | Metric | Result | Placement check |
|---|---|---:|---|
| HM native and Data 1/2/4 | Churn-positive AUROC | 0.661947712862 | Byte-identical predictions |
| HM native and Data 1/2/4 | Log loss | 0.4668493 | Same probabilities and class order |
| HM Ensemble 1/2/4 | Churn-positive AUROC | 0.663524465091 | Byte-identical predictions |
| HM Ensemble 1/2/4 | Log loss | 0.4653579 | Same probabilities and class order |
| F1 native and Data 1/2/4 | Median MAE / RMSE | 3.410428 / 4.211033 | All 499 × 999 quantiles byte-identical |
| F1 Ensemble 1/2/4 | Median MAE / RMSE | 3.325543 / 4.094762 | All 499 × 999 quantiles byte-identical |

All regression outputs were finite and had no quantile crossings. The difference between native and ensemble quality is not a GPU-count effect: they use different recipe/RNG execution policies, while placement within each policy preserves all saved output bytes.

The initial binary scorer used the second encoded category as positive. HM's prediction columns were `['1', '0']`, so the originally logged ensemble AUROC was 0.663542226765 for class 0. Independent rescoring explicitly selected churn class 1, producing the value above. The tiny difference comes from floating-point probability ties. Original result files remain unchanged; `quality-independent-audit.json` records the corrected class-aware scores and raw prediction hashes. The runner was corrected in `3fc2ec0b1` for subsequent measurements.

## Profiling and limitations

The native profile captured CUDA execution and showed substantial copies alongside attention, dense layers, normalization, and graph work. The main-thread PyTorch profiler did not capture CUDA work in the ensemble executor's pre-existing worker threads. Its ensemble trace is therefore only coordinator CPU evidence; it cannot establish GPU compute, synchronization, or communication percentages. Nsight Systems or per-worker profiling is required for that attribution. Named module ranges are inclusive and must not be added to kernel self-times.

The telemetry files sample per-GPU utilization, memory, and power every 200 ms. Initial runs did not record absolute start/end timestamps for the timed prediction passes, so whole-process utilization includes fitting, loading, warmup, and profiler work. It should not be presented as timed inference utilization. Later runner revision `7ce33c801` adds exact prediction windows and invocation files.

These are three repeated prediction passes on one fitted instance per arm in a fixed run order, not independent randomized trials. Small differences deserve further confirmation. Sampling is excluded from inference throughput: it separately cost 0.283 seconds for the HM query prefix and 0.0345 seconds for F1. The [protocol](relational_protocol.md) contains graph counts and setup costs.

## Evidence

External evidence root: `.kumo-multigpu-20261008/results/relational/`. Each `relational-{hm-c1024-b250,f1-c1024-b125}-{native1,ensemble1,ensemble2,ensemble4,data1,data2,data4}-e4` directory contains result JSON, raw prediction arrays, telemetry, and an independent quality/parity audit. Native HM and Ensemble 4 also retain profile outputs. Raw inference targets were unavailable to the query model; evaluation opened VAL labels after inference. TEST was not used.
