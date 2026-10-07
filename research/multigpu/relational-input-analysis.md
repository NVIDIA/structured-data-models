# Native relational input and memory investigation

This investigation uses actual sampled RelBench input, current SDM `842c408fe`, the shared TRAIN IDs, exact temporal `[16,16]` sampling, and the default KumoRelational recipe. Graphs remain native related tables with disjoint `__example__` linkage. No graph flattening or GPU execution is involved. Measurements were performed locally with PyTorch 2.9.1, pyg-lib 0.7.0, and eight CPU threads; they are not EC2 throughput measurements.

## Actual graph expansion

| Task | TRAIN context | Native sampled nodes | Directed edges | Graph-index storage | Largest BF16 row-embedding buffer | BF16 GNN statistics | BF16 ICL K/V per member |
|---|---:|---:|---:|---:|---:|---:|---:|
| rel-hm/user-churn | 1,024 | 19,887 | 39,828 | 1.06 MiB | 0.057 GiB | 0.095 GiB | 0.023 GiB |
| rel-hm/user-churn | 16,384 | 319,381 | 641,272 | 17.11 MiB | 0.918 GiB | 1.523 GiB | 0.375 GiB |
| rel-hm/user-churn | 65,536 | 1,279,622 | 2,571,320 | 68.62 MiB | 3.677 GiB | 6.102 GiB | 1.500 GiB |
| rel-f1/driver-position | 4,096 | 172,492 | 523,126 | 13.29 MiB | 0.196 GiB | 0.823 GiB | 0.094 GiB |

Node/edge counts and index bytes are measured from `TaskGraph`. The three rightmost columns are explicit tensor-shape estimates, **not measured GPU peak memory**. They exclude other concurrently live tensors, attention intermediates, preprocessing, allocator reservation and model parameters. The row-embedding buffer is `rows × (numeric_features + 4) × 128 × 2` bytes; numerical feature counts include relative-time and task-feature injections. GNN statistics are `nodes × 5 × 512 × 2` bytes. ICL K/V is `context × 12 layers × 2 K/V × 512 × 2` bytes, assuming BF16 storage.

At 65,536 context, rel-hm has 571,256 article rows, 65,536 customer rows and 642,830 transaction rows. Post-recipe numerical widths are respectively 23, 4 and 5; the model adds one relative-time feature to transactions. Despite a much smaller raw database, rel-f1 expands to about 42 graph nodes per task row, compared with about 19.5 for rel-hm. Raw dataset size alone therefore does not predict inference memory.

## CPU input measurements

The runner's native preparation completed rel-hm at contexts 1,024, 16,384 and 65,536 in approximately 5–6 seconds each, including raw database loading, sampler construction, context/query sampling, graph identity hashing and serialization of 256 query rows. Maximum process RSS was approximately 4.4–4.7 GB and the system reported no swap events. These are one-off local preparation observations, not repeated runtime benchmarks. The reusable full-database sampler dominates preparation memory, so preparation peak was similar across those three context sizes.

For the 65,536-row sampled context, a separate run measured one cold `TaskGraph` construction at 0.511 seconds and three warm repetitions at 0.088, 0.078 and 0.078 seconds. One-member default recipe processing took 0.667 seconds. Full CPU process peak RSS for that sampled-input profile was 1.77 GB. A 16,384-context standalone one-member profile measured recipe processing at 0.278 seconds. Other exploratory profiles ran concurrently and their timings should not be used to infer scaling.

## What the implementation reveals

- **The sampler deliberately duplicates shared entities across task examples.** It samples with `disjoint=True` and adds composite `__example__` keys. Replacing this with a globally deduplicated graph changes target injection, temporal ownership and task assignments; it is not a safe memory optimization without a new model contract.
- **Ensemble parallelism reduces persistent cache per GPU but retains one member's full graph.** At 65,536 context, E8 final ICL K/V alone is approximately 12 GiB. Distributing members across four GPUs reduces that component to roughly 3 GiB per GPU, while each active member still needs its row embedding and GNN buffers. GPU measurement must determine the combined peak.
- **ICL-only context parallelism cannot remove the upstream graph peak.** The 6.10 GiB GNN statistics tensor and 3.68 GiB largest row-embedding buffer arise before final ICL attention. A design that performs complete row embedding/GNN on every rank and shards only ICL caches duplicates that work and those allocations.
- **Query data parallelism is semantically simple once fitted state is fixed.** Split whole task examples and their disjoint neighborhoods; each worker keeps the complete fitted estimator cache. This increases aggregate cache memory but distributes query graph encoding and GNN computation as well as attention.
- **Graph topology is independent of ensemble feature permutations.** `TaskGraph.from_input` reconstructs relationships, performs composite joins, sorts edges and propagates task ownership on each call. Reusing immutable topology/task ownership for identical sampled input can eliminate repeated graph work across ensemble members, provided identity-changing processors are excluded or invalidate it. This should live at the sampled-input/model boundary, with explicit ownership rather than a global cache.
- **Row embedding is an attractive memory target.** The full `[rows, columns+4, 128]` buffer is allocated before attention's existing automatic batch limits take effect. A chunked row-encoding design could first compute each column layer's global inducing summaries from the same selected TRAIN keys, then apply that layer to bounded row chunks. It must preserve the global preprocessing fit, selected keys and generator state; independently fitting a recipe/inducing summary per chunk changes predictions.
- **GNN statistics projection could avoid its largest intermediate.** `segment_multi_reduce` produces five statistics per node and immediately feeds a linear map from `5×512` to `512`. Computing destination blocks or fusing statistics projection could reduce the materialized `[nodes,5,512]` tensor. Preserve all five statistics and compare against the existing kernel before claiming equivalence. The current code already frees statistics before the next hop and selects readout nodes before the last normalization, so those optimizations are not new opportunities.
- **Context sharding before graph encoding needs more than graph partitioning.** Although sampled neighborhoods are disjoint, row-embedding column attention fits global inducing summaries and the recipe fits global statistics. An exact distributed design needs shared/global preprocessing and distributed inducing attention; independent fits on graph shards are an approximate method whose quality must be measured separately.

## Reproduction and evidence

`profile_relational_input.py` consumes `graphs.pt` and `workload.json` from `relational_bench.py prepare`. It runs CPU graph construction and the default recipe, writes graph dimensions and explicit memory estimates, and never loads model weights or starts a GPU job. Full data preparation and model provenance are in `workloads.md`.

Raw profiles are under `/Users/ardrianw/repositories/.kumo-multigpu-20261008/cpu-profiles/` in `rel-hm-c1024-q256`, `rel-hm-c16384-q256`, `rel-hm-c65536-q256` and `rel-f1-c4096-q499`. Prefer the `input-profile-*-repeat.json` records for sequential warm graph timing; the original E8/1k/F1 profiles overlap in execution and are useful for sizes only. The runner's `max_rss_kib` field contains bytes on macOS; this profiling script normalizes its separate `cpu_max_rss_bytes` field correctly.
