# Relational foundation-model literature: implications for SDM inference

Reviewed 2026-10-08 using primary papers, author-maintained repositories and official documentation. This is a research/design note, not a new benchmark. Published results below belong to their authors and protocols; proposed SDM changes are explicitly hypotheses.

## What transfers to the measured problem

Our measured native H&M capacity workload expands 65,536 context examples into 1,279,607 nodes and 2,571,320 directed edges. An earlier CPU-profile sample used a different seed and had 15 more article nodes; the [exact workload provenance](blocked-gnn.md) distinguishes them. On L4, the tested E4 one-GPU and four-GPU ICL-layer configurations both failed at the GNN's 6.10 GiB statistics allocation. This is an observed allocator/runtime boundary, not proof that every implementation must fail. Separately, replacing Arrow joins with cuDF in the same environment slowed the small-context native/EP arms by 14–22% and introduced prediction-repeatability differences. See [graph and memory evidence](relational-input-analysis.md) and [backend evidence](cudf-runtime.md).

The useful literature themes are consequently: bounded relational input, task-conditioned hierarchical encoders, reusable preparation, support/context selection, and independently parallelizable predictions. Larger database capacity, larger model context, more pretraining GPUs, and faster single-request inference are four different claims.

## Verified model lineage and source limitations

Official [SDM KumoRelational API documentation](https://nvidia.github.io/structured-data-models/api/generated/sdm.models.KumoRelational.html) calls the implementation an **adapted** KumoRFM-2 model. It describes a TabICLv2-style row encoder, schema-agnostic GNN and dataset-level ICL block. This supports architectural lineage, not identity between the downloaded `nvidia/Kumo-Relational` checkpoint and a commercial KumoRFM model, paper configuration or SDK service. Our checkpoint identity remains the exact revision recorded in [workloads.md](workloads.md).

The original *KumoRFM: A Foundation Model for In-Context Learning on Relational Data* URL now redirects to an [NVIDIA overview](https://docs.nvidia.com/sdgm/rfm/overview). The current [official research summary](https://docs.nvidia.com/sdgm/research/kumorfm-paper) discusses the 2025 work, but its paper link resolves to the 2026 KumoRFM-2 arXiv paper. We did not recover/inspect a verified full 2025 manuscript and do not cite the redirect as one. The summary's row-encoder → relational-transformer → ICL decomposition is useful historical context only.

## Primary reading notes

### 1. KumoRFM-2: Scaling Foundation Models for Relational Learning

Hudovernik et al.; [arXiv 2604.12596v1](https://arxiv.org/html/2604.12596v1), April 2026. Read §§2–3, §4.4 and §5. The abstract page lists v1 as the current version.

**Reported architecture/system:** task information enters early; table-level row/column processing precedes foreign-key and cross-sample processing. Context selection mixes entity-local history and recent global examples. Ensembling includes column/class permutations and hop choices. The system describes SQL pushdown or a memory-mapped retrieval engine for very large databases, plus stateless replicated serving. These are not demonstrations of splitting one prediction across GPUs. The paper explicitly distinguishes massive database size from the still-open problem of million-example model contexts; benchmark contexts are capped at 10,000.

**Protocol caution:** its reported benchmark setup can populate context from TRAIN and VAL before TEST evaluation. Our experiment evaluates VAL, so that recipe cannot be copied: only TRAIN-derived labels may enter context. [Paper §§2–4](https://arxiv.org/html/2604.12596v1).

**SDM inference:** pursue prepared-input reuse and bounded encoder work before enlarging final attention context. Early task conditioning means encoded features are not automatically reusable across tasks, label changes or context fits. The design suggestions below require SDM-specific equivalence tests.

### 2. Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data

Ranjan et al.; [arXiv 2510.06377v3](https://arxiv.org/html/2510.06377v3), March 2026 revision, ICLR 2026. Read §§3, 4.1, 4.5 and Appendix A. [Official implementation](https://github.com/stanford-star/relational-transformer) identifies `rt-v1` as the legacy paper implementation; current main also contains newer RT-J recipes.

**Reported:** cell tokens combine values and schema metadata; distinct attention patterns cover columns, rows/parents and children. Its sampler prioritizes foreign-key-to-parent links, limits child expansion, respects time and stops at a cell budget, with a Rust implementation. §4.5 reports cost–quality curves and attributes part of RT's efficiency to FlexAttention. §4.1 reports training on eight A100s; that is training evidence, not an inference strong-scaling curve.

**SDM inference:** introduce a measured row/cell/byte budget alongside fanout rather than assuming a fixed hop count bounds memory. Borrow the preparation/kernel separation, not RT's cell-attention architecture as a drop-in replacement for Kumo checkpoints. Changing which rows are sampled changes the predictor and needs quality evaluation. [Paper §§3–4, Appendix A](https://arxiv.org/html/2510.06377v3).

### 3. RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions

Ranjan et al., 2026; [author project page](https://star-project.stanford.edu/rt-j/) and [official repository](https://github.com/stanford-star/relational-transformer). The linked [OpenReview manuscript](https://openreview.net/forum?id=oQINTd9din) required browser verification; full-paper inspection was unavailable. The following is therefore verified author-page/repository evidence, not a claim to have read the full paper.

**Reported:** an 85M-parameter RT model uses varied training context lengths, target-only masking and random-walk retrieval. Independent sampled-context ensembling and validation-based context tuning improve quality. The project states an 8,192-cell context limit and training on up to eight B200s; it does not provide a reviewed multi-GPU inference scaling curve. Its few-shot claims concern label efficiency, not exact equivalence to full-context inference. [Project sections “How RT-J is trained” and “Spending more compute at inference”](https://star-project.stanford.edu/rt-j/).

**SDM inference:** compare multiple small, independently selected TRAIN contexts against one large context at equal total labels, encoded nodes and GPU-seconds. This is context/support ensembling, not the exact same-context member parallelism already tested. The generic executor could host either, but metadata and quality claims must distinguish them.

### 4. OpenRFM: Dissecting Relational In-Context Learning

Chen et al.; [arXiv 2606.04320v1](https://arxiv.org/html/2606.04320v1), June 2026. Read §§3–5, especially §§4.1, 5.2 and 5.3.

**Reported:** sparse relation-level label coverage motivates a second, batch-level ICL stage using a pretrained tabular model. A normalized bridge performs better than the examined simple linear bypass. Context ablations use support sizes 256–1,024, E1/E8 and random/balanced/temporal selection; query-agnostic support is chosen with K/V reuse in mind. The pipeline comparison separates preparation from inference on one RTX A6000 and acknowledges higher inference cost than a row-level GNN. No multi-GPU inference evidence was found in those experiments. [Paper §§4–5](https://arxiv.org/html/2606.04320v1).

**SDM inference:** fixed, query-independent support improves cache amortization, but a balanced classifier support may change calibration. Do not assume normalization or a new bridge can be grafted into Kumo weights without training. Evaluate preprocessing/fit once and many query batches separately from cold end-to-end latency.

### 5. Griffin: Towards a Graph-Centric Relational Database Foundation Model

Wang et al.; [arXiv 2505.05568v2](https://arxiv.org/html/2505.05568v2), June 2025 revision, ICML 2025. Read §§3.1–3.3 and §4. [Official code and commands](https://github.com/yanxwb/Griffin).

**Reported:** shared semantic encoders/decoders support different schemas and tasks. Task-conditioned cross-attention extracts row information; message passing aggregates within relations before combining relation outputs. Large pretraining-corpus node counts do not mean one full graph is resident in a forward pass. Its official README shows Accelerate training commands and multi-GPU experiment parallelization, not a measured one-request inference speedup. [Architecture](https://arxiv.org/html/2505.05568v2), [repository README](https://raw.githubusercontent.com/yanxwb/Griffin/main/README.md).

**SDM inference:** relation-aware reduction order is an architectural design option for future models; replacing the current five-statistic GNN with Griffin's reducer changes the model. The checkpoint-preserving systems analogue is to compute the existing statistics in destination blocks and immediately project them, retaining all statistics and edge semantics.

### 6. Relational In-Context Learning via Synthetic Pre-training with Structural Prior (RDB-PFN)

Wang et al.; [arXiv 2603.03805v5](https://arxiv.org/html/2603.03805v5), May 2026 revision; [ICML proceedings](https://proceedings.mlr.press/v306/wang26jz.html); [official code](https://github.com/MuLabPKU/RDBPFN). Read §6.2, Appendix C.2 and Appendix D.2.

**Reported:** DFS aggregates relational inputs for a compact transformer. The newer manuscript explicitly says much of its default-baseline speed advantage comes from avoiding costly ensembles. Appendix D.2 splits larger supports into chunks, independently predicts and averages probabilities, with generally improving quality and diminishing returns from effective support 1,024–8,192. Appendix C.2 reports eight-4090 adaptation pretraining, not distributed inference. [Paper](https://arxiv.org/html/2603.03805v5).

**SDM inference:** this is direct inspiration for an approximate small-context ensemble, and a warning to match ensemble breadth when comparing throughput. DFS-plus-KumoTabular is a separate model/input baseline, not an optimization of native KumoRelational: it loses some row-level relational interactions and adds materialization cost. Preserve that distinction in both APIs and tables.

## What distributed evidence actually establishes

| Source | Evidence reviewed | What it does not establish |
|---|---|---|
| KumoRFM-2 §2 | Database retrieval scalability and horizontally replicable serving | Single-query tensor/context/graph parallel speedup |
| RT §4.1 | Eight-A100 training | An inference scaling curve |
| RT-J author page | Up-to-eight-B200 training; context ensembles | GPU count, implementation or speedup of those ensembles |
| OpenRFM §5.2 | One-A6000 end-to-end pipeline comparison | Multi-GPU capacity or throughput |
| Griffin official README | Multi-GPU training/parallel experiments | One-query distributed execution |
| RDB-PFN Appendix C.2/D.2 | Multi-GPU training and support ensembling | Exact full-context attention equivalence or inference scaling |

These distinctions are drawn from the linked sections above. We found useful algorithms and evaluation lessons, but no reviewed result that directly validates SDM's particular multi-GPU implementation. Our own measured scaling and numerical tests remain necessary.

## Prioritized SDM experiments: proposals, not published results

| Priority | Proposed change | Exactness contract | Why it addresses our measurements |
|---|---|---|---|
| P0 | Destination-block GNN statistics followed immediately by existing projection | Preserve all five reductions, weights, CSR neighborhoods and empty-segment behavior; specify floating-point tolerance | Avoid materializing the observed 6.10 GiB full statistics tensor |
| P0 | Prepared immutable topology and task-owner mappings | Same IDs, order, timestamps, relationships, hops and disjoint-example identity; invalidate on identity-changing processing | Remove repeated joins/builds rather than assuming another dataframe backend is faster |
| P1 | Partition disjoint example graphs after globally consistent row encoding | Preserve global recipe state, selected TRAIN keys and induced-attention summaries | Reduce per-device graph work upstream of final ICL; final attention-only placement does not do this |
| P1 | Fixed-support query data parallelism with bounded batches | Same fitted estimator, support and query order restored at collection | Amortize fit and distribute graph/query computation without changing context |
| P1, approximate | Smaller-support ensembles or retrieval-selected support | New predictor, explicit support/member identities and calibration/quality evaluation | Trade graph size per member against ensemble breadth and total label coverage |
| P2, approximate | Relation-aware or row/cell-budgeted sampling | New sampler; preserve temporal safety but do not claim exact baseline equivalence | Bound heterogeneous graph expansion and investigate quality per encoded node |
| P2, new model | Alternative bridge, hierarchical reducer, DFS+tabular predictor or fine-tuned head | New checkpoint/model identity and separate quality baseline | Longer-term architecture choices, not runtime patches to existing weights |

### Details that prevent incorrect “exact” optimizations

1. **Chunk GNN destinations, not arbitrary edge subsets followed by averaging.** Mean/std need their full counts and sufficient statistics; extrema and empty segments need matching handling. A different reduction order may produce BF16 differences. Compare FP32 intermediate outputs first, then BF16 end predictions. Griffin motivates structured aggregation, but does not prove this proposed implementation exact.
2. **Cache topology, not arbitrary learned embeddings.** Current task-conditioned row embeddings depend on TRAIN labels, fitted recipe state and ensemble RNG. A topology cache must retain original per-example duplication; deduplicating customers or products across examples changes target propagation. Preserve the working Arrow edge order initially, since canonical sorting itself can alter reduction results. Diagnose the cuDF repeatability issue before assigning its cause to ordering alone.
3. **Do not shard column-attention fit independently.** In SDM, each related table's row encoder can select up to 20,000 eligible TRAIN keys. Per-shard fitting changes both sampled keys and global summaries; an exact distributed design must coordinate those choices. Only after this coupling is respected can independent disjoint graph components be partitioned without silently creating a different predictor.
4. **Reuse and retrieval compete.** A fixed support provides reusable fitted caches. Query-specific retrieval may improve relevance but creates more fits and weaker amortization. Measure request-level costs, not just an isolated attention kernel. Shared supports per temporal cohort offer a concrete intermediate experiment, with cohort construction using TRAIN information only.
5. **Cache invalidation is part of correctness.** Keys should include data snapshot, query cutoffs, sampled row identities/order, graph schema, sampler configuration, processor identity and any label-dependent state relevant to the cached object. Avoid global process caches or a serving framework in core SDM; prefer explicit prepared objects at the model/input boundary.

## Minimal next measurement plan

First finish the unexecuted graph-repeat diagnostic and run a destination-block reduction prototype against current native tensors. Use H&M contexts 1k/16k/64k, identical `[16,16]` prepared graphs, one and four GPUs, and both fit and query phases. Re-run fresh processes with documented allocator settings; record failures as outcomes. Track Torch peaks, sampled device memory, host RSS, join/topology/recipe/row-encoder/GNN/ICL intervals and communication bytes. Memory capacity, latency and throughput need separate conclusions.

Then test the approximate frontier: one large support versus several smaller supports with the same union of TRAIN IDs; separately compare random, class-balanced and recent-history supports. Keep total GPU-seconds and encoded-node budgets visible. Add one regression workload because support ensembles can behave differently from classification. Report AUROC/AP/log loss/calibration for classification and MAE/RMSE for regression, plus hard-prediction agreement and probability differences only where exactness is expected.

Keep TEST untouched. Use a TRAIN-only inner split to choose retrieval/sampling/ensemble settings, then score the fixed VAL queries. Published numbers from different checkpoints, context-selection rules, label access or train/validation/test protocols are inspiration, not entries in the same benchmark ranking.
