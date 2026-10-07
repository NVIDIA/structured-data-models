# TabICL literature: implications for SDM multi-GPU inference

Reviewed 2026-10-08. This is a primary-source literature review, not additional GPU evidence. Paper versions are pinned below; official repository `main` links are mutable observations on the review date. Our measured results and their limitations remain in [context-parallel.md](context-parallel.md).

## Sources and published evidence

### TabICL: architecture gives several different partition axes

Qu et al., *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*, ICML 2025; inspected arXiv v2, 2025-05-24. Sections 3.2–3.5 distinguish column embedding, row interaction, and final ICL. Column embedding uses 128 inducing vectors, with training rows supplying the first attention's keys/values; row interaction then compresses features. Section 3.5 gives overall complexity `O(n² + nm²)` for rows `n` and features `m`. Section 4.4 and Appendix E.2 describe memory-aware batching and offloading. Their batching axes differ: features for column embedding, rows for row interaction, and datasets for ICL. These are architectural opportunities, not reported distributed-inference scaling results. [Paper, §§3.2–3.5, §4.4, Appendix E.2](https://arxiv.org/html/2502.05564v2), [official proceedings](https://proceedings.mlr.press/v267/qu25d.html).

### TabICLv2: preserve context-dependent scaling and distinguish inference regimes

Qu et al., *TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model*, ICML 2026; inspected arXiv v2, 2026-09-16. Section 3 introduces QASSMax: query-dependent scaling also depends on the training-context length. Appendix A.2 retains column/row/ICL stages. Appendix H.1 avoids unnecessary Q/K/V projections. Appendix H.2 reports one-H100 inference on one million rows × 500 features, 80/20 context/query: CPU offload takes 115 seconds with approximately 250 GB RAM; disk offload takes 450 seconds with under 24 GB RAM and 50 GB GPU memory, using 250 GB disk. These runs use AMP and FlashAttention-3. They are complete forward-pass/offloading results, not warm KV-replay or multi-GPU speedups. Section 3 and Appendix I describe 999 regression quantiles and monotonicity handling. I found no 1/2/4/8-GPU inference ladder in the inspected architecture, inference, and experimental sections. [Paper, §3, Appendix A.2, H.1–H.2, I](https://arxiv.org/html/2602.11139v2), [official proceedings](https://proceedings.mlr.press/v306/qu26a.html).

### Official implementation: cache modes and actual precision matter

The current regressor API distinguishes no cache, full column/ICL KV cache, and `kv_cache="repr"`: column KV plus row representations, with the ICL transformer recomputed during prediction. The documented approximately 24× ICL-storage advantage of `repr` belongs to that architecture; it is not a universal multiplier for Kumo's GQA variants. `batch_size` groups ensemble members, not query rows. [Official `regressor.py`, parameter documentation and `_build_kv_cache` / `_batch_forward_with_cache`](https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_sklearn/regressor.py).

`sdpa_with_flattened_batch` flattens leading dimensions, applies scalable-softmax using key length, then selects FA3 or PyTorch SDPA. In the inspected FA3 branch, float32 inputs are explicitly converted to **float16**, and outputs converted back; masks/dropout disable that branch. Thus float32 output tensors do not prove float32 attention arithmetic. The implementation's comment mentioning BF16 should not override the executed `fa_dtype=torch.float16` assignment. [Official attention implementation, function `sdpa_with_flattened_batch`, lines 89–118 in the inspected raw file](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/attention.py).

The changelog also records an AMP-cache dtype mismatch fixed by upcasting cached tensors when loaded without AMP. This supports recording cache creation precision and replay precision separately, rather than treating dtype as a single model setting. [Official changelog, release 2.0.3](https://github.com/soda-inria/tabicl/blob/main/CHANGES.md).

### Localized TabICLv2: approximate context selection, not exact parallelism

Guta, *Localized TabICLv2: Scaling Tabular In-Context Learning through k-NN*, arXiv v1, 2026-08-17. Section 3 retrieves cosine-nearest training rows in Stage-2 embedding space; it uses representation caching and optionally fine-tunes Stages 2–3. Section 4 reports A100 classification experiments, usually `k=32`, with 98.64% accuracy retention, median 2.18× batch speedup, and approximately 249× single-query speedup. Section 3 explicitly gives both full and localized methods the same cached Stage-1/2 representations. Consequently these speedups are **not established against full per-layer KV replay**. Appendix E's serving table must be read with that baseline. The accuracy-retention number is a ratio, not a guarantee of equal predictions, and the study does not establish relational regression quality or multi-GPU scaling. I did not find an author implementation link in the inspected manuscript. [Paper, §§3–4, Appendices D–G](https://arxiv.org/html/2608.16429v1).

### Set Transformer: the useful mathematical boundary

Lee et al., *Set Transformer*, ICML 2019. Section 3.1 defines ISAB using inducing-query aggregation followed by input-query attention to those induced representations, reducing dependence on set size from quadratic to linear for a fixed inducing count. Its MAB includes residuals, normalization, and a feed-forward network. Therefore averaging completed MAB outputs from separately processed shards is not the original computation. [Paper, §3.1, equations 6–10](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf), [official proceedings/code link](https://proceedings.mlr.press/v97/lee19d.html).

## SDM deductions and proposed experiments

The following are our engineering proposals informed by the sources and inspected SDM code. They are not capabilities or speedups claimed by those papers.

| Direction | Proposed SDM implementation boundary | Semantics and required evidence |
|---|---|---|
| Distributed column-summary fit | Extend `sdm/nn/set_transformer.py` at the inducing attention, not around an entire nonlinear block | Merge local attention numerator/normalizer before residuals, normalization, and MLP; then apply the shared global inducing representation to each rank's local rows |
| Distributed row encoding | Partition rows before allocating the large cell/readout buffer in the Kumo and TabICLv2 row encoders | Preserve globally fitted recipes, column ordering, target handling, and inducing summaries; measure fit peak separately from cache replay |
| Representation-cache policy | An explicit research adapter retaining upstream summaries and row embeddings, rebuilding downstream ICL state when needed | Compare against resident full KV, CPU-offloaded full KV, and current placement; never infer the original model's 24× saving for GQA |
| Localized context plus query parallelism | Retrieve context from fixed TRAIN row embeddings; assign independent query/local-context groups to GPUs | Approximate estimator requiring its own quality study; report retrieval/index construction, cache build, fit, and warm prediction separately |
| Larger query blocks within CP | Increase replay query batch only where capacity allows, keeping global context and ensemble plans fixed | Test whether amortizing collective launches helps; do not multiply throughput by replicated ranks or assume our unmeasured long-MHA crossover |

### Exact upstream sharding is more promising than independent shard fits

Our existing CP implementation only distributes retained final-ICL keys/values. It leaves row-embedding/GNN fit activations replicated. A stronger capacity experiment should avoid materializing the full `[..., rows, columns, channels]` buffer on every GPU in the first place.

At each induced-attention layer, all ranks could share the inducing queries, attend to their local TRAIN keys/values, combine stable output/LSE statistics, and then continue with the globally combined inducing states. The communication size would depend on inducing queries and feature batches, rather than transmitting every context row's full activation at that boundary. This does not eliminate repeated column passes or guarantee an acceleration.

SDM-specific correctness requirements are substantial: `RowEmbedding.forward` may subsample `max_keys`; select the **same global sample** before assigning ownership, rather than sampling independently per rank. Preserve mixed-radix class handling and global target vocabulary. QASSMax must receive the global eligible key count, not local shard length. Merge attention statistics before any nonlinear update. Empty/uneven ownership and differing valid-row counts need explicit tests.

For KumoRelational, distributing whole disjoint sampled example graphs could avoid cross-rank GNN edges in that specific representation. General shared-entity graphs require boundary exchange or an equivalent ownership scheme. Independent preprocessing or column-summary fits on each graph shard would change the estimator. This proposed extension needs graph-level parity tests and is not implemented in the current CP prototype.

### Precision and quality: the literature does not waive our failure

Our BF16 F1 CP experiment failed the declared all-999-quantile numerical screen even though point metrics were close. FP32 local attention/merge did not repair it. Neither TabICL paper supplies a distributed-KV numerical guarantee that would justify relaxing that gate. For future kernel comparisons, record parameter dtype, Q/K/V dtype, local output and LSE dtype, merge dtype, actual backend, and output dtype independently. Disable any hidden low-precision backend conversion in a purported full-FP32 control.

Retain separate native, single-rank same-kernel, and multi-rank controls. Validate every quantile, class probabilities and ordering, finite values, monotonicity, repeat stability, and task metrics. A classification-only retrieval result cannot validate relational quantiles. In external-library comparisons, equal ensemble counts alone are insufficient: match preprocessing plans, cache policy, queried rows, precision, and timing boundary.

## Priority and evidence limits

First complete the already specified full-FP32 numerical isolation and unmeasured all-gather/long-MHA controls when authorized compute access is available. Next investigate globally coordinated column-summary/row-encoding fit partitioning, which addresses a memory stage our current replay-only CP does not. Treat representation caching and localized retrieval as distinct memory/quality policies, not transparent replacements for full-context inference. Retain ensemble/query parallelism as the baseline throughput alternatives.

No GPU jobs or core source changes were made for this literature review. The papers' single-device scale figures and official repository's multi-GPU **fine-tuning** support are not evidence of distributed inference speedup. [Official repository, fine-tuning documentation](https://github.com/soda-inria/tabicl). All proposed SDM extensions above require new tests; none are included in our measured results.
