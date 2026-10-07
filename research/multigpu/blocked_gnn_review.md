# Independent bounded-memory GNN review

Scope: research-only destination blocking of the existing KumoRelational five-statistic GNN. This is not a new reducer, graph sampler, checkpoint, or measured GPU speedup. CUDA and native full-model validation remain required before promotion.

## Semantics that must remain unchanged

The baseline applies `src_lin` to every source row, adds the fitted edge-type embedding to each message, and reduces complete destination neighborhoods in CSR order. Statistics are concatenated in the order **sum, mean, standard deviation, minimum, maximum**. For FP16/BF16 input, message addition and reduction use FP32; the five-statistic tensor is then cast back to the source dtype **before** the existing `torch.addmm` projection. Moving projection inside the reduction changes this rounding boundary.

Mean divides by the full destination degree, clamped to one for empty segments. Variance is population `E[x²] - E[x]²`, not an unbiased sample estimate. Standard deviation is exactly zero when variance is at most `1e-5`; otherwise it is the square root of clamped variance. Empty segments produce all-zero statistics. Infinite extrema are replaced with zero, while NaNs propagate. These details are inherited safely by reusing the existing reducer rather than introducing a separate implementation.

`src_lin` bias participates once per message before the reductions. `skip_lin` bias participates once per destination. The statistics projection has no bias. Normalization and GELU occur after projected statistics plus skip, and the final hop selects the requested readout rows before normalization. Edge-type embeddings are generated once per GNN call when recording, or reused from the fitted cache; blocking must not consume additional random numbers.

## Why complete destination blocks are viable

Each block must retain every incoming edge of its destination rows, including duplicate edges and edge types, in the original order. The CUDA reducer supports absolute offsets into the unchanged global source-index and edge-type arrays. CPU eager reduction instead needs sliced edges and rebased offsets to avoid re-gathering all messages for every block. No merge of partial means, standard deviations, or extrema is necessary.

Projecting each complete `[block_rows, 5, channels]` statistics block immediately reduces the largest statistics allocation from `N × 5 × channels` to `block_rows × 5 × channels`. This does not remove full source projections, output embeddings, row-encoder state, model weights, or ICL caches. The previous 6.10 GiB H&M C64k statistics estimate is one allocation, not a promise of equal end-to-end peak reduction. Delete each block's statistics before computing the next block to avoid overlapping two such allocations.

Changing destination GEMM dimensions can select different CPU/CUDA implementations and alter reduced-precision rounding. Algebraic equivalence therefore does not imply byte identity. Validate FP32 intermediate values first, then BF16 full-model predictions and quality against the same fitted member/RNG plan. CPU fallback tests alone do not validate the Triton absolute-offset path or CUDA peak memory.

## Immutable topology reuse: separate opportunity

`TaskGraph.from_input` rebuilds joins, graph topology, task readout mapping, and task-owner propagation for each transformed member. An explicit prepared topology object could reuse those integer structures when table/observation IDs, row order, relationship schema, task links, hops, device, and sampled temporal neighborhoods are identical. Preserve `__example__` disjoint identities and original edge order; do not deduplicate entities across observations.

The current `TaskGraph` also holds feature tables. Reusing that object unchanged would accidentally reuse member-specific transformed features. A safe design must bind fresh `x` and `related_tables` to cached integer topology and ownership maps. Task-conditioned row embeddings, label-derived features, selected TRAIN keys, and edge-type embeddings are not generally reusable across member fits. Changes to identifiers/order/schema invalidate topology. Canonical re-sorting is itself a numerical-policy change and should not be bundled into the first reuse prototype.

No topology-reuse implementation or speedup claim is part of this review.

## Independent CPU result

The real classifier and regressor GNN weights from KumoRelational v1.0.1 (`2bd603d3d8f25f67a7aaa8579908595e20567e22`) were tested on a deterministic 64-node, 512-channel synthetic graph with degrees 0/1/3/17, three edge types, duplicate readout rows, two hops, and frozen-cache query reuse. The inputs are synthetic, not a RelBench quality evaluation. Three block sizes (1/17/256) were tested with FP32 parameters, BF16 parameters, and FP32 parameters under CPU BF16 autocast. Edge RNG seed 1729 aligns the two implementations.

The final matrix also includes 257-node graphs with blocks of 64 and 128, exercising five and three blocks respectively. Of 31 checks, 29 passed. The exact statistics test includes empty and singleton destinations and variance below/above the `1e-5` threshold. All BF16 and BF16-autocast checkpoint cases passed `rtol=0.02, atol=0.02`; FP32 blocks 17/256 at 64 nodes and blocks 64/128 at 257 nodes passed `rtol=1e-5, atol=2e-6`. **FP32 block size 1 failed the unchanged gate for both checkpoints**, with maximum reported violating fit differences 1.26361847e-5 (classifier, 117/2560 elements outside tolerance) and 7.03334808e-6 (regressor, 59/2560). The classifier frozen-query maximum difference was 1.66893005e-5. The implementation already computed native full-size `skip_lin` in this check; the shaped statistics-projection GEMM remains a plausible rounding source. This does not establish exact numerical equivalence for all block sizes.

The optional real-checkpoint tests deliberately retain those failures rather than broadening tolerance. Set `SDM_RELATIONAL_CHECKPOINT_ROOT` to the offline snapshot directory to enable them. Without that variable they skip checkpoint-dependent cases. Set `SDM_GNN_REVIEW_OUTPUT` to an external directory to save one small JSON per checkpoint/precision/graph/block combination, including checkpoint and source SHA256, exact graph specification and edge seed, tolerances, fit/query error statistics and pass/fail. No checkpoint weights are copied into the receipts.

The final 30 JSON receipts are in `.kumo-multigpu-20261008/blocked-gnn-cpu-review/`; final JUnit evidence is `.kumo-multigpu-20261008/blocked-gnn-checkpoint-final.xml`. Adapter SHA256 is `b19353e17ff432371efcc9affc53a6151b50fc94e1d11b4ba2badac2a4c2c47c`. Run after integrating the prototype and test into the same source tree:

```sh
SDM_RELATIONAL_CHECKPOINT_ROOT=/path/to/offline/snapshot \
SDM_GNN_REVIEW_OUTPUT=/path/to/new/receipts \
OMP_NUM_THREADS=4 PYTHONPATH=. \
python -m pytest -q research/multigpu/test_blocked_gnn_review.py
```

The benchmark runner exposes `--gnn-block-size`, default zero (native unchanged), for future controlled GPU comparisons. Positive values install the adapter on all inner replica cores before fitting; negatives are rejected. No CUDA path, end-to-end prediction quality, speed, or peak-memory claim is validated by this CPU review.
