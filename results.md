Classification (sorted by accuracy):

┌────────────────────┬──────┬─────────┬──────────┬─────────────────────────────┐
│ Dataset │ Rows │ Classes │ Accuracy │ Notes │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ connect-4 │ 67K │ 3 │ 15.8% │ full size — below random ⛏️ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ bank-marketing │ 45K │ 2 │ 33.0% │ full size — below random │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ LED-display │ 500 │ 10 │ 44.7% │ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ cmc │ 1473 │ 3 │ 46.8% │ ⛏️ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ yeast │ 1484 │ 10 │ 53.6% │ ⛏️ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ mnist_784 │ 10K\* │ 10 │ 61.0% │ \*capped, OOM'd at full size │
│ │ │ │ │⛏️ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ wine-quality-white │ 1599 │ 6 │ 67.3% │ ⛏️ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ eucalyptus │ 736 │ 5 │ 69.7% │ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ diabetes │ 768 │ 2 │ 71.4% │ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ credit-g │ 1000 │ 2 │ 73.0% │ ⛏️ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ mfeat-fourier │ 2000 │ 10 │ 91.0% │ │
├────────────────────┼──────┼─────────┼──────────┼─────────────────────────────┤
│ everything else │ │ │ 98%+ │ solved │
└────────────────────┴──────┴─────────┴──────────┴─────────────────────────────┘

Regression (sorted by R²):

┌────────────────────┬──────┬───────┬─────────────────────────────┐
│ Dataset │ Rows │ R² │ Notes │
├────────────────────┼──────┼───────┼─────────────────────────────┤
│ california housing │ 8885 │ 0.065 │ barely any signal │
├────────────────────┼──────┼───────┼─────────────────────────────┤
│ pol │ 576 │ 0.375 │ ⛏️ │
├────────────────────┼──────┼───────┼─────────────────────────────┤
│ concrete │ 10K\* │ 0.472 │ \*capped, OOM'd at full size │
├────────────────────┼──────┼───────┼─────────────────────────────┤
│ superconductor │ 4209 │ 0.486 │ ⛏️ │
├────────────────────┼──────┼───────┼─────────────────────────────┤
│ house_prices │ 166K │ 0.500 │ full size ⛏️ │
├────────────────────┼──────┼───────┼─────────────────────────────┤
│ particulate-matter │ 4440 │ 0.636 │ │
├────────────────────┼──────┼───────┼─────────────────────────────┤
│ space_ga │ 3107 │ 0.793 │ │
└────────────────────┴──────┴───────┴─────────────────────────────┘

1. Vanilla forward pass → log metrics (round 0 baseline)
2. Extract pre-head embeddings, compress, inject via KV hooks → log metrics (round 1)
3. Extract again from round 1's output, compress, inject → log metrics (round 2)
4. Repeat → log metrics (round 3)
5. Repeat → log metrics (round 4)

Compression:
Simplified than learned MLP in the paper.

- Mean pooling — average all query row embeddings into 1 feedback token. Cheapest, loses per-row detail.
- Top-K by entropy — keep the embeddings of the K most uncertain query rows. Preserves detail where it matters most.
- No compression — feed all query row embeddings back. Expensive, but cleanest signal for whether the idea works at all.

Metrics:

- Prediction performance — accuracy + log-loss for classification, RMSE + R² for regression. Per round.
- Prediction entropy — are predictions getting sharper each round? Decreasing entropy = resolving uncertainty.
- Embedding drift — cosine distance between pre-head embeddings from round N to round N+1. Convergence = stabilization, continued drift = something's still changing.

## Experiment 1: Feedback Token Injection for TabICLv2

Context

We want to test whether feeding back a model's own hidden states ("feedback tokens") can improve TabICLv2 predictions at inference time, inspired by the latent chain-of-thought paper (arxiv 2605.11262v2). The sweep identified candidate datasets with moderate performance where there's room for improvement.

Goal

Write a standalone experiment script that:

1. Runs vanilla TabICLv2 (round 0 baseline) on each candidate dataset
2. Extracts pre-head embeddings, compresses them into feedback tokens
3. Reinjects feedback tokens into the ICL block's attention layers via hooks
4. Repeats for 4 rounds, logging metrics per round
5. Compares 3 compression variants: mean pooling, top-K by entropy, no compression
6. Each compression variant runs independently from scratch (not chained)

Candidate datasets

Classification:

- cmc (OpenML 23, 1.4K rows, 3 classes, 46.8% accuracy)
- yeast (OpenML 181, 1.5K rows, 10 classes, 53.6% accuracy)
- wine-quality-white (OpenML 40691, 1.6K rows, 6 classes, 67.3% accuracy)
- credit-g (OpenML 31, 1K rows, 2 classes, 73.0% accuracy)

Regression:

- pol (OpenML 546, 576 rows, R²=0.375)
- superconductor (OpenML 42570, 4.2K rows, R²=0.486)
- house_prices (OpenML 41540, 166K rows,

Implementation approach

Extraction: Hook model.{cls,reg}\_model.icl_block.head with register_forward_pre_hook to capture pre-head embeddings (shape [..., R_query, 512]).

Compression (3 variants, each run independently):

- mean_pool: average all query embeddings into 1 token [..., 1, 512]
- top_k: keep embeddings of K=10 query rows with highest prediction entropy [..., 10, 512]
- no_compress: use all query row embeddings as-is [..., R_query, 512]

Injection: Hook each of the 12 icl_block.layers[i] with register_forward_pre_hook(hook, with_kwargs=True). The hook intercepts the key_value kwarg (a raw tensor of shape [..., R_train, 512]) and concatenates feedback tokens along the sequence dimension: torch.cat([key_value, feedback], dim=-2). The layer's existing LayerNorm and K/V projection handle normalization. This makes query rows attend to both context rows and feedback tokens.

Round loop (per dataset, per compression variant):

- Round 0: vanilla forward pass (no hooks on layers), capture embeddings → baseline metrics
- Rounds 1-4: inject previous round's compressed embeddings via layer hooks, capture new embeddings → per-round metrics

Metrics per round:

- Classification: accuracy, log-loss, prediction entropy
- Regression: RMSE, R²
- Both: embedding drift (mean cosine distance between current and previous round's pre-head embeddings)

Output: Per-dataset per-variant table showing metrics per round. Summary table at end showing round-4 gain over baseline. CSV export.

## Experiment 2: Feedback Token Injection via query rows

Changes from previous experiment:

1. adapt injection
2. distinguish noise from real signal

Injection: After each round's forward pass, take the query rows and pair them with the model's predicted labels to form pseudo-labeled rows. Concatenate these onto the original context, creating an enlarged context set. Run a standard forward pass with the enlarged context and the same query rows. The model processes the pseudo-labeled rows through the same path it uses for real context rows — no hooks, no KV manipulation. The query rows effectively get to see the model's own prior predictions as additional labeled examples.

Variance:
Fix the seed before each forward pass — call torch.manual_seed(seed) and torch.cuda.manual_seed(seed) before every forward pass. This handles PyTorch-level randomness but won't fix CUDA non-determinism from parallel reductions.

Run multiple seeds and average — run each experiment 3-5 times with different train/test splits, report mean and standard deviation. This doesn't eliminate variance but tells you whether a result is real or noise. It's the standard approach in ML benchmarking.
