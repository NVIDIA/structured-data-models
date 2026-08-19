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

## Experiment 1

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
