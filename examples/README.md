# Examples

This folder contains runnable examples of `structured-data-models`:

- [**`tabular/`**](tabular/): Tabular foundation model examples
- [**`relational/`**](relational/): Relational foundation model examples

See [`tabular/quickstart.py`](tabular/quickstart.py) for a minimal runnable `structured-data-models` example with `TabICLv2`.

## Experimental ensemble-parallel inference

`sdm.models.EnsembleParallel` distributes independent ensemble members across explicitly placed `ICLModel` replicas, including KumoTabular and KumoRelational. Construct distinct evaluation models with identical checkpoint weights, normally one per CUDA device. Keep their weights and placement unchanged while the executor is fitted. Use a context manager to release fitted state and worker threads:

```python
import sdm

# replicas: identical evaluation models already placed on separate devices.
# x_train/y_train: context only; x_val: held-out queries without targets.
with sdm.models.EnsembleParallel(replicas) as ensemble:
    ensemble.fit(x_train, y_train, num_estimators=8, member_seed=42)
    predictions = ensemble.predict(x_val)
```

The wrapper fits one shared recipe on the context, assigns logical member `i` to replica `i % len(replicas)`, retains each fitted member cache there, and combines outputs in logical member order using the recipe's output processors. For relational input, pass `related_tables` to both `fit` and `predict`, with the appropriate context/query graph and temporal boundary. Query targets must never enter either prediction input or context preprocessing.

`generator` controls preprocessing; `member_seed + i` independently controls each model member's fit-time randomness. This is intentionally not the native model's shared RNG stream: use a **one-replica `EnsembleParallel`** with the same recipe, input order, generator seed and member seed as the matched serial baseline. A native model is a useful additional reference, but stochastic equivalence must be checked, not assumed.

Preprocessing state and output reduction remain on the input device; model weights are replicated and only member caches are distributed. This is not context or model sharding, and it does not make an individually oversized member fit. `cache_bytes` reports logical tensor-cache bytes per replica, not allocator peak, retained storage, preprocessing state or total GPU memory. Calls are synchronous and must not overlap on one executor; call `clear()` to release fitted caches before reusing the executor. The current API has no callbacks, gradients, process workers or distributed runtime requirement.
