# KumoRelational on rel-amazon from Parquet

This example predicts the RelBench `rel-amazon` `user-churn` task without loading the full database into pandas. Install SDM with its test dependencies and `relbench`, then run from the repository root:

```bash
python examples/relational/amazon_parquet.py
```

RelBench downloads the dataset to its cache on first use. Use `--context-size`, `--query-size`, and `--num-neighbors` to change the sampled workload. The default uses 512 training and validation task rows as context, 64 test task rows as queries, and two hops with eight neighbors each. The script removes target labels from queries before sampling.

`ParquetRelationalSampler` builds its graph from relationship keys and timestamps in the Parquet tables, using the same CPU neighbor sampler as `RelationalData.sampler()`. Polars reads the selected feature rows after sampling. The resulting `RelatedTables` go directly to `KumoRelational.fit()` and `predict()`; no full `RelationalData` is created from the database.

The keys, timestamps, graph index, and categorical dictionaries still occupy RAM. Feature rows are read from Parquet for each sample. This approach is useful when the full feature tables exceed memory but the graph metadata fits. The example reports accuracy for one small query batch and is not a full benchmark evaluation.

On a 15 GiB RAM machine with an NVIDIA L4, the default run completed against 23,218,245 source rows (6.82 GiB compressed Parquet). It sampled 5,018 related context rows and 1,019 related query rows, with 4.95 GiB peak process RAM and 0.7344 query accuracy. On the same machine, a direct attempt to load the full pandas database through `task.get_db()` was killed before sampling began.

The focused sampler test in `test/relational/test_parquet_sampler.py` compares sampled task rows, relationship links, and related table contents with the in-memory sampler, including categorical values and temporal sampling.
