# CPU prediction compilation comparison

This experiment compares three prediction entry points on the frozen preprocessing integration `313bc37ee`: eager `model.predict`, eager `model.predict` with each internal module compiled, and `torch.compile(model.predict)`. It does not modify production code.

One eager fit supplies all three arms. The fitted model is deep-copied for the compiled arms, preserving the same weights, processor state, and attention caches. Every arm uses FP32 on CPU, the actual Inductor backend, one PyTorch thread, `dynamic=True`, and `fullgraph=False`. The context contains 32 training rows; these measurements are not representative of large training contexts or GPU inference.

Data is the existing breast-cancer classification and raw-target diabetes regression validation split, with pretrained Kumo-Tabular small checkpoints. The query sizes are 32/128 for classification and 32/111 for regression, because the regression split has only 111 validation rows. Data and checkpoints are external; each result records the input data checksum.

The first invocation for each shape is timed separately from warmed execution, including compilation and guard construction. The existing Inductor disk cache is not cleared; these are first-call costs, not clean-cache compiler benchmarks. A return to the original shape checks reuse. Then each shape receives two additional warmups followed by nine repetitions, rotating the order of the three arms. The recorded time includes the complete public prediction call and output extraction. Timing excludes input construction and parity checks.

Parity uses the existing `atol=1e-5, rtol=1e-4` for the same dtype. Every sample records maximum absolute error, parity, and graph count changes. Small FP32 numerical differences from fusion are expected; these tests do not assert bitwise equality or change the tolerance.

This is a shared macOS host. Raw samples and minimum/maximum ranges are retained; noisy timing differences must not be presented as guaranteed speedups. No GPU, memory, or fullgraph=True performance conclusion follows from this experiment.

## Reproduction

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 TORCHINDUCTOR_CPP_CACHE_PRECOMPILE_HEADERS=0 \
  python experiments/predict_compile_cpu/benchmark.py \
  --data /path/to/data_train_validation.npz \
  --checkpoint /path/to/small/classifier.pt \
  --task classification --estimators 1 \
  --output /tmp/classification-1.json
```

Use the raw diabetes split, `regressor.pt`, and `--task regression` for regression. Repeat with `--estimators 4`. The PCH environment setting works around the local macOS PyTorch 2.14 compiler setup.

## Results

Warm median milliseconds; nine alternating-order samples per cell. Each row uses the same fitted state and dtype. Regression timings describe completed computation, not passing per-value parity.

| Task | Estimators | Query rows | Eager | Inner compiled | Predict compiled | Per-value parity |
|---|---:|---:|---:|---:|---:|---|
| classification | 1 | 32 | 17.52 | 15.90 | 14.77 | Pass |
| classification | 1 | 128 | 57.33 | 56.45 | 55.18 | Pass |
| classification | 4 | 32 | 68.55 | 63.01 | 58.54 | Pass |
| classification | 4 | 128 | 221.11 | 218.56 | 213.68 | Pass |
| regression | 1 | 32 | 11.91 | 10.86 | 8.63 | Pass |
| regression | 1 | 111 | 27.32 | 26.16 | 23.77 | Fail |
| regression | 4 | 32 | 44.41 | 39.10 | 33.03 | Fail |
| regression | 4 | 111 | 105.42 | 101.12 | 93.93 | Fail |

Classification passes the unchanged tolerance in all measured cases. Whole-predict compilation adds about 7% lower latency than inner-only compilation at 32 query rows and about 2% at 128 rows on this CPU. The larger-query difference is small relative to host noise and must not be generalized.

Regression does not pass per-value tolerance for every quantile. With one estimator, 1 of 110,889 values fails at 111 rows; with four estimators, 1 of 31,968 values fails at 32 rows and 3 of 110,889 fail at 111 rows. Both compiled entry points produce bitwise-identical regression outputs: broader prediction compilation adds no extra drift relative to compiling only the internal model. Every failing coordinate, eager/compiled value, and tolerance threshold is included in the final regression JSON files.

For example, one-estimator quantile q004 at query row 94 changes from 0.16009521484375 to 0.160125732421875. Its absolute difference is 0.000030517578125, exceeding its allowed 0.0000260095202975. This is a strict tolerance failure even though the aggregate metric effect is very small.

Using q500 as a median prediction on all 111 regression validation rows, one-estimator RMSE changes from 65.4748077 to 65.4748001; MAE is unchanged at 54.7854195. Four-estimator RMSE changes from 64.8680573 to 64.8680649 and MAE from 53.9904861 to 53.9904900. These are same-dtype comparisons; no tolerance was relaxed.

Each entry point captures one graph for the first shape and another for the larger shape; returning to 32 rows adds no graph. There are no observed graph breaks or warmed-run recompiles in these numerical cases. The saved recompile log reports a tensor-size guard failure at the query-size transition even with dynamic=True. These two sizes do not establish arbitrary-shape support. Initial query rows equal context rows, so the cause of specialization is not isolated here.

First-call costs remain substantial and include compiler-cache effects. Exact per-call times are in the JSON cold arrays. The initial regression1 timing run was unusually noisy and is retained as regression-1-initial-noisy.json; regression-1-retiming.json and regression-4-initial.json retain intermediate repeats. The table uses the final one-estimator regression run and the initial four-estimator run, which had narrower timing ranges. The detailed four-estimator rerun confirms identical numerical findings but had visibly larger host-load variability, so it is retained for parity evidence rather than used to estimate speed.
