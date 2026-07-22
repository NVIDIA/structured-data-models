# KumoRFM RelBench benchmark

`kumorfm.py` runs KumoRFM over complete RelBench test splits. With no dataset
or task filters, it attempts all tasks registered for the standard `rel-*`
datasets except SALT, which is excluded from this generic sweep. The default
sweep includes `rel-mimic` and therefore requires its documented access
credentials. It uses cached `fit/predict` by default, also supports direct
`forward`, and reports every metric defined by each task.

Install RelBench 2.1.2 or later and a `pyg-lib` wheel compatible with the
installed PyTorch build. The sampler's import error includes the matching
`pyg-lib` installation command.

```bash
python -m examples.kumorfm
```

The dataset and task arguments work as follows:

- No filters: run all tasks across the standard RelBench datasets.
- `--dataset`: run all tasks for that dataset.
- `--dataset` and `--task`: run one dataset/task pair.

For example:

```bash
python -m examples.kumorfm --dataset rel-f1

python -m examples.kumorfm \
  --dataset rel-f1 \
  --task driver-dnf \
  --interface forward
```

Run the command once with each interface when comparing `forward` with
`fit/predict`.

Each completed or skipped task in a sweep is printed immediately with its
status and metrics. An explicitly selected unsupported task and unexpected
failures stop the benchmark with their traceback.

This benchmark currently implements RelBench entity tasks for regression,
binary classification, and multiclass classification with at most 10 classes.
It reports recommendation, link-prediction, multilabel, and larger multiclass
tasks as skipped. These are limitations of the benchmark integration, not a
statement about SDM-wide model support.

For each interface, the script reports the average batch model-call runtime
across the complete test split. It times `forward` or `predict` for each test
batch, excluding model setup, `fit`, sampling, data transfer, output
conversion, and metric evaluation. Peak CUDA memory allocated by PyTorch
covers the complete dataset/task/interface run. CUDA runtime uses CUDA events;
CPU runtime uses a wall-clock timer.
