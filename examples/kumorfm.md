# KumoRFM example

`kumorfm.py` runs KumoRFM over a full RelBench test split. It uses cached
`fit/predict` by default, also supports direct `forward` or both interfaces,
and reports every metric defined by the selected RelBench task.

Install RelBench 2.1.2 or later and a `pyg-lib` wheel compatible with the
installed PyTorch build. The sampler's import error includes the matching
`pyg-lib` installation command.

```bash
python -m examples.kumorfm \
  --dataset rel-f1 \
  --task driver-dnf \
  --interface both
```

The benchmark supports RelBench entity tasks for regression, binary
classification, and multiclass classification with at most 10 classes.
Recommendation, link-prediction, multilabel, and larger multiclass tasks need
model interfaces or capabilities that are outside this benchmark. The
`rel-mimic` dataset also requires its documented access credentials.
