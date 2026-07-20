# TabICLv2 on TabArena

This optional example evaluates the repository-local `TabICLv2` model through
TabArena and AutoGluon. Install those packages in the active environment before
running it. It has no Ray dependency and runs every selected job in one local
process.

Run a small outer-evaluation smoke test from the repository root:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-local-smoke \
  --outer \
  --num-estimators 1 \
  --num-cpus 1 \
  --num-gpus 1 \
  --datasets blood-transfusion-service-center anneal QSAR_fish_toxicity
```

Omit `--datasets` and `--subset` to run the complete TabArena suite locally:

```bash
python -m examples.tabiclv2_tabarena.run_local \
  --output-root outputs/tabiclv2-local-full \
  --num-estimators 1 \
  --num-cpus 1 \
  --num-gpus 1
```

Use `--num-gpus 0` for CPU-only execution. Each run requires a fresh output
directory and writes completed SDM result records to
`report/results_per_split.csv`.
