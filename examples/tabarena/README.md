# SDM tabular models on TabArena

This example benchmarks tabular models from `sdm.models` on [TabArena](https://tabarena.ai).

## Setup

Install the source revisions of AutoGluon and TabArena used by this example:

```bash
pip install structured-data-models \
  "autogluon.common @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=common" \
  "autogluon.core @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=core" \
  "autogluon.features @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=features" \
  "autogluon.tabular @ git+https://github.com/autogluon/autogluon.git@61764c3921250b2bff1c94e1217b5f3f089a25ac#subdirectory=tabular" \
  "bencheval @ git+https://github.com/autogluon/tabarena.git@f64c3742f2cb1b734ecbfa6b429cba76afec2c73#subdirectory=packages/bencheval" \
  "tabarena[plot] @ git+https://github.com/autogluon/tabarena.git@f64c3742f2cb1b734ecbfa6b429cba76afec2c73#subdirectory=packages/tabarena"
```

## Run

Run the benchmark:

```bash
python main.py --model tabiclv2
```

Pass a dataset name to run only that TabArena dataset:

```bash
python main.py \
  --model tabiclv2 \
  --dataset blood-transfusion-service-center
```

Evaluate all available model results with:

```bash
python evaluate.py
```
