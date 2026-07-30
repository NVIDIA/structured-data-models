# TabICLv2 on TabArena

This example evaluates `sdm.models.TabICLv2` on [TabArena](https://tabarena.ai).

## Setup

Install the source revisions of AutoGluon and TabArena used by this example:

```bash
pip install \
  "autogluon.common @ git+https://github.com/autogluon/autogluon.git@0e2db0c68f4f54ba9c2c418721c8dba92a34df72#subdirectory=common" \
  "autogluon.core @ git+https://github.com/autogluon/autogluon.git@0e2db0c68f4f54ba9c2c418721c8dba92a34df72#subdirectory=core" \
  "autogluon.features @ git+https://github.com/autogluon/autogluon.git@0e2db0c68f4f54ba9c2c418721c8dba92a34df72#subdirectory=features" \
  "autogluon.tabular @ git+https://github.com/autogluon/autogluon.git@0e2db0c68f4f54ba9c2c418721c8dba92a34df72#subdirectory=tabular" \
  "tabarena[plot] @ git+https://github.com/autogluon/tabarena.git@7fee3bef1670be0bd52b4ecb99f7761e97b06068#subdirectory=packages/tabarena"
```

## Run

Run the benchmark:

```bash
python examples/tabiclv2/tabarena/main.py
```

Evaluate the results with:

```bash
python examples/tabiclv2/tabarena/evaluate.py
```
