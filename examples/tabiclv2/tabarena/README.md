# TabICLv2 on TabArena

This example evaluates the repository-local `TabICLv2` implementation on TabArena.

## Setup

From the repository root, create a Python 3.11–3.13 environment and install SDM:

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -e .
```

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

Completed jobs in the output directory are reused. Evaluate the results with:

```bash
python examples/tabiclv2/tabarena/evaluate.py
```
