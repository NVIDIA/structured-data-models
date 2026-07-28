# Examples

Examples live in the repository's
[`examples`](https://github.com/NVIDIA/structured-data-models/tree/main/examples)
directory so they are versioned with the code they demonstrate. The Colab link
opens the same notebook from GitHub rather than a separate copy.

| Example                                                                                                               | Format        | Purpose                                                                                    |
| --------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| [`examples/kumorfm/`](https://github.com/NVIDIA/structured-data-models/tree/main/examples/kumorfm)                    | Directory     | KumoRFM RelBench example runner.                                                           |
| [`examples/tabiclv2.py`](https://github.com/NVIDIA/structured-data-models/blob/main/examples/tabiclv2.py)             | Python script | Minimal TabICLv2 forward pass and cached prediction flow.                                  |
| [`examples/TabICL_demo.ipynb`](https://github.com/NVIDIA/structured-data-models/blob/main/examples/TabICL_demo.ipynb) | Notebook      | TabICLv2 quickstart covering regression, recipes, caching, ensembling, and classification. |

[![Open TabICL demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/structured-data-models/blob/main/examples/TabICL_demo.ipynb)

The examples index is checked in CI so new top-level examples stay
discoverable. Notebook outputs are stripped before commit to avoid stale
results in documentation diffs.
