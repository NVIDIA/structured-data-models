<p align="center">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/source/images/logo_light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/images/logo_dark.svg">
  <img src="docs/source/images/logo_light.svg" width="125">
</picture>
</p>

<h1 align="center">Structured Data Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-brightgreen.svg?style=flat&color=76B900)](https://www.python.org/downloads)
[![License: Apache 2.0](https://img.shields.io/badge/license-apache%202.0-brightgreen.svg?style=flat&color=76B900)](https://opensource.org/licenses/Apache-2.0)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&color=76B900)](CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat&color=76B900)](https://musical-invention-2y4yjlw.pages.github.io)

</h1>

**A GPU-native library of foundation models, tensor subclasses, and data processors for structured data.**

- **Models:** Reference implementations of structured data foundation models, including tabular models ([`TabICLv2`](https://musical-invention-2y4yjlw.pages.github.io/api/generated/sdm.models.TabICLv2), [`KumoTabular`](https://musical-invention-2y4yjlw.pages.github.io/api/generated/sdm.models.KumoTabular), *etc*), and relational models ([`KumoRelational`](https://musical-invention-2y4yjlw.pages.github.io/api/generated/sdm.models.KumoRelational)), built on a unified interface with room for future model families.
- **Tensor semantics:** PyTorch-compatible tensor types for numerical, categorical, datetime, text, and relational data.
- **Data processing:** Composable, extensible, and GPU-accelerated preprocessing and postprocessing for structured data workflows.

## Installation

The `structured-data-models` package is available from Python 3.11 and PyTorch 2.7 onwards.
Install from the `main` branch:

```bash
pip install git+https://github.com/NVIDIA/structured-data-models.git
```

> [!NOTE]
> For CUDA workloads, we highly recommend installing [`cudf`](https://docs.rapids.ai/install) as an additional dependency to keep dataframe-style operations on GPU and avoid unnecessary data movement.

## Model Families

**Tabular Foundation Models:**

- **[`TabICLv2`](https://musical-invention-2y4yjlw.pages.github.io/api/generated/sdm.models.TabICLv2)** from Qu *et al.*: [TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model](https://arxiv.org/abs/2602.11139) (ICML '26)
- **[`KumoTabular`](https://musical-invention-2y4yjlw.pages.github.io/api/generated/sdm.models.KumoTabular)** from Qu *et al.*: [NVIDIA Kumo Tabular Sets a New Accuracy-Efficiency Frontier for Tabular Prediction](https://huggingface.co/blog/nvidia/kumo-tabular) ('26)
- **[`TabFM`](https://musical-invention-2y4yjlw.pages.github.io/api/generated/sdm.models.TabFM)** from Kong *et al.*: [Introducing TabFM: A Zero-shot Foundation Model for Tabular Data](https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data) ('26)

**Relational Foundation Models:**

- **[`KumoRelational`](https://musical-invention-2y4yjlw.pages.github.io/api/generated/sdm.models.KumoRelational)** from Hudovernik *et al.*: [KumoRFM-2: Scaling Foundation Models for Relational Learning](https://arxiv.org/abs/2604.12596) (CoRR '26)

## Quick Tour

```python
from sklearn.datasets import load_breast_cancer

import sdm

df = load_breast_cancer(as_frame=True).frame

# A lossless, fully tensorized representation of the raw data on GPU:
table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}),
    device="cuda",
)

# Access to a variety of pre-trained structured data models:
model = sdm.models.TabICLv2(device="cuda")

# Default in-context learning forward pass:
model(
    x_context=table[:300].drop_columns("target"),
    y_context=table[:300, "target"],
    x_query=table[300:].drop_columns("target"),
    num_estimators=8,
)

# Fit + Predict forward pass via key/value caching for fast inference:
model.fit(
    x=table[:300].drop_columns("target"),
    y=table[:300, "target"],
    num_estimators=8,
)
model.predict(table[300:].drop_columns("target"))
model.clear()
```

Additional examples are available in [`examples/`](examples).
Benchmarks for reproducing reported results live in [`benchmark/`](benchmark).

## Notice and Disclaimer

This software automatically retrieves, accesses or interacts with external materials.
Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses.
You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case.
This software is provided "AS IS", without warranty of any kind.
The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials.
Use this software and the retrieved materials at your own risk.

## License

The NVIDIA-authored source code is licensed under the [Apache License 2.0](LICENSE).
Third-party software and separately distributed model assets are documented in [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md):

- [**`sdm/models/tabiclv2/`**](sdm/models/tabiclv2/) contains code derived from [`TabICLv2`](https://github.com/soda-inria/tabicl) under the BSD 3-Clause License; its terms are distributed in [`sdm/models/tabiclv2/LICENSE`](sdm/models/tabiclv2/LICENSE).
- [**`sdm/models/tabfm/`**](sdm/models/tabfm/) contains code derived from [`TabFM`](https://github.com/google-research/tabfm) under the Apache License 2.0; its terms are distributed in [`sdm/models/tabfm/LICENSE`](sdm/models/tabfm/LICENSE).
- [**`sdm/models/timesfm3/`**](sdm/models/timesfm3/) contains code derived from [`TimesFM`](https://github.com/google-research/timesfm) under the Apache License 2.0; its terms are distributed in [`sdm/models/timesfm3/LICENSE`](sdm/models/timesfm3/LICENSE).
- [**`sdm/models/kumo/relational/NOTICE`**](sdm/models/kumo/relational/NOTICE) documents [`KumoRelational`](sdm/models/kumo/relational/)'s reuse of [`TabICLv2`](sdm/models/tabiclv2/)-derived components.
- [**`third_party/pytorch/`**](third_party/pytorch/) contains the BSD 3-Clause License for material adapted from [PyTorch](https://github.com/pytorch/pytorch) in [`CONTRIBUTING.md`](CONTRIBUTING.md); its terms are distributed in [`third_party/pytorch/LICENSE`](third_party/pytorch/LICENSE).
- [**`third_party/contributor-covenant/`**](third_party/contributor-covenant/) contains the MIT License for [Contributor Covenant version 1.4](https://www.contributor-covenant.org/version/1/4/code-of-conduct); its terms are distributed in [`third_party/contributor-covenant/LICENSE`](third_party/contributor-covenant/LICENSE).
