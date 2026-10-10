# Third-Party Software

This project contains code or documentation derived from the following third-party projects.
Runtime, test, and documentation dependencies are declared in [`pyproject.toml`](pyproject.toml)/[`uv.lock`](uv.lock), and example/benchmark dependencies are resolved separately and are not bundled with this repository.

## TabICLv2

- Source: https://github.com/soda-inria/tabicl
- License: BSD 3-Clause
- License terms: [`sdm/models/tabiclv2/LICENSE`](sdm/models/tabiclv2/LICENSE)
- Optional pretrained weights: https://huggingface.co/jingang/TabICL

The [`sdm/models/tabiclv2/`](sdm/models/tabiclv2/) and [`sdm/models/kumo/relational`](sdm/models/kumo/relational) implementations contain code derived from [`TabICLv2`](https://github.com/soda-inria/tabicl).

## TabFM

- Source: https://github.com/google-research/tabfm
- Code license: Apache License 2.0
- License terms: [`sdm/models/tabfm/LICENSE`](sdm/models/tabfm/LICENSE)
- Optional pretrained weights: https://huggingface.co/google/tabfm-1.0.0-pytorch
- Weights license: [TabFM Non-Commercial License v1.0](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE)

The [`sdm/models/tabfm/`](sdm/models/tabfm/) implementation contains code derived from [`TabFM`](https://github.com/google-research/tabfm).
Its weights are not bundled with this project and may be downloaded only after the user accepts their separate license.

## TimesFM 3.0

- Source: https://github.com/google-research/timesfm
- Code license: Apache License 2.0
- License terms: [`sdm/models/timesfm3/LICENSE`](sdm/models/timesfm3/LICENSE)
- Optional pretrained weights: https://huggingface.co/google/timesfm-3.0-pytorch
- Weights license: [TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)

The [`sdm/models/timesfm3/`](sdm/models/timesfm3/) implementation contains code derived from [`TimesFM`](https://github.com/google-research/timesfm).
Its weights are not bundled with this project and may be downloaded only after the user accepts their separate license.

## Kumo Time-Series

The Kumo-Forecast and Kumo-Anomaly implementations in [`sdm/models/kumo/timeseries/`](sdm/models/kumo/timeseries/) are based on NVIDIA's Kumo-TS project.

- Source reference: [NVIDIA/Kumo-TS at `af74798bbf7d020d3298f545fb9be383560a18ce`](https://github.com/NVIDIA/Kumo-TS/tree/af74798bbf7d020d3298f545fb9be383560a18ce)
- NVIDIA-authored source license: Apache License 2.0, provided in the repository root [`LICENSE`](LICENSE).
- Redistribution notices: [`sdm/models/kumo/timeseries/NOTICE`](sdm/models/kumo/timeseries/NOTICE).
- DPM-Solver: MIT License, Copyright (c) 2022 Cheng Lu. [Upstream license in Kumo-TS](https://github.com/NVIDIA/Kumo-TS/blob/af74798bbf7d020d3298f545fb9be383560a18ce/third_party/dpm-solver/LICENSE); full terms: [`third_party/dpm-solver/LICENSE`](third_party/dpm-solver/LICENSE).

### Pretrained weights

Source-code notices do not replace the terms attached to pretrained weights.
The following model cards identify Apache License 2.0 as the governing terms and also describe the models as intended for research and development:

- Kumo-Forecast: [model card at `abff20a58834638b28227ff4ab934f26206e4b09`](https://huggingface.co/nvidia/Kumo-Forecast/blob/abff20a58834638b28227ff4ab934f26206e4b09/README.md).
- Kumo-Anomaly: [model card at `226c5003b0582adc1cac433123f47324bfae39f2`](https://huggingface.co/nvidia/Kumo-Anomaly/blob/226c5003b0582adc1cac433123f47324bfae39f2/README.md).

Pretrained weights are not bundled with this project.
Refer to the corresponding model card and license when downloading or using them.

## TabPFN Extensions

- Source: https://github.com/PriorLabs/tabpfn-extensions
- License: Apache License 2.0
- License terms: [`third_party/tabpfn-extensions/LICENSE`](third_party/tabpfn-extensions/LICENSE)

The [`sdm/models/ecoc.py`](sdm/models/ecoc.py) implementation contains code derived from [`TabPFN Extensions`](https://github.com/PriorLabs/tabpfn-extensions).

## PyTorch Contribution Guide

Portions of [`CONTRIBUTING.md`](CONTRIBUTING.md) are adapted from the [PyTorch contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md), distributed under the BSD 3-Clause License. The complete copyright notices and license terms are distributed in [`third_party/pytorch/LICENSE`](third_party/pytorch/LICENSE).

## Contributor Covenant

[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) is adapted from [Contributor Covenant version 1.4](https://www.contributor-covenant.org/version/1/4/code-of-conduct/), distributed under the MIT License. The complete copyright notice and license terms are distributed in [`third_party/contributor-covenant/LICENSE`](third_party/contributor-covenant/LICENSE).
