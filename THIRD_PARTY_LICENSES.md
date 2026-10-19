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

## DPM-Solver

- Source: https://github.com/LuChengTHU/dpm-solver
- License: MIT
- License terms: [`third_party/dpm-solver/LICENSE`](third_party/dpm-solver/LICENSE)

The [`sdm/models/kumo/timeseries/anomaly`](sdm/models/kumo/timeseries/anomaly) implementation contains code derived from [DPM-Solver](https://github.com/LuChengTHU/dpm-solver).

## TabPFN Extensions

- Source: https://github.com/PriorLabs/tabpfn-extensions
- License: Apache License 2.0
- License terms: [`third_party/tabpfn-extensions/LICENSE`](third_party/tabpfn-extensions/LICENSE)

The [`sdm/models/ecoc.py`](sdm/models/ecoc.py) implementation contains code derived from [`TabPFN Extensions`](https://github.com/PriorLabs/tabpfn-extensions).

## PyTorch Contribution Guide

Portions of [`CONTRIBUTING.md`](CONTRIBUTING.md) are adapted from the [PyTorch contribution guide](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md), distributed under the BSD 3-Clause License. The complete copyright notices and license terms are distributed in [`third_party/pytorch/LICENSE`](third_party/pytorch/LICENSE).

## Contributor Covenant

[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) is adapted from [Contributor Covenant version 1.4](https://www.contributor-covenant.org/version/1/4/code-of-conduct/), distributed under the MIT License. The complete copyright notice and license terms are distributed in [`third_party/contributor-covenant/LICENSE`](third_party/contributor-covenant/LICENSE).

## Timestral

- Source: [mistralai/timestral](https://github.com/mistralai/timestral), revision `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`.
- Source license: [Apache-2.0](sdm/models/timestral/LICENSE).
- Reference checkpoint: [Timestral-5M](https://huggingface.co/mistralai/Timestral-5M/tree/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f), under its [Modified MIT license](https://huggingface.co/mistralai/Timestral-5M/blob/bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f/LICENSE).

The `sdm/models/timestral/` transformer block composes shared SDM attention, normalization, and feedforward modules. Optional parity tests use cached 5M weights; weights are not bundled. This component does not provide an end-to-end forecasting model.
