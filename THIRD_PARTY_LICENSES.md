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

Licensing materials for the planned Kumo-Forecast and Kumo-Anomaly integrations are retained here ahead of the model implementations.
These notices do not add model code, pretrained weights, or runtime dependencies to SDM.
Kumo-TS is an NVIDIA project; the third-party references below concern its component provenance, not ownership of the Kumo models.

- Source reference: [NVIDIA/Kumo-TS at `af74798bbf7d020d3298f545fb9be383560a18ce`](https://github.com/NVIDIA/Kumo-TS/tree/af74798bbf7d020d3298f545fb9be383560a18ce)
- NVIDIA-authored source license: Apache License 2.0
- Upstream license and redistribution notices: [`sdm/models/kumo/timeseries/LICENSE`](sdm/models/kumo/timeseries/LICENSE) and [`sdm/models/kumo/timeseries/NOTICE`](sdm/models/kumo/timeseries/NOTICE)

The retained Kumo-TS license and notice describe the upstream distribution, including its DPM-Solver component.
They do not imply that every upstream component is bundled with SDM.
The following pinned references identify the source and license texts inspected for the integrations.

### Forecasting components

- MOMENT patch embeddings and normalization: [source at `38f7310ad594100747ca2a8357e9c7ca7d323e0e`](https://github.com/moment-timeseries-foundation-model/moment/tree/38f7310ad594100747ca2a8357e9c7ca7d323e0e/momentfm/models/layers), MIT License, Copyright (c) 2024 Auton Lab, Carnegie Mellon University. Full terms: [`third_party/moment/LICENSE`](third_party/moment/LICENSE).
- RevIN, the normalization reference used by MOMENT: [source at `fee40bc6c87cb536d048bcf1c14c4ed644b875e1`](https://github.com/ts-kim/RevIN/blob/fee40bc6c87cb536d048bcf1c14c4ed644b875e1/RevIN.py), MIT License, Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI). Full terms: [`third_party/revin/LICENSE`](third_party/revin/LICENSE).
- Transformers T5 encoder reference: [source at `8cb5963cc22174954e7dca2c0a3320b7dc2f4edc`](https://github.com/huggingface/transformers/blob/8cb5963cc22174954e7dca2c0a3320b7dc2f4edc/src/transformers/models/t5/modeling_t5.py), Apache License 2.0. Attribution: [`third_party/transformers/NOTICE`](third_party/transformers/NOTICE); full terms: [`LICENSE`](LICENSE). This notice does not add a Transformers runtime dependency.

### Anomaly components

- CSDI diffusion-step embedding and residual-block reference: [source at `7f24a436f08d98853a6b43d4f7f04e5a65ecdf27`](https://github.com/ermongroup/CSDI/blob/7f24a436f08d98853a6b43d4f7f04e5a65ecdf27/diff_models.py), MIT License, Copyright (c) 2021 Yusuke Tashiro. Full terms: [`third_party/csdi/LICENSE`](third_party/csdi/LICENSE).
- DiffWave, acknowledged by CSDI and providing the diffusion-step embedding reference: [source at `0594106093b8d8c444de8bd8cd26482f653c569f`](https://github.com/lmnt-com/diffwave/blob/0594106093b8d8c444de8bd8cd26482f653c569f/src/diffwave/model.py), Apache License 2.0. Attribution: [`third_party/diffwave/NOTICE`](third_party/diffwave/NOTICE); full terms: [`LICENSE`](LICENSE).
- DPM-Solver, included in upstream Kumo-Anomaly: [vendored source at `af74798bbf7d020d3298f545fb9be383560a18ce`](https://github.com/NVIDIA/Kumo-TS/blob/af74798bbf7d020d3298f545fb9be383560a18ce/Kumo-Anomaly/utils/dpm_solver_pytorch.py), MIT License, Copyright (c) 2022 Cheng Lu. Full terms: [`third_party/dpm-solver/LICENSE`](third_party/dpm-solver/LICENSE). The solver implementation is not bundled with SDM by this license-only change.

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
