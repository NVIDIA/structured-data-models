sdm.models
==========

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 15 13 15 13 20

   * - Model
     - Release Date
     - Parameter Count
     - Code License
     - Weights License
   * - | :class:`~sdm.models.TabICLv2`
       | (`Paper <https://arxiv.org/abs/2602.11139>`__)
     - 2026-02-12
     - 27.55M–28.54M
     - `BSD-3-Clause <https://github.com/soda-inria/tabicl/blob/main/LICENSE>`__
     - `BSD-3-Clause <https://huggingface.co/jingang/TabICL>`__
   * - | :class:`~sdm.models.TabFM`
       | (`Paper <https://arxiv.org/abs/2609.37959>`__)
     - 2026-06-30
     - 1.64B–1.65B
     - `Apache-2.0 <https://github.com/google-research/tabfm/blob/b8a8b090c66d1b9e7af278003461582219996b6a/LICENSE>`__
     - `TabFM Non-Commercial v1.0 <https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83/LICENSE>`__
   * - | :class:`~sdm.models.KumoTabular`
       | (`Blog <https://huggingface.co/blog/nvidia/kumo-tabular>`__)
     - 2026-09-28
     - 27.46M–215.68M
     - `Apache-2.0 <https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE>`__
     - `OpenMDW-1.1 <https://huggingface.co/nvidia/Kumo-Tabular/>`__
   * - | :class:`~sdm.models.KumoRelational`
       | (`Paper <https://arxiv.org/abs/2604.12596>`__)
     - 2026-04-14
     - 29.93M–30.94M
     - `Apache-2.0 <https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE>`__
     - `OpenMDW-1.1, with BSD-3-Clause notice <https://huggingface.co/nvidia/Kumo-Relational>`__
   * - | :class:`~sdm.models.KumoForecast`
       | (`Blog <https://developer.nvidia.com/blog/new-nvidia-nv-tesseract-time-series-models-advance-dataset-processing-and-anomaly-detection>`__)
     - 2025-05-06
     - 313.06M–317.26M
     - `Apache-2.0 <https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE>`__
     - `Apache-2.0 <https://huggingface.co/nvidia/Kumo-Forecast>`__
   * - | :class:`~sdm.models.TimesFM3`
       | (`Blog <https://arxiv.org/abs/2609.37959>`__)
     - 2026-08-31
     - 330.71M
     - `Apache-2.0 <https://github.com/google-research/timesfm/blob/master/LICENSE>`__
     - `TimesFM Non-Commercial v1.0 <https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE>`__

   * - | :class:`~sdm.models.Timestral`
       | (`Source <https://github.com/mistralai/timestral>`__)
     - —
     - 5M–209M
     - `Apache-2.0 <https://github.com/mistralai/timestral/blob/main/LICENSE>`__
     - `Modified MIT <https://huggingface.co/mistralai/Timestral-5M/blob/main/LICENSE>`__

Model API
---------

.. autosummary::
   :toctree: generated
   :template: model_class
   :nosignatures:

{% for name in api_names("sdm.models") if name != "ECOC" %}
   ~sdm.models.{{ name }}
{% endfor %}
