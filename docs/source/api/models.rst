sdm.models
==========

Overview
--------

.. list-table::
   :header-rows: 1

   * - Model
     - Release Date
     - Parameter Count
     - Code License
     - Weights License
   * - :class:`~sdm.models.TabICLv2` (`Paper <https://arxiv.org/abs/2602.11139>`__)
     - 2026-02-12
     - | 27.55M (classification)
       | 28.54M (regression)
     - `BSD-3-Clause <https://github.com/soda-inria/tabicl/blob/main/LICENSE>`__
     - `BSD-3-Clause <https://huggingface.co/jingang/TabICL>`__
   * - :class:`~sdm.models.TabFM` (`Blog <https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data>`__)
     - 2026-06-30
     - | 1.64B (classification)
       | 1.65B (regression)
     - `Apache-2.0 <https://github.com/google-research/tabfm/blob/b8a8b090c66d1b9e7af278003461582219996b6a/LICENSE>`__
     - `tabfm-non-commercial-v1.0 <https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83/LICENSE>`__
   * - :class:`~sdm.models.NemotronRelational` (`Paper <https://arxiv.org/abs/2604.12596>`__)
     - 2026-04-14
     - | 29.93M (classification)
       | 30.94M (regression)
     - `Apache-2.0 <https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE>`__
     -

Model API
---------

.. autosummary::
   :toctree: generated
   :template: model_class
   :nosignatures:

{% for name in api_names("sdm.models") %}
   ~sdm.models.{{ name }}
{% endfor %}
