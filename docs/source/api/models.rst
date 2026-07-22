sdm.models
==========

.. list-table::
   :header-rows: 1

   * - Model
     - Release Date
     - Parameter Count
     - Code License
     - Weights License
   * - `TabICLv2 <https://arxiv.org/abs/2602.11139>`__
     - 2026-02-12
     - | 27.55M (classification)
       | 28.54M (regression)
     - `BSD-3-Clause <https://github.com/soda-inria/tabicl/blob/main/LICENSE>`__
     - `BSD-3-Clause <https://huggingface.co/jingang/TabICL>`__
   * - `KumoRFM <https://arxiv.org/abs/2604.12596>`__
     - 2026-04-14
     - | 29.93M (classification)
       | 30.94M (regression)
     - TBD
     - TBD

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.models") %}
   sdm.models.{{ name }}
{% endfor %}
