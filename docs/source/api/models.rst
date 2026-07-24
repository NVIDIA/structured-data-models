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
   * - :class:`~sdm.models.KumoRFM` (`Paper <https://arxiv.org/abs/2604.12596>`__)
     - 2026-04-14
     - | 29.93M (classification)
       | 30.94M (regression)
     - `BSD-3-Clause <https://github.com/soda-inria/tabicl/blob/main/LICENSE>`__
     - `BSD-3-Clause <https://huggingface.co/jingang/TabICL>`__

Model API
---------

.. autosummary::
   :toctree: generated
   :template: model_cls
   :nosignatures:

{% for name in api_names("sdm.models") %}
   ~sdm.models.{{ name }}
{% endfor %}
