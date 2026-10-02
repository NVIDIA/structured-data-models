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
   * - :class:`~sdm.models.KumoTabular` (`Blog <https://huggingface.co/blog/nvidia/kumo-tabular>`__)
     - 2026-09-28
     - | 27.46M (small, classification)
       | 28.47M (small, regression)
       | 61.49M (medium, classification)
       | 62.49M (medium, regression)
       | 213.67M (large, classification)
       | 215.68M (large, regression)
     - `Apache-2.0 <https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE>`__
     - `OpenMDW-1.1 <https://huggingface.co/nvidia/Kumo-Tabular/>`__
   * - :class:`~sdm.models.KumoRelational` (`Paper <https://arxiv.org/abs/2604.12596>`__)
     - 2026-04-14
     - | 29.93M (classification)
       | 30.94M (regression)
     - `Apache-2.0 <https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE>`__
     - `OpenMDW-1.1, with BSD-3-Clause third-party notice <https://huggingface.co/nvidia/Kumo-Relational>`__

Model API
---------

.. autosummary::
   :toctree: generated
   :template: model_class
   :nosignatures:

{% for name in api_names("sdm.models") if name != "ECOC" %}
   ~sdm.models.{{ name }}
{% endfor %}

Kumo Tabular batching
--------------------

:func:`sdm.models.kumo.tabular.estimate_batch_size` estimates capacity from a
Kumo Tabular model, unprocessed context tables, and a memory budget in bytes.
For fitting, the result is the number of ensemble members to run together:

.. code-block:: python

   from sdm.models.kumo.tabular import estimate_batch_size

   batch_size = estimate_batch_size(
       model, x_context, y_context,
       memory_budget=fit_memory_budget,
       num_estimators=16,
   )
   model.fit(
       x_context, y_context,
       num_estimators=16,
       estimator_batch_size=batch_size,
   )

For repeated predictions, use ``mode="predict"`` with the same context and
ensemble settings. Its result is a query row count for splitting the query
table before calling ``model.predict``:

.. code-block:: python

   query_batch_size = estimate_batch_size(
       model, x_context, y_context,
       memory_budget=predict_memory_budget,
       num_estimators=16,
       mode="predict",
       estimator_batch_size=batch_size,
   )
   for query in x_query.split(query_batch_size, dim=0):
       prediction = model.predict(query)

The prediction budget must exclude resident or staged context caches,
including overlapping transfers.

The default execution dtype is FP16; pass ``dtype`` to match your actual
execution or autocast dtype. These default-recipe inference estimates reserve
half the budget and return at least one item, even when it may not fit.

.. autofunction:: sdm.models.kumo.tabular.estimate_batch_size
