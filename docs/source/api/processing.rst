sdm.processing
==============

Processor API
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~sdm.processing.base.Processor
   ~sdm.processing.base.InvertibleMixin
   ~sdm.processing.recipe.Recipe

Ensemble Execution API
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~sdm.processing.ensemble.EnsembleProcessor
   ~sdm.processing.ensemble.VariableSchemaBatchMixin
   ~sdm.processing.ensemble.EnsembleFitContext
   ~sdm.processing.ensemble_table.EnsembleTable
   ~sdm.processing.ensemble_table.EnsembleRelatedTables

Common Processors
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.processing.common") %}
   ~sdm.processing.common.{{ name }}
{% endfor %}

Numerical Processors
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.processing.numerical") %}
   ~sdm.processing.numerical.{{ name }}
{% endfor %}

Categorical Processors
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.processing.categorical") %}
   ~sdm.processing.categorical.{{ name }}
{% endfor %}

Datetime Processors
-------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.processing.datetime") %}
   ~sdm.processing.datetime.{{ name }}
{% endfor %}

Post-Processors
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.processing.output") %}
   ~sdm.processing.output.{{ name }}
{% endfor %}
