sdm.processing
==============

Processor API
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~sdm.processing.base.Processor
   ~sdm.processing.base.InvertibleMixin
   ~sdm.processing.ensemble.EnsembleProcessor
   ~sdm.processing.ensemble.EnsembleInvertibleMixin
   ~sdm.processing.ensemble.EnsembleProcessorAdapter
   ~sdm.processing.recipe.Recipe

Common Processors
-----------------

.. autosummary::
   :toctree: generated
   :template: processor_class
   :nosignatures:

{% for name in api_names("sdm.processing.common") %}
   ~sdm.processing.common.{{ name }}
{% endfor %}

Numerical Processors
--------------------

.. autosummary::
   :toctree: generated
   :template: processor_class
   :nosignatures:

{% for name in api_names("sdm.processing.numerical") %}
   ~sdm.processing.numerical.{{ name }}
{% endfor %}

Categorical Processors
----------------------

.. autosummary::
   :toctree: generated
   :template: processor_class
   :nosignatures:

{% for name in api_names("sdm.processing.categorical") %}
   ~sdm.processing.categorical.{{ name }}
{% endfor %}

Text Processors
---------------

.. autosummary::
   :toctree: generated
   :template: processor_class
   :nosignatures:

{% for name in api_names("sdm.processing.text") %}
   ~sdm.processing.text.{{ name }}
{% endfor %}

Datetime Processors
-------------------

.. autosummary::
   :toctree: generated
   :template: processor_class
   :nosignatures:

{% for name in api_names("sdm.processing.datetime") %}
   ~sdm.processing.datetime.{{ name }}
{% endfor %}

Post-Processors
---------------

.. autosummary::
   :toctree: generated
   :template: processor_class
   :nosignatures:

{% for name in api_names("sdm.processing.output") %}
   ~sdm.processing.output.{{ name }}
{% endfor %}
