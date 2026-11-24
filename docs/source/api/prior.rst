sdm.prior
=========

.. autosummary::
   :toctree: generated
   :nosignatures:

{% set components = (
    api_names("sdm.prior.root_samplers")
    + api_names("sdm.prior.activations")
    + api_names("sdm.prior.functions")
    + api_names("sdm.prior.matrix_samplers")
    + api_names("sdm.prior.converters")
) %}
{% for name in api_names("sdm.prior") if name not in components %}
   ~sdm.prior.{{ name }}
{% endfor %}

Root Samplers
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.root_samplers") %}
   ~sdm.prior.{{ name }}
{% endfor %}

Activations
-----------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.activations") %}
   ~sdm.prior.{{ name }}
{% endfor %}

Functions
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.functions") %}
   ~sdm.prior.{{ name }}
{% endfor %}

Matrix Samplers
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.matrix_samplers") %}
   ~sdm.prior.{{ name }}
{% endfor %}

Converters
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.converters") %}
   ~sdm.prior.{{ name }}
{% endfor %}

Tabular Priors
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.tabular") %}
   ~sdm.prior.tabular.{{ name }}
{% endfor %}

Tabular Transforms
------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.tabular.transform") %}
   ~sdm.prior.tabular.transform.{{ name }}
{% endfor %}
