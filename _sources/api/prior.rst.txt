sdm.prior
=========

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~sdm.prior.tabular.SCMPrior

Sampling Utilities
------------------

**Hyperparameters**

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.hp") %}
   ~sdm.prior.{{ name }}
{% endfor %}

**Correlated Sampling**

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.correlated") %}
   ~sdm.prior.{{ name }}
{% endfor %}

Schema
------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.tabular.schema") %}
   ~sdm.prior.tabular.{{ name }}
{% endfor %}

Directed Acyclic Graphs
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.dag") %}
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

{% for name in api_names("sdm.prior.activations") if name.endswith("Activation") %}
   ~sdm.prior.{{ name }}
{% endfor %}

**Activation Functions for** :class:`~sdm.prior.CallableActivation`

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.activations") if not name.endswith("Activation") %}
   ~sdm.prior.{{ name }}
{% endfor %}

Random Functions
----------------

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

Filters
-------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~sdm.prior.should_drop

Transforms
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

{% for name in api_names("sdm.prior.tabular.transform") %}
   ~sdm.prior.tabular.transform.{{ name }}
{% endfor %}
