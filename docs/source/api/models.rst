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
{% for name in api_names("sdm.models") if name not in api_names("sdm.models.base") %}
{% set cls = import_module("sdm.models") | attr(name) %}
   * - | :class:`~sdm.models.{{ name }}`
       | |{{ name }}-paper| |{{ name }}-source|
     - {{ cls.release_date }}
     - {{ cls.parameter_count }}
     - `{{ cls.code_license[0] }} <{{ cls.code_license[1] }}>`__
     - `{{ cls.weights_license[0] }} <{{ cls.weights_license[1] }}>`__
{% endfor %}

{% for name in api_names("sdm.models") if name not in api_names("sdm.models.base") %}
{% set cls = import_module("sdm.models") | attr(name) %}
.. |{{ name }}-paper| image:: https://cdn.simpleicons.org/arxiv
   :width: 16px
   :height: 16px
   :alt: Paper for {{ name }}
   :target: {{ cls.paper }}

.. |{{ name }}-source| image:: https://cdn.simpleicons.org/github/808080
   :width: 16px
   :height: 16px
   :alt: Source for {{ name }}
   :target: {{ cls.source }}
{% endfor %}

Model API
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ~sdm.models.Model
   ~sdm.models.ICLModel
   ~sdm.models.Callback

Models
------

.. autosummary::
   :toctree: generated
   :template: model_class
   :nosignatures:

{% for name in api_names("sdm.models") if name not in api_names("sdm.models.base") %}
   ~sdm.models.{{ name }}
{% endfor %}
