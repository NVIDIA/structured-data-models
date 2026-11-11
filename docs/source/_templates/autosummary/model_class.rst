{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

{% set cls = import_module(module) | attr(objname) %}

.. list-table::
   :header-rows: 1

   * - Release Date
     - Parameter Count
     - Code License
     - Weights License
   * - {{ cls.release_date }}
     - {{ cls.parameter_count }}
     - `{{ cls.code_license[0] }} <{{ cls.code_license[1] }}>`__
     - `{{ cls.weights_license[0] }} <{{ cls.weights_license[1] }}>`__

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :exclude-members: supported_feature_stypes, supported_target_stypes, supported_tasks, supports_multi_target, supports_context_only_columns, supports_related_tables, default_recipe, paper, source, release_date, parameter_count, code_license, weights_license, metadata

Capabilities
------------

.. list-table::

   * - **Supported Input Semantic Types**
     - {% for stype in cls.supported_feature_stypes | sort(attribute="value") -%}
         ``{{ stype }}``{{ ", " if not loop.last }}
       {%- endfor %}
   * - **Supported Target Semantic Types**
     - {% for stype in cls.supported_target_stypes | sort(attribute="value") -%}
         ``{{ stype }}``{{ ", " if not loop.last }}
       {%- endfor %}
   * - **Supported Prediction Tasks**
     - {% for task in cls.supported_tasks | sort(attribute="value") -%}
         ``{{ task }}``{{ ", " if not loop.last }}
       {%- endfor %}
   * - **Multi-Target Support**
     - {{ "✅" if cls.supports_multi_target else "❌" }}
   * - **Context-only Column Support**
     - {{ "✅" if cls.supports_context_only_columns else "❌" }}
   * - **Related Table Support**
     - {{ "✅" if cls.supports_related_tables else "❌" }}

{% if cls.metadata %}
Model Details
-------------

.. list-table::

{% for name, value in cls.metadata.items() %}
   * - **{{ name }}**
     - {{ value | indent(7) }}
{% endfor %}
{% endif %}

Default Recipe
--------------

{% set doc = getdoc(cls.default_recipe) %}
{% if doc and ":meta private:" not in doc %}{{ doc }}{% endif %}

.. code-block:: python

{{ cls.default_recipe() | string | indent(3, true) }}
