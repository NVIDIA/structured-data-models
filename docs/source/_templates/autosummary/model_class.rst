{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :exclude-members: supported_feature_stypes, supported_target_stypes, supported_tasks, supports_multi_target, supports_context_only_columns, supports_related_tables, default_recipe, release_date, parameter_count, code_license, weights_license, metadata

{% set cls = import_module(module) | attr(objname) %}

Model Details
-------------

.. list-table::

   * - **Release Date**
     - {{ cls.release_date }}
   * - **Parameter Count**
     - {{ cls.parameter_count }}
   * - **Code License**
     - `{{ cls.code_license[0] }} <{{ cls.code_license[1] }}>`__
   * - **Weights License**
     - `{{ cls.weights_license[0] }} <{{ cls.weights_license[1] }}>`__
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
{% for name, value in cls.metadata.items() %}
   * - **{{ name }}**
     - {{ value | indent(7) }}
{% endfor %}

Default Recipe
--------------

{% set doc = getdoc(cls.default_recipe) %}
{% if doc and ":meta private:" not in doc %}{{ doc }}{% endif %}

.. code-block:: python

{{ cls.default_recipe() | string | indent(3, true) }}
