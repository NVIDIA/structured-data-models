{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
{%- if objname != "ICLModel" %}
   :exclude-members: supported_execution_modes, supported_feature_stypes, supported_target_stypes, supports_related_tables
{%- endif %}

{% if objname != "ICLModel" %}
{% set cls = import_module(module) | attr(objname) %}
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
   * - **Related Table Support**
     - {{ "✅" if cls.supports_related_tables else "❌" }}
   * - **Supported Estimator Execution Modes**
     - {% for mode in cls.supported_execution_modes | sort -%}
         ``{{ mode }}``{{ ", " if not loop.last }}
       {%- endfor %}

Default Recipe
--------------

.. code-block:: python

{{ cls.default_recipe() | string | indent(3, true) }}
{% endif %}
