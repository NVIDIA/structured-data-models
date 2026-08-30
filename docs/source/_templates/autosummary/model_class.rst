{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
{%- if objname != "ICLModel" %}
   :exclude-members: supported_feature_stypes, supported_target_stypes, supported_tasks, supports_related_tables
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
   * - **Supported Prediction Tasks**
     - {% for task in cls.supported_tasks | sort(attribute="value") -%}
         ``{{ task }}``{{ ", " if not loop.last }}
       {%- endfor %}
   * - **Related Table Support**
     - {{ "✅" if cls.supports_related_tables else "❌" }}

Default Recipe
--------------

.. code-block:: python

{{ cls.default_recipe() | string | indent(3, true) }}
{% endif %}
