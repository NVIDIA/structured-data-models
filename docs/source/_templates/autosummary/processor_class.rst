{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :exclude-members: operates_on_stypes, unoperated_stype_policy, requires_fit

{% set cls = import_module(module) | attr(objname) %}
Capabilities
------------

.. list-table::

   * - **Operates On Semantic Types**
     - {% if cls.operates_on_stypes is defined and cls.operates_on_stypes is iterable -%}
         {% for stype in cls.operates_on_stypes | sort(attribute="value") -%}
           ``{{ stype }}``{{ ", " if not loop.last }}
         {%- endfor %}
       {%- else -%}
         instance-specific
       {%- endif %}
   * - **Unoperated Semantic Type Policy**
     - {% if cls.unoperated_stype_policy is defined -%}
         ``{{ cls.unoperated_stype_policy }}``
       {%- else -%}
         instance-specific
       {%- endif %}
   * - **Requires Fitting**
     - {% if cls.requires_fit is not defined -%}
         instance-specific
       {%- elif cls.requires_fit -%}
         ✅
      {%- else -%}
         ❌
      {%- endif %}
