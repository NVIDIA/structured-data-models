{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :exclude-members: supported_stypes, requires_fit

{% set cls = import_module(module) | attr(objname) %}
Capabilities
------------

.. list-table::

   * - **Supported Semantic Types**
     - {% if cls.supported_stypes is defined and cls.supported_stypes is iterable -%}
         {% for stype in cls.supported_stypes | sort(attribute="value") -%}
           ``{{ stype }}``{{ ", " if not loop.last }}
         {%- endfor %}
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
