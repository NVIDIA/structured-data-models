{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
{%- if objname == "PriorTransform" %}
   :special-members: __call__
{%- endif %}
{%- if objname == "ICLModel" %}
   :inherited-members: Module
   :member-order: groupwise
{%- endif %}
