{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
{%- if objname == "ICLModel" %}
   :inherited-members: Module
   :member-order: groupwise
{%- endif %}
