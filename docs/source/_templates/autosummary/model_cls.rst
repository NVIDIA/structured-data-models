{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:

{% if "default_recipe" in members %}
{% set cls = import_module(module) | attr(objname) %}
{% set recipe = cls.default_recipe() %}
{% if recipe is not none %}
Default Recipe
--------------

.. code-block:: python

{{ recipe | string | indent(3, true) }}
{% endif %}
{% endif %}
