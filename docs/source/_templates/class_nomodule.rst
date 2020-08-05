{% set escapedname = objname|escape %}
{% set title = "*" ~ objtype ~ "* " ~ escapedname %}
{{ title | underline }}

.. currentmodule:: {{ module }}

.. auto{{objtype}}:: {{ objname }}

   {% block methods %}
   {% if objtype == "class" %}
   .. automethod:: __init__
   {% endif %}

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
