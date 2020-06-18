{% set escapedname = objname|escape %}
{% set title = "*" ~ objtype ~ "* " ~ escapedname %}
{{ title | underline }}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
