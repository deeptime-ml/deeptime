# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
from docutils.parsers.rst import directives
from sphinx.ext.autosummary import Autosummary, get_documenter
from sphinx.util.inspect import safe_getattr

project = 'scikit-time'
copyright = '2020, AI4Science Group'
author = 'AI4Science Group'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Autosummary settings -----------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    'members':  True,
    'member-order': 'groupwise',
    'inherited-members': True
}

# -- Napoleon settings --------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Alabaster theme settings -------------------------------------------------
html_theme_options = {
    'page_width': '80%'
}

def setup(app):
    class AutoAutoSummary(Autosummary):

        option_spec = {
            'methods': directives.unchanged,
            'attributes': directives.unchanged,
            'toctree': directives.unchanged
        }

        required_arguments = 1

        @staticmethod
        def get_members(obj, typ, include_public=None):
            if not include_public:
                include_public = []
            items = []
            for name in dir(obj):
                try:
                    documenter = get_documenter(app, safe_getattr(obj, name), obj)
                except AttributeError:
                    continue
                if documenter.objtype == typ:
                    items.append(name)
            public = [x for x in items if x in include_public or not x.startswith('_')]
            return public, items

        def run(self):
            try:
                clazz = str(self.arguments[0])
                (module_name, class_name) = clazz.rsplit('.', 1)
                m = __import__(module_name, globals(), locals(), [class_name])
                c = getattr(m, class_name)
                default = 'members' not in self.options and 'attributes' not in self.options
                if 'methods' in self.options or default:
                    _, methods = self.get_members(c, 'method', ['__init__'])

                    self.content = ["~%s.%s" % (clazz, method) for method in methods if not method.startswith('_')]
                if 'attributes' in self.options or default:
                    _, attribs = self.get_members(c, 'attribute')
                    self.content = ["~%s.%s" % (clazz, attrib) for attrib in attribs if not attrib.startswith('_')]
            finally:
                return super(AutoAutoSummary, self).run()
    app.add_directive('autoautosummary', AutoAutoSummary)
