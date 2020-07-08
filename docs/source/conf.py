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
import logging
from logging import LogRecord

import sphinx.util
import sphinxcontrib.bibtex
from docutils.parsers.rst import directives
from sphinx.ext.autosummary import Autosummary, get_documenter
from sphinx.util.inspect import safe_getattr

project = 'scikit-time'
copyright = '2020, AI4Science Group'
author = 'AI4Science Group'

# The full version, including alpha/beta/rc tags
release = '0.1'

# -- Disable certain warnings ------------------------------------------------

sphinxlog_adapter = sphinx.util.logging.getLogger(sphinxcontrib.bibtex.__name__)


class DuplicateLabelForKeysFilter(logging.Filter):

    def filter(self, record: LogRecord) -> int:
        return not (record.msg.startswith("duplicate label for keys") and record.levelno == logging.WARN)


sphinxlog_adapter.logger.addFilter(DuplicateLabelForKeysFilter())

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinxcontrib.bibtex',
    'matplotlib.sphinxext.plot_directive',
    'sphinxcontrib.katex',
    'sphinx_gallery.gen_gallery'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# Exclude build directory and Jupyter backup files:
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# prerender tex
katex_prerender = True

# -- Autosummary settings -----------------------------------------------------
autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'member-order': 'groupwise',
    'inherited-members': True
}

# -- Gallery settings ---------------------------------------------------------

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',  # path to your example scripts
    'gallery_dirs': 'examples',  # path to where to save gallery generated output
    'capture_repr': ()
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
    'page_width': '65%',
    'sidebar_width': '250px',
    'body_max_width': 'auto',
    'fixed_sidebar': 'true',
    'github_button': 'true',
    'github_user': 'scikit-time',
    'github_repo': 'scikit-time',
    'github_type': 'star',
    'sidebar_collapse': 'true',

    'sidebar_header': '#96929c'
}
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html',
    ]
}

# -- nbsphinx settings --------------------------------------------------------

# List of arguments to be passed to the kernel that executes the notebooks:
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]


# hack to always update index rst so that static files are copied over during incremental build
def env_get_outdated(app, env, added, changed, removed):
    return ['index']


def setup(app):
    app.connect('env-get-outdated', env_get_outdated)
    app.add_css_file('custom.css')
    app.add_css_file('perfect-scrollbar/css/perfect-scrollbar.css')
    app.add_js_file('perfect-scrollbar/js/perfect-scrollbar.min.js')

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
