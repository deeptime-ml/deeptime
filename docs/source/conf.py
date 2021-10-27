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
from matplotlib import animation
from sphinx.application import Sphinx

import deeptime

project = 'deeptime'
copyright = '2020, AI4Science Group'
author = 'AI4Science Group'

version = f"{deeptime.__version__.split('+')[0]}"
# The full version, including alpha/beta/rc tags
release = f"{deeptime.__version__}"

master_doc = 'contents'

# -- Disable certain warnings ------------------------------------------------

sphinxlog_adapter = sphinx.util.logging.getLogger(sphinxcontrib.bibtex.__name__)
bibtex_bibfiles = ['references.bib']

class DuplicateLabelForKeysFilter(logging.Filter):

    def filter(self, record: LogRecord) -> int:
        return not (record.msg.startswith("duplicate label for keys") and record.levelno == logging.WARN)


sphinxlog_adapter.logger.addFilter(DuplicateLabelForKeysFilter())

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib.bibtex',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
    'sphinxcontrib.katex',
    'sphinx_gallery.gen_gallery',
    'sphinx_gallery.load_style'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# Exclude build directory and Jupyter backup files:
exclude_patterns = ['_build', '**.ipynb_checkpoints', '**/notebooks', '*.ipynb']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_additional_pages = {
    'index': 'index.html'
}

# html_logo = 'logo/deeptime_partial_white.svg'

# prerender tex
katex_prerender = False

# -- Autosummary settings -----------------------------------------------------
autosummary_generate = True

autodoc_default_options = {
    'inherited-members': True,
    'members': True,
    'member-order': 'groupwise',
    'special-members': '__call__',
    'exclude-members': '__init__'
}

# -- Gallery settings ---------------------------------------------------------
sphinx_gallery_conf = {
    'examples_dirs': ['../../examples/methods', '../../examples/datasets'],  # path to your example scripts
    'gallery_dirs': ['examples', 'datasets'],  # path to where to save gallery generated output
    'line_numbers': True,
    'show_memory': True,
    'capture_repr': (),
    'matplotlib_animations': True,
    'download_all_examples': False,
    'show_signature': False
}

plot_rcparams = {
    'animation.html': 'html5',
    'animation.writer': 'imagemagick' if animation.ImageMagickWriter.isAvailable() else 'ffmpeg'
}

# -- Napoleon settings --------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# -- Alabaster theme settings -------------------------------------------------
html_theme_options = {
    'page_width': '65%',
    'sidebar_width': '250px',
    'body_max_width': 'auto',
    'fixed_sidebar': 'true',
    'github_button': 'false',  # explicitly added in templates
    'github_user': 'deeptime-ml',
    'github_repo': 'deeptime',
    'github_type': 'star',
    'sidebar_collapse': 'true',
    'sidebar_header': '#96929c',
    'logo': 'logo/deeptime_romand_white.svg',
    'logo_name': 'false',
}
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'version.html',
        'relations.html',
        'searchbox.html',
        'github_button.html',
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
    return ['index', 'content']


def skip(app, what, name, obj, skip, options):
    if name == '__init__':
        return True
    return None


# # Patch parse, see https://michaelgoerz.net/notes/extending-sphinx-napoleon-docstring-sections.html
# from sphinx.ext.napoleon.docstring import NumpyDocstring
#
#
# # first, we define new methods for any new sections and add them to the class
# def parse_keys_section(self, section):
#     return self._format_fields('Keys', self._consume_fields())
#
#
# NumpyDocstring._parse_keys_section = parse_keys_section
#
#
# def parse_attributes_section(self, section):
#     return self._format_fields('Attributes', self._consume_fields())
#
#
# NumpyDocstring._parse_attributes_section = parse_attributes_section
#
#
# def parse_class_attributes_section(self, section):
#     return self._format_fields('Class Attributes', self._consume_fields())
#
#
# NumpyDocstring._parse_class_attributes_section = parse_class_attributes_section
#
#
# # we now patch the parse method to guarantee that the the above methods are
# # assigned to the _section dict
# def patched_parse(self):
#     self._sections['keys'] = self._parse_keys_section
#     self._sections['class attributes'] = self._parse_class_attributes_section
#     self._unpatched_parse()
#
#
# NumpyDocstring._unpatched_parse = NumpyDocstring._parse
# NumpyDocstring._parse = patched_parse


def setup(app: Sphinx):
    app.connect('env-get-outdated', env_get_outdated)
    app.add_css_file('custom.css')
    app.add_css_file('perfect-scrollbar/css/perfect-scrollbar.css')
    app.add_js_file('perfect-scrollbar/js/perfect-scrollbar.min.js')
    app.add_js_file('perfect-scrollbar/js/perfect-scrollbar.min.js')
    app.add_js_file('d3.v5.min.js')
    app.add_js_file('d3-legend.min.js')
    app.connect("autodoc-skip-member", skip)

    if app.tags.has('notebooks'):
        global katex_prerender
        global exclude_patterns
        # katex_prerender = True
        exclude_patterns.remove('**/notebooks')
        exclude_patterns.remove('*.ipynb')
        app.setup_extension('nbsphinx')
