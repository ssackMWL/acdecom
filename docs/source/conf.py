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
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        message='Matplotlib is currently using agg, which is a'
                                ' non-GUI backend, so cannot show the figure.')

sys.path.insert(0, os.path.abspath('../../src'))
sys.path.append(os.path.abspath('sphinxext'))

# -- Project information -----------------------------------------------------

project = 'acdecom'
copyright = '2020, Stefan Sack'
author = 'Stefan Sack'

# The full version, including alpha/beta/rc tags
release = '20/06/12'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx_gallery.gen_gallery','matplotlib.sphinxext.plot_directive', 'sphinx.ext.napoleon',
 'numpydoc','sphinx.ext.autodoc', 'sphinx.ext.autosummary','sphinxcontrib.rawfiles']

numpydoc_show_class_members=False

autosummary_generate = True
autoclass_content = "both"
html_logo = 'image/logo.png'
html_favicon = 'image/logo.ico'

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output

}



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
html_static_path = ['_static']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinxdoc'

rawfiles = ['image']