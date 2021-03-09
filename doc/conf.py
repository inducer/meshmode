import sys  # noqa: F401
import os  # noqa: F401

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = "1.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
#source_encoding = "utf-8-sig"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = u"meshmode"
copyright = u"2014, Andreas Klöckner"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
ver_dic = {}
exec(
        compile(
            open("../meshmode/version.py").read(), "../meshmode/version.py", "exec"),
        ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output ----------------------------------------------

html_theme = "furo"

html_theme_options = {
        }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/pytools": None,
    "https://documen.tician.de/pyopencl": None,
    "https://documen.tician.de/meshpy": None,
    "https://documen.tician.de/modepy": None,
    "https://documen.tician.de/loopy": None,
    "https://documen.tician.de/gmsh_interop": None,
    "https://firedrakeproject.org/": None,
    "https://tisaac.gitlab.io/recursivenodes/": None,
    "https://fenics.readthedocs.io/projects/fiat/en/latest/": None,
    "https://finat.github.io/FInAT/": None,
    "h5py": ("https://docs.h5py.org/en/stable", None),
}

autoclass_content = "class"
autodoc_typehints = "description"
