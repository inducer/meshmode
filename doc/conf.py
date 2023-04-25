from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

extensions.extend([  # noqa: F821
    "sphinx.ext.graphviz",
    "sphinxcontrib.tikz",
])

tikz_tikzlibraries = "decorations.markings"

copyright = "2014-21, Meshmode contributors"

ver_dic = {}
exec(
        compile(
            open("../meshmode/version.py").read(), "../meshmode/version.py", "exec"),
        ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]

intersphinx_mapping = {
    "arraycontext": ("https://documen.tician.de/arraycontext", None),
    "fenics": ("https://fenics.readthedocs.io/projects/fiat/en/latest", None),
    "FInAT": ("https://finat.github.io/FInAT/", None),
    "firedrake": ("https://firedrakeproject.org", None),
    "gmsh_interop": ("https://documen.tician.de/gmsh_interop", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "loopy": ("https://documen.tician.de/loopy", None),
    "meshpy": ("https://documen.tician.de/meshpy", None),
    "modepy": ("https://documen.tician.de/modepy", None),
    "mpi4py": ("https://mpi4py.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pymetis": ("https://documen.tician.de/pymetis", None),
    "pyopencl": ("https://documen.tician.de/pyopencl", None),
    "python": ("https://docs.python.org/3", None),
    "pytools": ("https://documen.tician.de/pytools", None),
    "recursivenodes": ("https://tisaac.gitlab.io/recursivenodes", None),
}


# Some modules need to import things just so that sphinx can resolve symbols in
# type annotations. Often, we do not want these imports (e.g. of PyOpenCL) when
# in normal use (because they would introduce unintended side effects or hard
# dependencies). This flag exists so that these imports only occur during doc
# build. Since sphinx appears to resolve type hints lexically (as it should),
# this needs to be cross-module (since, e.g. an inherited arraycontext
# docstring can be read by sphinx when building meshmode, a dependent package),
# this needs a setting of the same name across all packages involved, that's
# why this name is as global-sounding as it is.
import sys

sys._BUILDING_SPHINX_DOCS = True
