from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

extensions.extend([
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
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/pytools": None,
    "https://documen.tician.de/pyopencl": None,
    "https://documen.tician.de/meshpy": None,
    "https://documen.tician.de/modepy": None,
    "https://documen.tician.de/arraycontext": None,
    "https://documen.tician.de/loopy": None,
    "https://documen.tician.de/gmsh_interop": None,
    "https://documen.tician.de/pymetis": None,
    "https://firedrakeproject.org/": None,
    "https://tisaac.gitlab.io/recursivenodes/": None,
    "https://fenics.readthedocs.io/projects/fiat/en/latest/": None,
    "https://finat.github.io/FInAT/": None,
    "h5py": ("https://docs.h5py.org/en/stable", None),
}
