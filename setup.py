#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main():
    from setuptools import setup, find_packages

    version_dict = {}
    init_filename = "meshmode/version.py"
    exec(compile(open(init_filename, "r").read(), init_filename, "exec"),
            version_dict)

    setup(name="meshmode",
          version=version_dict["VERSION_TEXT"],
          description="High-order polynomial discretizations of and on meshes",
          long_description=open("README.rst", "rt").read(),
          author="Andreas Kloeckner",
          author_email="inform@tiker.net",
          license="MIT",
          url="https://documen.tician.de/meshmode",
          classifiers=[
              "Development Status :: 3 - Alpha",
              "Intended Audience :: Developers",
              "Intended Audience :: Other Audience",
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Natural Language :: English",
              "Programming Language :: Python",
              "Programming Language :: Python :: 3",
              "Programming Language :: Python :: 3.6",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Topic :: Scientific/Engineering",
              "Topic :: Scientific/Engineering :: Information Analysis",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Scientific/Engineering :: Visualization",
              "Topic :: Software Development :: Libraries",
              "Topic :: Utilities",
              ],

          packages=find_packages(),
          python_requires="~=3.6",
          install_requires=[
              "numpy",
              "modepy>=2020.2",
              "gmsh_interop",
              "pytools>=2020.4.1",
              "pytest>=2.3",

              # 2019.1 is required for the Firedrake CIs, which use an very specific
              # version of Loopy.
              "loopy>=2019.1",

              "recursivenodes",
              "dataclasses; python_version<='3.6'",
              ],
          extras_require={
              "visualization": ["h5py"],
              },
          )


if __name__ == "__main__":
    main()
