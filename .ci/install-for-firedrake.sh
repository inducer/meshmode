sudo chown -R $(whoami) /github/home || true
sudo apt update
sudo apt upgrade -y
sudo apt install time
sudo apt install -y pocl-opencl-icd ocl-icd-opencl-dev

. /home/firedrake/firedrake/bin/activate
grep -v loopy requirements.txt > /tmp/myreq.txt
sed -i s/pyopencl.git/pyopencl.git@v2020.2.2/ /tmp/myreq.txt

# This shouldn't be necessary, but...
# https://github.com/inducer/meshmode/pull/48#issuecomment-687519451
pip install pybind11

# The Firedrake container is based on Py3.6 as of 2020-10-10, which
# doesn't have dataclasses.
pip install dataclasses

pip install pytest

pip install -r /tmp/myreq.txt
pip install --force-reinstall git+https://github.com/benSepanski/loopy.git@firedrake-usable_for_potentials
pip install .
