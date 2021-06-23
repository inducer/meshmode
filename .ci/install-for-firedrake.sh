sudo chown -R $(whoami) /github/home || true
sudo apt update
sudo apt upgrade -y
sudo apt install time

# Otherwise we get missing CL symbols
export PYOPENCL_CL_PRETEND_VERSION=2.1
sudo apt install -y pocl-opencl-icd ocl-icd-opencl-dev

. /home/firedrake/firedrake/bin/activate
grep -v loopy requirements.txt > /tmp/myreq.txt

# no need for these in the Firedrake tests
sed -i "/boxtree/ d" /tmp/myreq.txt
sed -i "/sumpy/ d" /tmp/myreq.txt
sed -i "/pytential/ d" /tmp/myreq.txt

# This shouldn't be necessary, but...
# https://github.com/inducer/meshmode/pull/48#issuecomment-687519451
pip install pybind11

pip install pytest

pip install -r /tmp/myreq.txt 

# Context: https://github.com/OP2/PyOP2/pull/605
python -m pip install --force-reinstall git+https://github.com/inducer/pytools.git

pip install .
