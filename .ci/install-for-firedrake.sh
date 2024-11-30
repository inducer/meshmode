sudo chmod -R a+rwX /github/home || true
sudo chmod -R a+rwX /__w || true

sudo apt update
sudo apt upgrade -y
sudo apt install time

. /home/firedrake/firedrake/bin/activate
grep -v loopy requirements.txt > /tmp/myreq.txt

# no need for these in the Firedrake tests
sed -i "/boxtree/ d" /tmp/myreq.txt
sed -i "/sumpy/ d" /tmp/myreq.txt
sed -i "/pytential/ d" /tmp/myreq.txt
sed -i "/pyopencl/ d" /tmp/myreq.txt

# This shouldn't be necessary, but...
# https://github.com/inducer/meshmode/pull/48#issuecomment-687519451
pip install pybind11
pip install pytest
pip install -r /tmp/myreq.txt
pip install pyopencl[pocl]

# Context: https://github.com/OP2/PyOP2/pull/605
python -m pip install --force-reinstall git+https://github.com/inducer/pytools.git

# bring in up-to-date loopy
pip uninstall -y loopy
pip install "git+https://github.com/inducer/loopy.git#egg=loopy"

pip install .
