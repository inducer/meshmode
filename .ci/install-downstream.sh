#!/bin/bash

# Installs the downstream project specified in $DOWNSTREAM_PROJECT

set -o errexit

if [[ "$DOWNSTREAM_PROJECT" = "mirgecom" ]]; then
    git clone "https://github.com/illinois-ceesd/$DOWNSTREAM_PROJECT.git"
else
    git clone "https://github.com/inducer/$DOWNSTREAM_PROJECT.git"
fi

cd "$DOWNSTREAM_PROJECT"
echo "*** $DOWNSTREAM_PROJECT version: $(git rev-parse --short HEAD)"

transfer_requirements_git_urls ../requirements.txt ./requirements.txt
edit_requirements_txt_for_downstream_in_subdir

export CONDA_ENVIRONMENT=.test-conda-env-py3.yml

# Avoid slow or complicated tests in downstream projects
export PYTEST_ADDOPTS="-k 'not (slowtest or octave or mpi)'"

if [[ "$DOWNSTREAM_PROJECT" = "mirgecom" ]]; then
    # can't turn off MPI in mirgecom
    sudo apt-get update
    sudo apt-get install openmpi-bin libopenmpi-dev
    export CONDA_ENVIRONMENT=conda-env.yml
    export CISUPPORT_PARALLEL_PYTEST=no
else
    sed -i "/mpi4py/ d" requirements.txt
fi
