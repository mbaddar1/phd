#!/bin/bash

set -e

TEST_NAME=pypi-SpectralToolbox
ENV_NAME=venv_${TEST_NAME}_${PYTHON_EXE}
ENV_DIR=$HOME/workspace/${TEST_NAME}/
if [ "$PYTHON_EXE" = "python2" ]; then
    CONDA_HOME=/home/dabi/anaconda2/
elif [ "$PYTHON_EXE" = "python3" ]; then
    CONDA_HOME=/home/dabi/anaconda3/
fi

if [ "$VENV_SYS" = "virtualenv" ]; then
    rm -rf $ENV_DIR/$ENV_NAME
    virtualenv --python=$PYTHON_EXE $ENV_DIR/$ENV_NAME
    source $ENV_DIR/$ENV_NAME/bin/activate
elif [ "$VENV_SYS" = "anaconda" ]; then
    $CONDA_HOME/bin/conda-env remove -y --name $ENV_NAME
    if [ "$PYTHON_EXE" = "python2" ]; then
        $CONDA_HOME/bin/conda create -y --name $ENV_NAME python=2 pip
    elif [ "$PYTHON_EXE" = "python3" ]; then
        $CONDA_HOME/bin/conda create -y --name $ENV_NAME python=3 pip
    fi
    source $CONDA_HOME/bin/activate $ENV_NAME
else
    echo "VENV_SYS not recognized"
    exit 1
fi
pip install --upgrade pip

# Install
rm -fr tmp_sdist
mkdir tmp_sdist
cd tmp_sdist
pip install --no-binary :all: SpectralToolbox

# Clean
cd ..
rm -rf tmp_sdist

# Unit tests
echo "[UNITTESTS]"
python -c "import SpectralToolbox as ST; ST.tests.run_all()"

if [ "$VENV_SYS" = "virtualenv" ]; then
    deactivate
    rm -rf $ENV_DIR/$ENV_NAME
elif [ "$VENV_SYS" = "anaconda" ]; then
    source $CONDA_HOME/bin/deactivate
    $CONDA_HOME/bin/conda-env remove -y --name $ENV_NAME
fi
