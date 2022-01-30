#!/bin/bash

rm -rf dist SpectralToolbox.egg SpectralToolbox.egg-info
python setup.py sdist bdist_wheel
twine upload dist/* -r testpypi
