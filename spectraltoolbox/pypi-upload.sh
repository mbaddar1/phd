#!/bin/bash

rm -rf dist SpectralToolbox.egg SpectralToolbox.egg-info
python setup.py sdist 
twine upload dist/*

