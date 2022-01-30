# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:17:13 2012

@author: dabi

Comparison MatLab/Python Sparse grids

"""

from numpy import *

pyNodes = genfromtxt('nodes.txt', unpack=True)
pyWeights = genfromtxt('weights.txt', unpack=True)

matNodes = genfromtxt('SpGridMatlab/nodesMat.txt', unpack=True)
matWeights = genfromtxt('SpGridMatlab/weightsMat.txt', unpack=True)

errNodes = norm( (pyNodes-matNodes).ravel(), inf)
errWeights = norm( (pyWeights-matWeights).ravel(), inf)

print "Max error on nodes: %e" % errNodes
print "Max error on weights: %e" % errWeights