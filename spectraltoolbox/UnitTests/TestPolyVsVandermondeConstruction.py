#
# This file is part of SpectralToolbox.
#
# TensorToolbox is free software: you can redistribute it and/or modify
# it under the terms of the LGNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TensorToolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LGNU Lesser General Public License for more details.
#
# You should have received a copy of the LGNU Lesser General Public License
# along with TensorToolbox.  If not, see <http://www.gnu.org/licenses/>.
#
# DTU UQ Library
# Copyright (C) 2012-2015 The Technical University of Denmark
# Scientific Computing Section
# Department of Applied Mathematics and Computer Science
#
# Author: Daniele Bigoni
#

import sys
import operator
import time

import numpy as np
import numpy.linalg as npla
import numpy.random as npr

from SpectralToolbox import Spectral1D as S1D

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

PLOTTING = True
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''

def print_ok(string):
    print bcolors.OKGREEN + "[SUCCESS] " + string + bcolors.ENDC

def print_fail(string,msg=''):
    print bcolors.FAIL + "[FAILED] " + string + bcolors.ENDC
    if msg != '':
        print bcolors.FAIL + msg + bcolors.ENDC

exp = int(sys.argv[1])
if exp == 1:
    poly_type = S1D.JACOBI
    params = [0.,0.]
    print "Legendre polynomials"
elif exp == 2:
    poly_type = S1D.JACOBI
    params = [-0.5,-0.5]
    print "Chebyshev polynomials"
elif exp == 3:
    poly_type = S1D.HERMITEP
    params = None
    print "Hermite Physicists' polynomials"
elif exp == 4:
    poly_type = S1D.HERMITEP_PROB
    params = None
    print "Hermite Probabilists' polynomials"
elif exp == 5:
    poly_type = S1D.HERMITEF
    params = None
    print "Hermite functions"
elif exp == 6:
    poly_type = S1D.LAGUERREP
    params = [1.]
    print "Laguerre polynomials"
elif exp == 7:
    poly_type = S1D.LAGUERREF
    params = [1.]
    print "Laguerre functions"
else:
    raise "Input error"
if len(sys.argv) == 2:
    grad = 0
else:
    grad = int(sys.argv[2])

P = S1D.Poly1D(poly_type,params)

###########################
# Test Jacobi polynomials
###########################
N = 80
ntime = 10000
(x,w) = P.Quadrature(N)

r_old = P.GradEvaluate(x,N,grad,norm=True)

if PLOTTING:
    plt.figure()
    plt.plot(x,r_old)
    plt.show(block=False)

start = time.clock()
for i in xrange(ntime): pol = P.GradEvaluate(x,N,grad)
stop = time.clock()
print "Time polynomial: " + str((stop-start)/float(ntime))

##############################
# Construct Vandermonde by GradEvaluate
##############################
N = 80
ntime = 1000
(x,w) = P.Quadrature(N)

start = time.clock()
for i in xrange(ntime):
    VGE = np.zeros((len(x),N+1))
    for i in xrange(N+1):
        VGE[:,i] = P.GradEvaluate(x,i,grad)
stop = time.clock()
print "Time Vandermonde with GradEvaluate: " + str((stop-start)/float(ntime))

###########################
# Test Vandermonde construction
###########################
N = 80
ntime = 1000
(x,w) = P.Quadrature(N)

start = time.clock()
for i in xrange(ntime): V = P.GradVandermonde1D(x,N,grad)
stop = time.clock()
print "Time Vandermonde: " + str((stop-start)/float(ntime))

l2err = npla.norm(VGE-V,ord='fro') / npla.norm(V,ord='fro')
if l2err < 1e-11:
    print_ok("l2error Vandermonde: " + str( l2err ))
else:
    print_fail("l2error Vandermonde: " + str( l2err ))

