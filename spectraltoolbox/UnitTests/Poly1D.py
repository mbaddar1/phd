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

###########################
# Test Jacobi polynomials
###########################
alpha = 0.
beta = 0.
N = 80
ntime = 10000
P = S1D.Poly1D(S1D.JACOBI,[alpha,beta])
(xp,wp) = P.Quadrature(N)
x = np.linspace(-1.,1.,1000)

r_old = P.GradEvaluate(x,N,0,norm=True)

if PLOTTING:
    plt.figure()
    plt.plot(x,r_old,label="JACOBI")
    plt.legend()
    plt.show(block=False)

start = time.clock()
for i in range(ntime): pol = P.GradEvaluate(x,N,0)
stop = time.clock()
print "Time polynomial: " + str((stop-start)/float(ntime))

###########################
# Test Vandermonde construction
###########################
alpha = 0.
beta = 0.
N = 80
ntime = 1000
P = S1D.Poly1D(S1D.JACOBI,[alpha,beta])
(xp,wp) = P.Quadrature(N)
x = np.linspace(-1.,1.,1000)

start = time.clock()
for i in range(ntime): V = P.GradVandermonde1D(x,N,0)
stop = time.clock()
print "Time Vandermonde: " + str((stop-start)/float(ntime))

print "l2error: " + str( npla.norm(pol-V[:,-1]) )
