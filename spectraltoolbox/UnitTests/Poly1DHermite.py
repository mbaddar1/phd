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

maxord = 10
skip = 2
polder = 3

TESTS = [2]

if 0 in TESTS:
    ########################################
    # Test Hermite Physicists' polynimials
    ########################################
    P = S1D.Poly1D(S1D.HERMITEP, None)

    # Visually check polynomials
    x = np.linspace(-2,2,100)
    V = P.GradVandermonde1D(x,maxord,0)

    plt.figure()
    for i in xrange(0,maxord+1,skip):
        plt.plot(x,V[:,i],label='ord %d' % i)
    plt.title('Hermite Physicists\' polynomials')
    plt.legend()
    plt.grid()
    plt.show(False)

    # Visually check first 3 derivatives of polder polynomial
    x = np.linspace(-2,2,100)
    Pn = P.GradEvaluate(x,polder,0)
    dPn = P.GradEvaluate(x,polder,1)
    ddPn = P.GradEvaluate(x,polder,2)
    dddPn = P.GradEvaluate(x,polder,3)

    plt.figure()
    plt.plot(x,Pn,label='f')
    plt.plot(x,dPn,label='df')
    plt.plot(x,ddPn,label='ddf')
    plt.plot(x,dddPn,label='dddf')
    plt.title('Derivatives Hermite Physicists\'')
    plt.legend()
    plt.grid()
    plt.show(False)

    # Check orthogonality of the polynomials
    (x,w) = P.Quadrature(maxord)
    V = P.GradVandermonde1D(x,maxord,0,norm=False)
    orth = np.dot( V.T, np.dot( np.diag(w), V ) )

    plt.figure()
    plt.imshow(orth,interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('Orthogonality Hermite Physicists\'')
    plt.show(False)

    print "Orthogonality Hermite Physicists\' coeffs: " 
    print np.diag(orth)

    # Check orthonormality of the polynomials
    (x,w) = P.Quadrature(maxord)
    V = P.GradVandermonde1D(x,maxord,0)
    orth = np.dot( V.T, np.dot( np.diag(w), V ) )

    plt.figure()
    plt.imshow(orth,interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('Orthonormality Hermite Physicists\'')
    plt.show(False)

    print "Orthonormality Hermite Physicists\' coeffs: " 
    print np.diag(orth)


if 1 in TESTS:
    ########################################
    # Test Hermite Proabilists' polynimials
    ########################################
    P = S1D.Poly1D(S1D.HERMITEP_PROB, None)

    # Visually check polynomials
    x = np.linspace(-2,2,100)
    V = P.GradVandermonde1D(x,maxord,0)

    plt.figure()
    for i in xrange(0,maxord+1,skip):
        plt.plot(x,V[:,i],label='ord %d' % i)
    plt.title('Hermite Probabilists\' polynomials')
    plt.legend()
    plt.grid()
    plt.show(False)

    # Visually check first 3 derivatives of polder polynomial
    x = np.linspace(-2,2,100)
    Pn = P.GradEvaluate(x,polder,0)
    dPn = P.GradEvaluate(x,polder,1)
    ddPn = P.GradEvaluate(x,polder,2)
    dddPn = P.GradEvaluate(x,polder,3)

    plt.figure()
    plt.plot(x,Pn,label='f')
    plt.plot(x,dPn,label='df')
    plt.plot(x,ddPn,label='ddf')
    plt.plot(x,dddPn,label='dddf')
    plt.title('Derivatives Hermite Probabilists\'')
    plt.legend()
    plt.grid()
    plt.show(False)

    # Check orthogonality of the polynomials
    (x,w) = P.Quadrature(maxord)
    V = P.GradVandermonde1D(x,maxord,0,norm=False)
    orth = np.dot( V.T, np.dot( np.diag(w), V ) )

    plt.figure()
    plt.imshow(orth,interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('Orthogonality Hermite Probabilists\'')
    plt.show(False)

    print "Orthogonality Hermite Probabilists\' coeffs: " 
    print np.diag(orth)

    # Check orthonormality of the polynomials
    (x,w) = P.Quadrature(maxord)
    V = P.GradVandermonde1D(x,maxord,0)
    orth = np.dot( V.T, np.dot( np.diag(w), V ) )

    plt.figure()
    plt.imshow(orth,interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('Orthonormality Hermite Probabilists\'')
    plt.show(False)

    print "Orthonormality Hermite Probabilists\' coeffs: " 
    print np.diag(orth)


if 2 in TESTS:
    ########################################
    # Test Hermite Proabilists' polynimials
    ########################################
    P = S1D.Poly1D(S1D.HERMITEF, None)

    # Visually check polynomials
    x = np.linspace(-8,8,100)
    V = P.GradVandermonde1D(x,maxord,0)

    plt.figure()
    for i in xrange(0,maxord+1,skip):
        plt.plot(x,V[:,i],label='ord %d' % i)
    plt.title('Hermite functions')
    plt.legend()
    plt.grid()
    plt.show(False)

    # Visually check first 3 derivatives of polder polynomial
    x = np.linspace(-8,8,100)
    Pn = P.GradEvaluate(x,polder,0)
    dPn = P.GradEvaluate(x,polder,1)
    ddPn = P.GradEvaluate(x,polder,2)
    dddPn = P.GradEvaluate(x,polder,3)

    plt.figure()
    plt.plot(x,Pn,label='f')
    plt.plot(x,dPn,label='df')
    plt.plot(x,ddPn,label='ddf')
    plt.plot(x,dddPn,label='dddf')
    plt.title('Derivatives Hermite functions\'')
    plt.legend()
    plt.grid()
    plt.show(False)

    # Check orthogonality of the polynomials
    (x,w) = P.Quadrature(maxord)
    V = P.GradVandermonde1D(x,maxord,0,norm=False)
    orth = np.dot( V.T, np.dot( np.diag(w), V ) )

    plt.figure()
    plt.imshow(orth,interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('Orthogonality Hermite functions\'')
    plt.show(False)

    print "Orthogonality Hermite functions\' coeffs: " 
    print np.diag(orth)

    # Check orthonormality of the polynomials
    (x,w) = P.Quadrature(maxord)
    V = P.GradVandermonde1D(x,maxord,0)
    orth = np.dot( V.T, np.dot( np.diag(w), V ) )

    plt.figure()
    plt.imshow(orth,interpolation='none', origin='lower')
    plt.colorbar()
    plt.title('Orthonormality Hermite functions\'')
    plt.show(False)

    print "Orthonormality Hermite functions\' coeffs: " 
    print np.diag(orth)
