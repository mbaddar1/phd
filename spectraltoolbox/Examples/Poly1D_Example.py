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

import numpy as np
import numpy.linalg as npla
import numpy.random as npr
import scipy.stats as stats

from SpectralToolbox import Spectral1D as S1D

import matplotlib.pyplot as plt

STORE_FIG = True
FORMATS = ['pdf','png','eps']

N = 3
ls = ['-','--','-.',':']

# Support [0,1]
x = np.linspace(0,1,100)

# Uniform distribution
alpha = 0
beta = 0
dist = stats.beta(alpha+1,beta+1)
P = S1D.Poly1D(S1D.JACOBI,[alpha,beta])
V = P.GradVandermonde1D(2*x-1,N,0,norm=True)
plt.figure(figsize=(6,5))
plt.plot(x,dist.pdf(x),'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/LegendrePoly.'+ff,format=ff)

# Chebyshev
alpha = -0.5
beta = -0.5
dist = stats.beta(alpha+1,beta+1)
P = S1D.Poly1D(S1D.JACOBI,[alpha,beta])
V = P.GradVandermonde1D(2*x-1,N,0,norm=True)
plt.figure(figsize=(6,5))
plt.plot(x,dist.pdf(x),'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/ChebyshevPoly.'+ff,format=ff)

# Beta(2,5)
alpha = 1.
beta = 4.
dist = stats.beta(alpha+1,beta+1)
P = S1D.Poly1D(S1D.JACOBI,[beta,alpha])
V = P.GradVandermonde1D(2*x-1,N,0,norm=True)
plt.figure(figsize=(6,5))
plt.plot(x,dist.pdf(x),'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
plt.ylim([-1.5,5.])
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/Beta25Poly.'+ff,format=ff)



# Support [-inf,inf]
x = np.linspace(-3,3,100)

# Hermite Physicist's
w = np.exp(-x**2.)
P = S1D.Poly1D(S1D.HERMITEP,None)
V = P.GradVandermonde1D(x,N,0,True)
plt.figure(figsize=(6,5))
plt.plot(x,w,'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
plt.ylim([-5.,5.])
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/HermitePhysicistPoly.'+ff,format=ff)

# Hermite Function
w = np.ones(x.shape)
P = S1D.Poly1D(S1D.HERMITEF,None)
V = P.GradVandermonde1D(x,N,0,True)
plt.figure(figsize=(6,5))
plt.plot(x,w,'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
plt.ylim([-1.,1.1])
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/HermiteFunction.'+ff,format=ff)

# Hermite Probabilists'
dist = stats.norm()
P = S1D.Poly1D(S1D.HERMITEP_PROB,None)
V = P.GradVandermonde1D(x,N,0,True)
plt.figure(figsize=(6,5))
plt.plot(x,dist.pdf(x),'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
plt.ylim([-2.,2.])
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/HermiteProbabilistsPoly.'+ff,format=ff)



# Support [0,inf]
x = np.linspace(0,10,100)

# Laguerre polynomials
shape = 3.
scale = 1.
dist = stats.gamma(a=shape,scale=scale)
P = S1D.Poly1D(S1D.LAGUERREP, [shape-1])
V = P.GradVandermonde1D(x,N,0,True)
plt.figure(figsize=(6,5))
plt.plot(x,dist.pdf(x),'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
# plt.ylim([-2.,2.])
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/Laguerre3Poly.'+ff,format=ff)

# Laguerre polynomials
x = np.linspace(0,20,100)
shape = 9.
scale = 1.
dist = stats.gamma(a=shape,scale=scale)
P = S1D.Poly1D(S1D.LAGUERREP, [shape-1])
V = P.GradVandermonde1D(x,N,0,True)
plt.figure(figsize=(6,5))
plt.plot(x,dist.pdf(x),'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
# plt.ylim([-2.,2.])
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/Laguerre9Poly.'+ff,format=ff)

# Laguerre functions
x = np.linspace(0,20,100)
shape = 1.
scale = 1.
w = np.ones(x.shape)
P = S1D.Poly1D(S1D.LAGUERREF, [shape-1])
V = P.GradVandermonde1D(x,N,0,True)
plt.figure(figsize=(6,5))
plt.plot(x,w,'k-',linewidth=2,label='PDF')
for i in xrange(N+1):
    plt.plot(x,V[:,i],'k'+ls[i],label='$i$=%d'%i)
plt.xlabel('x')
# plt.ylim([-2.,2.])
plt.legend(loc='best')
plt.show(False)
if STORE_FIG:
    for ff in FORMATS:
        plt.savefig('figs/LaguerreFunctions.'+ff,format=ff)
