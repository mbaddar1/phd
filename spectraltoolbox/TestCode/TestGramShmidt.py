# -*- coding: utf-8 -*-
"""
Created on Mon May 21 16:56:26 2012

@author: dabi
"""

from numpy import *
import matplotlib.pyplot as plt

import Spectral1D

plt.close('all')

N = 4

poly = Spectral1D.Poly1D(Spectral1D.JACOBI,(0.,0.))
x,w = poly.GaussQuadrature(2*N)

# Gram Shmidt (Legendre Polynomials themselves)
def f(x):
    return x
V = poly.GramShmidt(f,N,Spectral1D.JACOBI,(0.,0.))
plt.figure()
for i in range(0,N+1):
    plt.plot(x,V[:,i])

L = poly.GradVandermonde1D(x,N,0,False)
plt.figure()
for i in range(0,N+1):
    plt.plot(x,L[:,i])

# Gram Shmidt: check orthogonality for generic polynomials
x,w = poly.GaussQuadrature(N)
def f(x):
    return x
V = poly.GradVandermonde1D(f(x),N,0,True)
(V,r) = qr(V)
#V = poly.GramShmidt(f,N,Spectral1D.JACOBI,(0.,0.))
plt.figure()
for i in range(0,N+1):
    plt.plot(x,V[:,i])

orth = zeros((N+1,N+1))
for i in range(0,N+1):
    for j in range(0,N+1):
        # Use numerical quadrature to compute the orthogonality constants
        orth[i,j] = dot( V[:,i], V[:,j])

