# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 09:00:57 2012

@author: dabi

Use Legendre polynomials to make a convergence test on the function

f(x1,...,xn) = sin(2*pi*x1) + ... + sin(2*pi*xn)

"""

from numpy import *

import itertools

import Spectral1D
import SpectralND

close('all')

def f(x):
    out = ones(size(x,0));
    for i in range(0,size(x,1)):
        out = out + x[:,i]**5.
        #out = out * sin(pi*x[:,i])
    return out

d = 2;

np = 100;
x = linspace(-1,1,np);
xs = [x]
for i in range(1,d):
    xs.append(x)
    
xKron = asarray(list(itertools.product(*xs)))
fx = f(xKron)

# If 2Dim
if d == 2:
    X1 = reshape(xKron[:,0],(np,np))
    X2 = reshape(xKron[:,1],(np,np))
    # Plot function
    figure()
    cs = contourf(X1,X2,reshape(f(xKron),(np,np)))
    cbar = colorbar(cs)

# Construct polynomials
pTypes = []
for i in range(0,d):
    pTypes.append(Spectral1D.Poly1D(Spectral1D.JACOBI,(0,0)))

NN = range(1,20)
ErrL2 = zeros(len(NN))
pp = SpectralND.PolyND(pTypes)
for i in range(0,len(NN)):
    N = NN[i]
    
    xGQ,wGQ = pp.GaussQuadrature(tile(N,d))
    rs = []
    for j in range(0,d):
        rs.append(unique(xGQ[:,j]))
    V = pp.GradVandermondePascalSimplex(rs,N,zeros(d,dtype=int))
    # project
    [fHat, res, rank, s] = linalg.lstsq(V,f(xGQ))
    
    # Interpolate
    VI = pp.GradVandermondePascalSimplex(xs,N,zeros(d,dtype=int))
    fI = dot(VI, fHat)
    
    # If 2Dim
    if d == 2:
        X1 = reshape(xKron[:,0],(np,np))
        X2 = reshape(xKron[:,1],(np,np))
        # Plot function
        figure()
        cs = contourf(X1,X2,reshape(fI,(np,np)))
        cbar = colorbar(cs)
    
    # Error estimate
    ErrL2[i] = norm(fI-fx,2)

# Plot convergence
figure()
semilogy(NN,ErrL2,'o-')