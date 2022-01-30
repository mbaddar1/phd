# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:21:41 2012

@author: dabi
"""

from numpy import *

from SpectralToolbox import Spectral1D

close('all')

# Poly Definition
alpha = (2.,)
N = 5
pp = Spectral1D.Poly1D(Spectral1D.LAGUERREP,alpha)

# Space
(x,w) = pp.GaussQuadrature(N)
r = linspace(0,x[-1],1000)

# Single Polynomial
P = pp.GradEvaluate(r,N+1,0)
dP = pp.GradEvaluate(r,N+1,1)

pFig = figure()
plot(r,P,'b',label='L_{N+1}')
plot(r,dP,'k',label='dL_{N+1}')
legend()
grid()

# Normalization constants
gs = zeros(10)
for i in range(0,10):
    gs[i] = pp.Gamma(i)
print gs

# Gauss points and weights
figure(pFig.number)
plot(x,zeros(x.shape),'or',label='Laguerre-Gauss')

# Gauss-Radau points and weights
(xr,wr) = pp.GaussRadauQuadrature(N)
figure(pFig.number)
plot(xr,zeros(xr.shape),'ok',label='Laguerre-Gauss-Radau')
legend()

# Vandermonde matrix
V = pp.GradVandermonde1D(r,N+1,0,norm=True)
vanderFig = figure()
for i in range(0,N+2):
    plot(r,V[:,i])
grid()