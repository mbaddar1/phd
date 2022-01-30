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
import matplotlib.pyplot as plt
import SpectralToolbox.Spectral1D as S1D

STORE_FIG = False
FORMATS = ['pdf','png','eps']

# def f(x):
#     return np.exp(x) # Exact quad e-e^-1
# exact = np.exp(1)-np.exp(-1)

# def f(x):
#     return np.exp(100*x) # Exact quad 1/100 * ( e^(100) - e^(-100) )
# exact = .01 * (np.exp(100) - np.exp(-100))

# def f(x):
#     return np.exp(x) * ( 2. / math.cosh( 4. * np.sin(40. * x) )**np.exp(x)
# analytical_exact = False

def f(x):
    return 1 + np.abs(x) # Exact quad 1
exact = 3.

nlev = 6
nquad = 5
npt = np.zeros((nquad,nlev))
err = np.zeros((nquad,nlev))

P = S1D.Poly1D(S1D.JACOBI,[0.,0.])

plt.figure()
# Gauss
n = 0
for l in range(1,nlev+1):
    (x,w) = P.GaussQuadrature(2**l)
    npt[n,l-1] = len(x)
    err[n,l-1] = np.abs( exact - np.dot(f(x),w) )
err[ n, np.where( err[n,:] == 0. ) ] = 1e-16
err /= np.abs(exact)
plt.semilogy(npt[n,:],err[n,:],'o-',label="Gauss")

# Gauss-Lobatto
n += 1
for l in range(1,nlev+1):
    (x,w) = P.GaussLobattoQuadrature(2**l)
    npt[n,l-1] = len(x)
    err[n,l-1] = np.abs( exact - np.dot(f(x),w) )
err[ n, np.where( err[n,:] == 0. ) ] = 1e-16
err /= np.abs(exact)
plt.semilogy(npt[n,:],err[n,:],'o-',label="Gauss-Lobatto")

# Clenshaw-Curtis
n += 1
for l in range(1,nlev+1):
    (x,w) = S1D.cc(l,norm=False)
    npt[n,l-1] = len(x)
    err[n,l-1] = np.abs( exact - np.dot(f(x),w) )
err[ n, np.where( err[n,:] == 0. ) ] = 1e-16
err /= np.abs(exact)
plt.semilogy(npt[n,:],err[n,:],'o-',label="Clenshaw-Curtis")

# Nested-Gauss
n += 1
for l in range(1,nlev+1):
    (x,w) = S1D.nestedgauss(l,norm=False)
    npt[n,l-1] = len(x)
    err[n,l-1] = np.abs( exact - np.dot(f(x),w) )
err[ n, np.where( err[n,:] == 0. ) ] = 1e-16
err /= np.abs(exact)
plt.semilogy(npt[n,:],err[n,:],'o-',label="Nested-Gauss")

# Nested-Lobatto
n += 1
for l in range(1,nlev+1):
    (x,w) = S1D.nestedlobatto(l,norm=False)
    npt[n,l-1] = len(x)
    err[n,l-1] = np.abs( exact - np.dot(f(x),w) )
err[ n, np.where( err[n,:] == 0. ) ] = 1e-16
err /= np.abs(exact)
plt.semilogy(npt[n,:],err[n,:],'o-',label="Nested-Lobatto")

plt.legend(loc='best')
plt.show(False)
