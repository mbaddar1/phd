# -*- coding: utf-8 -*-

#
# This file is part of SpectralToolbox.
#
# SpectralToolbox is free software: you can redistribute it and/or modify
# it under the terms of the LGNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SpectralToolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LGNU Lesser General Public License for more details.
#
# You should have received a copy of the LGNU Lesser General Public License
# along with SpectralToolbox.  If not, see <http://www.gnu.org/licenses/>.
#
# DTU UQ Library
# Copyright (C) 2012-2015 The Technical University of Denmark
# Scientific Computing Section
# Department of Applied Mathematics and Computer Science
#
# Author: Daniele Bigoni
#

import numpy as np
import numpy.fft as npfft
import matplotlib.pyplot as plt
import scipy.stats as stats

import SpectralToolbox.Spectral1D as S1D

# Even-odd example
evenodd = 1                     # 1 is odd, 0 even
N = 100 + evenodd

def f(x):
    return np.exp( - x**2. )

# def f(x):
#     return np.sin(10 * x)

P = S1D.Poly1D( S1D.FOURIER, None )
(x,w) = P.Quadrature(N, normed=True)

sig = f(x-np.pi)
fhat = P.project(x, sig, N)

plt.figure()
plt.semilogy( np.abs(fhat) )
plt.title('Coeff. decay')

# define an interpolation function
def interp(x):
    if not isinstance(x,np.ndarray):
        x = np.array([x])
    if N % 2 == 0:
        k = np.arange(1.,N // 2 + 1)
    else:
        k = np.arange(1.,(N+1) / 2)
    out = fhat[0] + np.dot( fhat[1::2], np.cos( k[:,np.newaxis] * x[np.newaxis,:] ) ) - np.dot( fhat[2::2], np.sin( k[:,np.newaxis] * x[np.newaxis,:] ) )
    # out = fhat[0].real / N + 2./N * ( np.dot( fhat[1:].real, np.cos( k[:,np.newaxis] * x[np.newaxis,:] ) ) - np.dot( fhat[1:].imag, np.sin( k[:,np.newaxis] * x[np.newaxis,:] ) ) )
    return out

F = f(x-np.pi)
fint = P.interpolate(x, F, x, N)

plt.figure()
plt.plot(x, f(x-np.pi), label='Exact')
plt.plot(x, fint, label='Approx')
plt.legend()

# Compute L2 error
Nsamp = 10000
xsamp = stats.uniform().rvs(Nsamp)

fexact = f( 2 * np.pi * xsamp - np.pi )
finterp = P.interpolate(x, F, 2 * np.pi * xsamp, N )
L2err = np.sqrt( 1./Nsamp * np.sum( (fexact - finterp)**2. ) )

print "L2err: %e" % L2err

# Convergence test
Nsamp = 10000
xsamp = stats.uniform().rvs(Nsamp)
Nref = 10
L2err = np.zeros(Nref)
for i in xrange(Nref):
    N = 2**(i+1) + evenodd
    P = S1D.Poly1D( S1D.FOURIER, None )
    (x,w) = P.Quadrature(N, normed=True)
    F = f(x-np.pi)
    fexact = f( 2 * np.pi * xsamp - np.pi )
    finterp = P.interpolate(x, F, 2 * np.pi * xsamp, N )
    L2err[i] = np.sqrt( 1./Nsamp * np.sum( (fexact - finterp)**2. ) )

plt.figure()
plt.loglog( 2**(np.arange(1,Nref+1))+evenodd, L2err, 'o-')
plt.xlabel('N points')
plt.ylabel('L2 err')
plt.grid()

plt.show(False)
