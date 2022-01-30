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
# Copyright (C) 2015-2016 Massachusetts Institute of Technology
# Uncertainty Quantification group
# Department of Aeronautics and Astronautics
#
# Author: Daniele Bigoni
#

import sys
import warnings
import numpy as np
from numpy import linalg as LA
from numpy import fft as FFT
import math

from scipy.special import gamma as gammaF
from scipy.special import gammaln as gammalnF
from scipy.special import factorial
from scipy.special import comb as SPcomb
from scipy import sparse as scsp

import SpectralToolbox.SparseGrids as SG

__all__ = ['FirstPolynomialDerivativeMatrix','PolynomialDerivativeMatrix',
           'BarycentricWeights', 'LagrangeInterpolationMatrix',
           'LagrangeInterpolate']

def FirstPolynomialDerivativeMatrix(x):
    """
    PolynomialDerivativeMatrix(): Assemble the first Polynomial Derivative Matrix using matrix multiplication.

    Syntax:
        ``D = FirstPolynomialDerivativeMatrix(x)``

    Input:
        * ``x`` = (1d-array,float) set of points on which to evaluate the derivative matrix

    Output:
        * ``D`` = derivative matrix

    Notes:
        Algorithm (37) from :cite:`Kopriva2009`
    """
    N = x.shape[0]
    w = BarycentricWeights(x)
    D = np.zeros((N,N))
    for i in range(0,N):
        for j in range(0,N):
            if (j != i):
                D[i,j] = w[j]/w[i] * 1./(x[i] - x[j])
                D[i,i] = D[i,i] - D[i,j]
    return D

def PolynomialDerivativeMatrix(x,k):
    """
    PolynomialDerivativeMatrix(): Assemble the Polynomial ``k``-th Derivative Matrix using the matrix recursion. This algorithm is generic for every types of polynomials.

    Syntax:
        ``D = PolynomialDerivativeMatrix(x,k)``

    Input:
        * ``x`` = (1d-array,float) set of points on which to evaluate the derivative matrix
        * ``k`` = derivative order

    Output:
        * ``D`` = ``k``-th derivative matrix

    Notes:
        Algorithm (38) taken from :cite:`Kopriva2009`
    """
    N = x.shape[0]
    w = BarycentricWeights(x)
    D = np.zeros((N,N,k))
    D[:,:,0] = FirstPolynomialDerivativeMatrix(x)
    if ( k == 1 ): return D[:,:,k-1]
    for m in range(2,k+1):
        for i in range(0,N):
            for j in range(0,N):
                if ( j != i ):
                    D[i,j,m-1] = m/(x[i]-x[j]) * ( w[j]/w[i] * D[i,i,m-2] - D[i,j,m-2])
                    D[i,i,m-1] = D[i,i,m-1] - D[i,j,m-1]
    return D[:,:,k-1]

def BarycentricWeights(x):
    """
    BarycentricWeights(): Returns a 1-d array of weights for Lagrange Interpolation

    Syntax:
        ``w = BarycentricWeights(x)``

    Input:
        * ``x`` = (1d-array,float) set of points

    Output:
        * ``w`` = (1d-array,float) set of barycentric weights

    Notes:
        Algorithm (30) from :cite:`Kopriva2009`
    """
    N = x.shape[0]
    w = np.zeros((N))
    for j in range(0,N):
        w[j] = 1.
    for j in range(1,N):
        for k in range(0,j):
            w[k] = w[k] * (x[k] - x[j])
            w[j] = w[j] * (x[j] - x[k])
    for j in range(0,N):
        w[j] = 1. / w[j]
    return w

def LagrangeInterpolationMatrix(x, w, xi):
    """
    LagrangeInterpolationMatrix(): constructs the Lagrange Interpolation Matrix from points ``x`` to points ``xi``

    Syntax:
        ``T = LagrangeInterpolationMatrix(x, w, xi)``

    Input:
        * ``x`` = (1d-array,float) set of ``N`` original points
        * ``w`` = (1d-array,float) set of ``N`` barycentric weights
        * ``xi`` = (1d-array,float) set of ``M`` interpolating points

    Output:
        * ``T`` = (2d-array(``MxN``),float) Lagrange Interpolation Matrix

    Notes:
        Algorithm (32) from :cite:`Kopriva2009`
    """
    N = x.shape[0]
    M = xi.shape[0]
    T = np.zeros((M,N))
    for k in range(0,M):
        rowHasMatch = False
        for j in range(0,N):
            T[k,j] = 0.
            if np.isclose(xi[k],x[j]):
                rowHasMatch = True
                T[k,j] = 1.
        if (rowHasMatch == False):
            s = 0.
            for j in range(0,N):
                t = w[j] / (xi[k] - x[j])
                T[k,j] = t
                s = s + t
            for j in range(0,N):
                T[k,j] = T[k,j] / s
    return T

def LagrangeInterpolate(x, f, xi):
    """
    LagrangeInterpolate(): Interpolate function values ``f`` from points ``x`` to points ``xi`` using Lagrange weights

    Syntax:
        ``fi = LagrangeInterpolate(x, f, xi)``

    Input:
        * ``x`` = (1d-array,float) set of ``N`` original points where ``f`` is evaluated
        * ``f`` = (1d-array/2d-array,float) set of ``N`` function values (if K functions are passed, the values are stored in a NxK matrix)
        * ``xi`` = (1d-array,float) set of ``M`` points where the function is interpolated

    Output:
        * ``fi`` = (1d-array,float) set of ``M`` function values (if K functions are passed, the values are stored in a MxK matrix)

    Notes:
        Modification of Algorithm (33) from :cite:`Kopriva2009`
    """
    # Obtain barycentric weights
    w = BarycentricWeights(x)
    # Generate the Interpolation matrix
    T = LagrangeInterpolationMatrix(x, w, xi)
    # Compute interpolated values
    fi = np.dot(T,f)
    return fi
