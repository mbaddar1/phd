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

__all__ = ['LinearShapeFunction', 'SparseLinearShapeFunction',
           'LinearInterpolationMatrix', 'SparseLinearInterpolationMatrix']

def LinearShapeFunction(x,xm,xp,xi):
    """ Hat function used for linear interpolation
    
    :param array x: 1d original points
    :param float xm,xp: bounding points of the support of the shape function
    :param array xi: 1d interpolation points

    :returns array N: evaluation of the shape function on xi
    """
    N = np.zeros(len(xi))
    if x != xm: N += (xi - xm)/(x - xm) * ((xi >= xm)*(xi <= x)).astype(float)
    if x != xp: N += ((x - xi)/(xp - x) + 1.) * ((xi >= x)*(xi <= xp)).astype(float)
    return N

def SparseLinearShapeFunction(x,xm,xp,xi):
    """ Hat function used for linear interpolation. 
    Returns sparse indices for construction of scipy.sparse.coo_matrix.
    
    :param array x: 1d original points
    :param float xm,xp: bounding points of the support of the shape function
    :param array xi: 1d interpolation points

    :returns tuple (idxs,vals): List of indexes and evaluation of the shape function on xi
    """
    idxs = []
    vals = []
    # Get all xi == x
    bool_idxs = (xi == x)
    idxs.extend( np.where(bool_idxs)[0] )
    vals.extend( [1.]*sum(bool_idxs) )
    # If not left end
    if x != xm:
        bool_idxs = (xi >= xm)*(xi < x)
        idxs.extend( np.where(bool_idxs)[0] )
        vals.extend( (xi[bool_idxs] - xm)/(x - xm) )
    # If not right end
    if x != xp:
        bool_idxs = (xi > x)*(xi <= xp)
        idxs.extend( np.where(bool_idxs)[0] )
        vals.extend( ((x - xi[bool_idxs])/(xp - x) + 1.) )
    return (idxs,vals)

def LinearInterpolationMatrix(x, xi):
    """
    LinearInterpolationMatrix(): constructs the Linear Interpolation Matrix from points ``x`` to points ``xi``

    Syntax:
        ``T = LagrangeInterpolationMatrix(x, xi)``

    Input:
        * ``x`` = (1d-array,float) set of ``N`` original points
        * ``xi`` = (1d-array,float) set of ``M`` interpolating points

    Output:
        * ``T`` = (2d-array(``MxN``),float) Linear Interpolation Matrix

    """
    
    M = np.zeros((len(xi),len(x)))
    
    M[:,0] = LinearShapeFunction(x[0],x[0],x[1],xi)
    M[:,-1] = LinearShapeFunction(x[-1],x[-2],x[-1],xi)
    for i in range(1,len(x)-1):
        M[:,i] = LinearShapeFunction(x[i],x[i-1],x[i+1],xi)

    return M

def SparseLinearInterpolationMatrix(x,xi):
    """
    LinearInterpolationMatrix(): constructs the Linear Interpolation Matrix from points ``x`` to points ``xi``.
    Returns a scipy.sparse.coo_matrix

    Syntax:
        ``T = LagrangeInterpolationMatrix(x, xi)``

    Input:
        * ``x`` = (1d-array,float) set of ``N`` original points
        * ``xi`` = (1d-array,float) set of ``M`` interpolating points

    Output:
        * ``T`` = (scipy.sparse.coo_matrix(``MxN``),float) Linear Interpolation Matrix

    """
    
    rows = []
    cols = []
    vals = []
    
    (ii,vv) = SparseLinearShapeFunction(x[0],x[0],x[1],xi)
    rows.extend( ii )
    cols.extend( [0] * len(ii) )
    vals.extend( vv )

    (ii,vv) = SparseLinearShapeFunction(x[-1],x[-2],x[-1],xi)
    rows.extend( ii )
    cols.extend( [len(x)-1] * len(ii) )
    vals.extend( vv )

    for j in range(1,len(x)-1):
        (ii,vv) = SparseLinearShapeFunction(x[j],x[j-1],x[j+1],xi)
        rows.extend( ii )
        cols.extend( [j] * len(ii) )
        vals.extend( vv )

    M = scsp.coo_matrix( (np.asarray(vals), (np.asarray(rows),np.asarray(cols))), shape=( len(xi), len(x) ) )
    return M
