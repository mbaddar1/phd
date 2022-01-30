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

__revision__ = filter(str.isdigit, "$Revision: 101 $")

__author__ = "Daniele Bigoni"
__copyright__ = """Copyright 2012, Daniele Bigoni"""
__credits__ = ["Daniele Bigoni"]
__maintainer__ = "Daniele Bigoni"
__email__ = "dabi@dtu.dk"
__status__ = "Production"

import numpy as np
import itertools

def MultiIndex(d,N):
    """ Generates the multi index ordering for the construction of multidimensional Generalized Vandermonde matrices
    
    :param integer d: dimension of the simplex
    :param integer N: maximum value of the sum of the indices
    :returns: array containing the ordered multi-indices
    :rtype: 2d-array of integer
    
    .. code-block:: python
    
        >>> Misc.MultiIndex(2,3)
        array([[0, 0],
               [1, 0],
               [0, 1],
               [2, 0],
               [1, 1],
               [0, 2],
               [3, 0],
               [2, 1],
               [1, 2],
               [0, 3]])
    """
    
    from scipy.special import comb
    
    # Compute the size of the number of multi-index elements (Pascal's simplex)
    NIDX = 0
    for i in range(0,N+1):
        NIDX = NIDX + comb( i+(d-1),d-1,True)
    
    # Allocate memory
    IDX = np.zeros((NIDX,d),dtype=int)
    
    iIDX = 1 # index in the multi-index table on which the first n-th order is
    for n in range(1,N+1):
        IDX[iIDX,0] = n
        # Start recursion
        iIDX = __MultiIndexRec(IDX,d,iIDX+1,0)
    
    return IDX

def __MultiIndexRec(IDX,d,m,i):
    # Start splitting
    mFirstSplit = m-1
    mLastSplit = m-1
    mNew = m
    if (i+1 < d):
        while (IDX[mLastSplit,i] > 1):
            IDX[mNew,:i] = IDX[mLastSplit,:i]
            IDX[mNew,i] = IDX[mLastSplit,i]-1
            IDX[mNew,i+1] = IDX[mLastSplit,i+1]+1
            mLastSplit = mNew
            mNew = mNew + 1
            # Call recursion on sub set
            mNew = __MultiIndexRec(IDX,d,mNew,i+1)
        # Move
        IDX[mNew,:i] = IDX[mFirstSplit,:i]
        IDX[mNew,i+1] = IDX[mFirstSplit,i]
        mNew = mNew + 1
        # Call recursion on sub set
        mNew = __MultiIndexRec(IDX,d,mNew,i+1)
    return mNew

def machineEpsilon(func=float):
    """ Returns the abolute machine precision for the type passed as argument
    
    :param dtype func: type
    
    :returns: absolute machine precision
    :rtype: float
    
    .. code-block:: python
        
        >>> Misc.machineEpsilon(np.float64)
        2.2204460492503131e-16
        >>> Misc.machineEpsilon(np.float128)
        1.084202172485504434e-19
    
    """
    machine_epsilon = func(1)
    while func(1)+func(machine_epsilon) != func(1):
        machine_epsilon_last = machine_epsilon
        machine_epsilon = func(machine_epsilon) / func(2)
    return machine_epsilon_last

def compare(x,y,tol):
    """ Compares two iterable objects up to a certain tolerance
    
    :param x,y: values to be compared
    :type x,y: iterable objects of floats
    :param float tol: tolerance to be used
    :return: -1 if ``(x-y) < tol``, 1 if ``(x-y) > tol``, 0 otherwise
    :rtype: integer
    
    .. code-block:: python
        
        >>> eps2 = 2.*Misc.machineEpsilon(np.float64)
        >>> Misc.compare(np.array([2.]),np.array([2.+0.5*eps2]),eps2)
        0
        >>> Misc.compare(np.array([2.]),np.array([2.+2.*eps2]),eps2)
        -1
    """
    for x_i,y_i in zip(x,y):
        d = x_i-y_i
        if (d < -tol):
            return -1
        elif (d > tol):
            return 1
    return 0

def almostEqual(x,y,tol):
    """ Check equality of two arrays objects up to a certain tolerance
    
    :param x,y: values to be compared
    :type x,y: numpy.ndarray objects of floats
    :param float tol: tolerance to be used
    :return: ``true`` if equal, ``false`` otherwise
    :rtype: bool
    
    .. code-block:: python
        
        >>> eps2 = 2.*Misc.machineEpsilon(np.float64)
        >>> Misc.almostEqual(np.array([2.]),np.array([2.+0.5*eps2]),eps2)
        True
        >>> Misc.almostEqual(np.array([2.]),np.array([2.+2.*eps2]),eps2)
        False
    """
    return np.all(np.abs(x-y) <= 0.5 * tol * (np.abs(x) + np.abs(y)))

def almostEqualList(xArray,y,tol):
    """ Check equality of a list of floats against an iterable value up to certain tolerance
    
    :param xArray: values to be compared to ``y``
    :type xArray: 2d-array of floats
    :param y: values to be compared to
    :type y: iterable objects of floats
    :param float tol: tolerance to be used
    
    :return: array of booleans containing true where equal, false elsewhere.
    :rtype: 1d-array of bool
    
    Syntax:
        ``b = almostEqualList(xArray,y,tol)``
    
    .. code-block:: python
        
        >>> eps2 = 2.*Misc.machineEpsilon(np.float64)
        >>> X = np.random.rand(4,2)
        >>> Misc.almostEqualList(X,X[1,:],eps2)
        array([False,  True, False, False], dtype=bool)
    """
    if type(y) != np.ndarray and (np.dtype(y) == float and len(xArray.shape) == 1):
        y = np.array([y])
        xArray = xArray.reshape((len(xArray),1))
        
    if xArray.shape[1] == len(y):
        out = np.all(np.abs(xArray-y[np.newaxis,:]) <= 0.5 * tol * (np.abs(xArray) + np.abs(y)[np.newaxis,:]), axis=1)
        # out = np.zeros(xArray.shape[0],dtype=bool)
        # for i in range(xArray.shape[0]):
        #     out[i] = (almostEqual(xArray[i,:],y,tol))
        return out
    else:
        print("Dimension error!")
        return

def binary_search(X, val, lo, hi, tol, perm=None):
    """ Search for the minimum X bigger than val
    
    :param X: values ordered by row according to the ``compare`` function
    :type X: 2d-array of floats
    :param val: value to be compared to
    :type val: 1d-array of floats
    :param integer lo,hi: staring and ending indices
    :param float tol: tolerance to be used
    :param perm: possible permutation to be used prior to the search (optional)
    :type perm: 1d-array of integers
    :return: index pointing to the maximum X smaller than val. If ``perm`` is provided, ``perm[idx]`` points to the maximum X smaller than val
    :rtype: integer
    
    .. code-block:: python
        
        >>> X = np.arange(1,5).reshape((4,1))
        >>> X
        array([[1],
               [2],
               [3],
               [4]])
        >>> Misc.binary_search(X,np.array([2.5]),0,4,eps2)
        >>> idx = Misc.binary_search(X,np.array([2.5]),0,4,eps2)
        >>> idx
        2
        >>> X[idx,:]
        array([3])
    """
    if perm == None:
        perm = np.arange(lo,hi)
    
    while lo+1 < hi:
        mid = (lo+hi)//2
        midval = X[perm[mid],]
        comp = compare(midval,val,tol)
        if comp == -1:
            lo = mid
        elif comp == 1: 
            hi = mid
        elif comp == 0:
            return mid
    
    return hi

def argsort_insertion(X,tol,start_idx=1,end_idx=None):
    """ Implements the insertion sort with ``binary_search``. Returns permutation indices.
    
    :param X: values ordered by row according to the ``compare`` function
    :type X: 2d-array of floats
    :param float tol: tolerance to be used
    :param int start_idx,end_idx: starting and ending indices for the ordering (optional)
    :return: permutation indices
    :rtype: 1d-array of integers
    
    .. code-block:: python
        
        >>> X = np.random.rand(5,2)
        >>> X
        array([[ 0.56865133,  0.18490129],
               [ 0.01411459,  0.46076606],
               [ 0.64384365,  0.24998971],
               [ 0.47840414,  0.32554137],
               [ 0.12961966,  0.43712056]])
        >>> perm = Misc.argsort_insertion(X,eps2)
        >>> X[perm,:]
        array([[ 0.01411459,  0.46076606],
               [ 0.12961966,  0.43712056],
               [ 0.47840414,  0.32554137],
               [ 0.56865133,  0.18490129],
               [ 0.64384365,  0.24998971]])
    """
    if end_idx == None:
        end_idx = X.shape[0]
    if start_idx < 1:
        start_idx = 1
    
    perm = range(0,end_idx)
    for i in range(start_idx,len(perm)):
        val = perm[i]
        
        # Binary search
        idx = binary_search(X,X[val,],-1,i+1,tol,perm=perm) # idx contains the index in perm of the first X > val
        
        perm[idx+1:i+1] = perm[idx:i]
        perm[idx] = val
        
    return perm

def findOverlapping(XF,X,tol):
    """ Finds overlapping points of ``XF`` on ``X`` grids of points. The two grids are ordered with respect to :py:func:`Misc.compare`.
    
    :param XF,X: values ordered by row according to the :py:func:`Misc.compare`.
    :type XF,X: 2d-array of floats
    :param float tol: tolerance to be used
    :return: true values for overlapping points of ``XF`` on ``X``, false for not overlapping points. Note: the overlapping return argument is a true-false indexing for ``XF``.
    :rtype: 1d-array of bool
    
    **Example**
    
    .. code-block:: python
        
        >>> XF
        array([[ -1.73205081e+00,   0.00000000e+00],
               [ -1.00000000e+00,  -1.00000000e+00],
               [ -1.00000000e+00,   0.00000000e+00],
               [ -1.00000000e+00,   1.00000000e+00],
               [  0.00000000e+00,  -1.73205081e+00],
               [  0.00000000e+00,  -1.00000000e+00],
               [  0.00000000e+00,   2.16406754e-16],
               [  0.00000000e+00,   1.00000000e+00],
               [  0.00000000e+00,   1.73205081e+00],
               [  1.00000000e+00,  -1.00000000e+00],
               [  1.00000000e+00,   0.00000000e+00],
               [  1.00000000e+00,   1.00000000e+00],
               [  1.73205081e+00,   0.00000000e+00]])
        >>> X
        array([[ -1.73205081e+00,   0.00000000e+00],
               [ -1.00000000e+00,  -1.00000000e+00],
               [ -1.00000000e+00,   0.00000000e+00],
               [ -1.00000000e+00,   1.00000000e+00],
               [  0.00000000e+00,  -1.00000000e+00],
               [  2.16406754e-16,   0.00000000e+00],
               [  0.00000000e+00,   1.00000000e+00],
               [  1.00000000e+00,  -1.00000000e+00],
               [  1.00000000e+00,   0.00000000e+00],
               [  1.00000000e+00,   1.00000000e+00],
               [  1.73205081e+00,   0.00000000e+00]])
        >>> tol = 2. * Misc.machineEpsilon()
        >>> bool_idx_over = Misc.findOverlapping(XF,X,tol)
        >>> XF[np.logical_not(bool_idx_over),:]
        array([[ 0.        , -1.73205081],
               [ 0.        ,  1.73205081]])
        
    
    """
    j_X = 0
    idxs_over = np.zeros(XF.shape[0],dtype=bool)
    for i in range(XF.shape[0]):
        if j_X == X.shape[0] or compare(XF[i,:],X[j_X,:],tol) != 0:
            idxs_over[i] = False
        else:
            idxs_over[i] = True
            j_X += 1
    
    return idxs_over

def unique_cuts(X,tol,retIdxs=False):
    """ Returns the unique values and a list of arrays of boolean indicating the positions of the unique values.
    If retIdx is true, then it returns the group of indices with the same values as a indicator function (true-false array)
    """
    if X.shape[0] == 0:
        if retIdxs:
            return (np.zeros((0,X.shape[1])),[])
        else:
            return np.zeros((0,X.shape[1]))
    
    uCuts = [X[0,:]]
    new_idx_list = np.zeros(X.shape[0], dtype=bool)
    new_idx_list[0] = True
    idxs = [ new_idx_list ]
    for i in range(1,X.shape[0]):
        is_eq_list = almostEqualList(np.asarray(uCuts),X[i,:],tol)
        if np.any(is_eq_list):
            (idx,) = np.where(is_eq_list)
            idxs[idx][i] = True
        else:
            uCuts.append(X[i,:])
            new_idx_list = np.zeros(X.shape[0], dtype=bool)
            new_idx_list[i] = True
            idxs.append( new_idx_list )
    
    if retIdxs:
        return (np.asarray(uCuts),idxs)
    else:
        return np.asarray(uCuts)

def powerset(iterable):
    """ Compute the power set of an iterable object.
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

class ExpandingArray:
    """
    ExpandingArray is used for the dynamic allocation of memory in applications where the total allocated memory needed cannot be predicted. Memory is preallocated with increases of 50% all the time data exceed the allocated memory.
    """
    
    __DEFAULT_ALLOC_INIT_DIM = 10   # default initial dimension for all the axis is nothing is given by the user
    __DEFAULT_MAX_INCREMENT = 10000 # default value in order to limit the increment of memory allocation
    
    __MAX_INCREMENT = []    # Max increment
    __ALLOC_DIMS = []       # Dimensions of the allocated np.array
    __DIMS = []             # Dimensions of the view with data on the allocated np.array (__DIMS <= __ALLOC_DIMS)
    
    __ARRAY = []            # Allocated array
    
    def __init__(self,initData,allocInitDim=None,dtype=np.float64,maxIncrement=None):
        """ Initialization of the Expanding Array.
        
        >>> EA = ExpandingArray(initData,[allocInitDim=None,[dtype=np.float64,[maxIncrement=None]]])
        
        :param ndarray initData: InitialData with which to be initially filled. This must provide the number of dimensions of the array
        :param 1darray-integer allocInitDim: Initial allocated dimension (optional)
        :param dtype dtype: type for the data that will be contained in ``EA`` (optional,default=np.float64)
        :param integer maxIncrement: upper limit for the allocation increment
        
        """
        self.__DIMS = np.array(initData.shape)
        
        self.__MAX_INCREMENT = maxIncrement or self.__DEFAULT_MAX_INCREMENT
        
        ''' Compute the allocation dimensions based on user's input '''
        if allocInitDim == None:
            allocInitDim = self.__DIMS.copy()
        
        for i in range(len(self.__DIMS)):
            while allocInitDim[i] < self.__DIMS[i] or allocInitDim[i] == 0:
                if allocInitDim[i] == 0:
                    allocInitDim[i] = self.__DEFAULT_ALLOC_INIT_DIM
                if allocInitDim[i] < self.__DIMS[i]:
                    allocInitDim[i] += min(allocInitDim[i]/2, self.__MAX_INCREMENT)
        
        ''' Allocate memory '''
        self.__ALLOC_DIMS = allocInitDim
        self.__ARRAY = np.zeros(self.__ALLOC_DIMS,dtype=dtype)
        
        ''' Set initData '''
        sliceIdxs = [slice(self.__DIMS[i]) for i in range(len(self.__DIMS))]
        self.__ARRAY[sliceIdxs] = initData
    
    def shape(self):
        """ Returns the shape of the data inside the array. Note that the allocated memory is always bigger or equal to ``shape()``.
        
        :returns: shape of the data
        :rtype: tuple of integer
        
        .. code-block:: python
        
            >>> import numpy as np
            >>> EA = Misc.ExpandingArray(np.random.rand(25,4))
            >>> EA.shape()
            (25, 4)
            >>> EA.getAllocArray().shape
            (25, 4)
            >>> EA.concatenate(np.random.rand(13,4))
            >>> EA.shape()
            (38, 4)
            >>> EA.getAllocArray().shape
            (45, 4)
        
        """
        return tuple(self.__DIMS)
    
    def getAllocArray(self):
        """ Return the allocated array.
        
        :returns: allocated array
        :rtype: ndarray
        
        .. code-block:: python
            
            >>> import numpy as np
            >>> EA = Misc.ExpandingArray(np.random.rand(25,4))
            >>> EA.shape()
            (25, 4)
            >>> EA.getAllocArray().shape
            (25, 4)
            >>> EA.concatenate(np.random.rand(13,4))
            >>> EA.shape()
            (38, 4)
            >>> EA.getAllocArray().shape
            (45, 4)
        
        """
        return self.__ARRAY
    
    def getDataArray(self):
        """ Get the view of the array with data.
        
        :returns: allocated array
        :rtype: ndarray
        
        .. code-block:: python
            
            >>> EA.shape()
            (38, 4)
            >>> EA.getAllocArray().shape
            (45, 4)
            >>> EA.getDataArray().shape
            (38, 4)
            
        """
        sliceIdxs = [slice(self.__DIMS[i]) for i in range(len(self.__DIMS))]
        return self.__ARRAY[sliceIdxs]
    
    def concatenate(self,X,axis=0):
        """ Concatenate data to the existing array. If needed the array is resized in the ``axis`` direction by a factor of 50%.
        
        :param ndarray X: data to be concatenated to the array. Note that ``X.shape[i]==EA.shape()[i]`` is required for all i!=axis
        :param integer axis: axis along which to concatenate the additional data (optional)
        
        .. code-block:: python
            
            >>> import numpy as np
            >>> EA = Misc.ExpandingArray(np.random.rand(25,4))
            >>> EA.shape()
            (25, 4)
            >>> EA.getAllocArray().shape
            (25, 4)
            >>> EA.concatenate(np.random.rand(13,4))
            >>> EA.shape()
            (38, 4)
            
        """
        if axis > len(self.__DIMS):
            print("Error: axis number exceed the number of dimensions")
            return
        
        ''' Check dimensions for remaining axis '''
        for i in range(len(self.__DIMS)):
            if i != axis:
                if X.shape[i] != self.shape()[i]:
                    print("Error: Dimensions of the input array are not consistent in the axis %d" % i)
                    return
        
        ''' Check whether allocated memory is enough '''
        needAlloc = False
        while self.__ALLOC_DIMS[axis] < self.__DIMS[axis] + X.shape[axis]:
            needAlloc = True
            ''' Increase the __ALLOC_DIMS '''
            self.__ALLOC_DIMS[axis] += min(self.__ALLOC_DIMS[axis]/2,self.__MAX_INCREMENT)
        
        ''' Reallocate memory and copy old data '''
        if needAlloc:
            ''' Allocate '''
            newArray = np.zeros(self.__ALLOC_DIMS)
            ''' Copy '''
            sliceIdxs = [slice(self.__DIMS[i]) for i in range(len(self.__DIMS))]
            newArray[sliceIdxs] = self.__ARRAY[sliceIdxs]
            self.__ARRAY = newArray
        
        ''' Concatenate new data '''
        sliceIdxs = []
        for i in range(len(self.__DIMS)):
            if i != axis:
                sliceIdxs.append(slice(self.__DIMS[i]))
            else:
                sliceIdxs.append(slice(self.__DIMS[i],self.__DIMS[i]+X.shape[i]))
        
        self.__ARRAY[sliceIdxs] = X
        self.__DIMS[axis] += X.shape[axis]
    
    def trim(self,N,axis=0):
        """ Trim the axis dimension of N elements. The allocated data is not reinitialized or deallocated. Only the dimensions of the view are redefined.
        
        :param integer N: number of elements to be removed along the ``axis`` dimension
        :param integer axis: axis along which to remove elements (optional)
        
        .. code-block:: python
            
            >>> EA = Misc.ExpandingArray(np.random.rand(4,2))
            >>> EA.getDataArray()
            array([[ 0.42129746,  0.76220921],
                   [ 0.9238783 ,  0.11256142],
                   [ 0.42031437,  0.87349243],
                   [ 0.83187297,  0.555708  ]])
            >>> EA.trim(2,axis=0)
            >>> EA.getDataArray()
            array([[ 0.42129746,  0.76220921],
                   [ 0.9238783 ,  0.11256142]])
            >>> EA.getAllocArray()
            array([[ 0.42129746,  0.76220921],
                   [ 0.9238783 ,  0.11256142],
                   [ 0.42031437,  0.87349243],
                   [ 0.83187297,  0.555708  ]])
        """
        self.__DIMS[axis] = max(self.__DIMS[axis]-N,0)
