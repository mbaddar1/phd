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

import sys

# import progressbar

import numpy as np

import scipy.special

import itertools

from SpectralToolbox import Misc

class HSparseGrid:
    """ Heterogeneous Sparse Grid class
    """
    polys = []
    Ns = []
    tol = []
    sdout = []
    
    def __init__(self, polys, Ns, tol=None, sdout=sys.stderr):
        """ Constructor of Heterogeneous Sparse Grid object (this does not allocate the sparse grid)
        
        :param polys: orthogonal polynomials to be used as basis functions
        :type polys: list of :py:class:`Spectral1D.Poly1D`
        :param Ns: accuracy for each dimension. It can be a list of accuracies or a single accuracy, in which case uniform accuracy is assumed
        :type Ns: list of integers
        :param float tol: tolerance to be used when comparing points of the grid (optional, default=:py:func:`Misc.machineEpsilon()`)
        :param stream sdout: default output stream for the class (optional,default=``sys.stderr``)
        
        .. note:: one of the following must hold: len(polys)==len(Ns) or len(Ns)==1, in which case the same order is used for all the directions.
        
        **Example**
        
        >>> from SpectralToolbox import HeterogeneousSparseGrids as HSG
        >>> pH = Spectral1D.Poly1D(Spectral1D.HERMITEP_PROB,None)
        >>> pL = Spectral1D.Poly1D(Spectral1D.JACOBI,[0.0,0.0])
        >>> polys = [pH,pL]
        >>> Ns = [2,4]
        >>> sg = HSG.HSparseGrid(polys,Ns)
        """
        self.sdout = sdout
        self.tol = tol or Misc.machineEpsilon()
        
        # Check consistency of arguments
        if type(Ns) != int and len(polys) != len(Ns):
            print("Size of parameters not consistent: either type(Ns)==int or len(polys) == len(Ns)")
            raise NameError("Input Error.")
        
        self.polys = polys
        if type(Ns) == int:
            self.Ns = np.tile(Ns,len(self.polys))
        else:
            self.Ns = Ns
    
    def sparseGrid(self,heter=False):
        """ Generates the full and partial sparse grids
        
        :param bool heter: if :py:data:`Ns` is homogeneous, this parameter will force the output of the partial sparse grid as well
        
        :return: tuple :py:data:`(XF,WF,X)` containing:
            
            * :py:data:`XF`: full grid points
            * :py:data:`WF`: full grid weights
            * :py:data:`X`: partial grid points
        
        **Example**
        
        >>> (XF,W,X) = sg.sparseGrid()
        [SG] Sparse Grid Generation [============================================] 100%
        [SG] Sparse Grid Generation: 0.01s
        
        """
        d = len(self.polys)
        
        k = np.max(self.Ns)
        
        heterogeneous = np.any(self.Ns<k) or heter
        
        # Compute the number of multi-index elements
        idxs = Misc.MultiIndex(d,k)
        
        # # Initialize the progress bar
        # bar = progressbar.ProgressBar(maxval=len(idxs), 
        #                               widgets=[progressbar.Bar(
        #                               '=', '[SG] Sparse Grid Generation [', ']'), ' ', progressbar.Percentage()])
        # bar.start()
        
        # Generate list of nodes and weights for univariate formulas
        quadLists = []
        for i_d in range(d):
            qList = []
            for j_N in range(k+1):
                qList.append(self.polys[i_d].GaussQuadrature(j_N))
            quadLists.append(qList)
        
        '''
        Preallocation of memory for points and weights.
        The allocated memory is doubled all the times the allocated memory if filled.
        A counter is used to keep track of the last data position.
        Starting size is set to 10.
        '''
        X_full_exp = Misc.ExpandingArray(np.zeros((0,d)),maxIncrement=min(k**d,10**5))
        if heterogeneous:
            X_part_exp = Misc.ExpandingArray(np.zeros((0,d)),maxIncrement=min(k**d,10**5))
        
        W_exp = Misc.ExpandingArray(np.zeros(0),maxIncrement=min(k**d,10**5))
        
        for (counter,idx) in enumerate(idxs):
            # bar.update(counter)
            
            bq = (-1)**(d+k-np.sum(idx+1)) * scipy.special.binom(d-1,np.sum(idx+1)-k-1)
            xs = []
            wKron = 1.
            for idx_i in range(d):
                (x,w) = quadLists[idx_i][idx[idx_i]]
                xs.append(x)
                wKron = np.kron(wKron,w)
            
            xKron = np.asarray(list(itertools.product(*xs)))
            lenX_full = X_full_exp.shape()[0]
            X_full_exp.concatenate(xKron,axis=0)
            if heterogeneous and np.all(idx<=self.Ns):
                lenX_part = X_part_exp.shape()[0]
                X_part_exp.concatenate(xKron,axis=0)
            W_exp.concatenate(bq * wKron, axis=0)
            
            # Sort
            perm_full = Misc.argsort_insertion(X_full_exp.getAllocArray(),self.tol,start_idx=lenX_full,end_idx=X_full_exp.shape()[0])
            X_full_exp.getDataArray()[:] = X_full_exp.getAllocArray()[perm_full,:].copy()
            if heterogeneous and np.all(idx<=self.Ns):
                perm_part = Misc.argsort_insertion(X_part_exp.getAllocArray(),self.tol,start_idx=lenX_part,end_idx=X_part_exp.shape()[0])
                X_part_exp.getDataArray()[:] = X_part_exp.getAllocArray()[perm_part,].copy()
            W_exp.getDataArray()[:] = W_exp.getAllocArray()[perm_full].copy()
            
            # Remove repetitions and add weights
            keep = [0]
            lastkeep = 0
            for i in range(1,X_full_exp.shape()[0]):
                if Misc.compare(X_full_exp.getAllocArray()[i,],X_full_exp.getAllocArray()[i-1,],self.tol) == 0:
                    W_exp.getAllocArray()[lastkeep] += W_exp.getAllocArray()[i]
                else:
                    lastkeep = i
                    keep.append(i)
            
            X_full_exp.getAllocArray()[:len(keep),] = X_full_exp.getAllocArray()[keep,]
            X_full_exp.trim(X_full_exp.shape()[0]-len(keep),axis=0)
            W_exp.getAllocArray()[:len(keep)] = W_exp.getAllocArray()[keep]
            W_exp.trim(W_exp.shape()[0]-len(keep),axis=0)
            
            if heterogeneous and np.all(idx<=self.Ns):
                keep = [0]
                lastkeep = 0
                for i in range(1,X_part_exp.shape()[0]):
                    if Misc.compare(X_part_exp.getAllocArray()[i,],X_part_exp.getAllocArray()[i-1,],self.tol) != 0:
                        lastkeep = i
                        keep.append(i)
                
                X_part_exp.getAllocArray()[:len(keep),] = X_part_exp.getAllocArray()[keep,]
                X_part_exp.trim(X_part_exp.shape()[0]-len(keep),axis=0)
        
        # bar.finish()
        # print("[SG] Sparse Grid Generation: %.2fs" % (bar.last_update_time-bar.start_time))
        
        if heterogeneous:
            return (X_full_exp.getDataArray(), W_exp.getDataArray(), X_part_exp.getDataArray())
        else:
            return (X_full_exp.getDataArray(), W_exp.getDataArray(), X_full_exp.getDataArray())
    
    
    def sparseGridInterp(self,X,fX,XF):
        """ Interpolate values of the Sparse Grid using 1D Polynomial interpolation along cuts.
        
        :param X: partial grid on which a function has been evaluated
        :type X: 2d-array of floats
        :param fX: values for the points in the partial grid
        :type fX: 1d-array of floats
        :param XF: full grid on which to interpolate
        :type XF: 2d-array of floats
        :return: :py:data:`fXF` the interpolated values on the full grid
        :rtype: 1d-array of floats
        
        ..note:: The partial and full grid must be overlapping
        
        **Example**
        
        >>> fXF = sg.sparseGridInterp(X,fX,XF)
        [SG] Sparse Grid Interpolation [=========================================] 100%
        [SG] Sparse Grid Interpolation: 0.00s
        
        """
        tol = 2. * Misc.machineEpsilon()
        
        if X.shape[1] != XF.shape[1]:
            print("[SG] Sparse Grid Interpolation: Error! Dimensions of full and partial grid are not consistent\n")
            return
        if X.shape[0] != fX.shape[0]:
            print("[SG] Sparse Grid Interpolation: Error! Dimensions of partial nodes and value are not consistent\n")
        
        d = X.shape[1];
        
        # Points on the full grid and on the actual interpolated grid, projected on d-1 dimensions
        XF_mD = np.zeros((XF.shape[0],XF.shape[1]-1))
        
        fXF = np.zeros(XF.shape[0])
        
        ''' Find not overlapping points (rules are ordered) '''
        bool_idxs_over = Misc.findOverlapping(XF,X,tol)
        
        if np.sum(bool_idxs_over) != X.shape[0]:
            print("[SG] Sparse Grid Interpolation: Error! The selected grids are not overlapping. Functionality not implemented yet\n")
        
        ''' insert values of partial grid in correct position in the full grid '''
        fXF[bool_idxs_over] = fX
        
        counter = 0       # Counts the number of interpolated values
        N_not_over = np.sum(np.logical_not(bool_idxs_over))
        # bar = progressbar.ProgressBar(maxval=N_not_over+1, 
        #                               widgets=[progressbar.Bar(
        #                               '=', '[SG] Sparse Grid Interpolation [', ']'), ' ', progressbar.Percentage()])
        # bar.start()
        # bar.update(counter)
        if N_not_over > 0:
            for i_d in range(d):
                XF_mD[:,:i_d] = XF[:,:i_d]
                XF_mD[:,i_d:] = XF[:,i_d+1:]
                ''' find the unique coordinates of cuts '''
                vals = Misc.unique_cuts(XF_mD[np.logical_not(bool_idxs_over)])
                ''' check for each value if the quadrature is up to the maximum accuracy '''
                for i_val, val in enumerate(vals):
                    
                    bool_idxs_not_over_cut = Misc.almostEqualList(XF_mD[np.logical_not(bool_idxs_over),],val,self.tol)
                    bool_idxs_over_cut = Misc.almostEqualList(XF_mD[bool_idxs_over,],val,self.tol)
                    
                    if np.sum(bool_idxs_not_over_cut) != 0 and np.sum(bool_idxs_over_cut) >= self.Ns[i_d]+1:
                        ''' The quadrature is not up to the maximum accuracy for this cut '''
                        ''' Apply 1D interpolation '''
                        X_base = XF[bool_idxs_over,i_d][bool_idxs_over_cut]
                        fXF_base = fXF[bool_idxs_over][bool_idxs_over_cut]
                        XF_interp = XF[np.logical_not(bool_idxs_over),i_d][bool_idxs_not_over_cut]
                        
                        fXF_interp = self.polys[i_d].PolyInterp(X_base,fXF_base,XF_interp,self.Ns[i_d])
                        
                        fXF[np.where(np.logical_not(bool_idxs_over))[0][bool_idxs_not_over_cut]] = fXF_interp
    
                        ''' Add new data for following interpolations '''
                        bool_idxs_over[np.where(np.logical_not(bool_idxs_over))[0][bool_idxs_not_over_cut]] = True
    
                        ''' Update the counter '''
                        counter += np.sum(bool_idxs_not_over_cut)
    
                        # bar.update(counter)
        
        # bar.finish()
        # print("[SG] Sparse Grid Interpolation: %.2fs\n" % (bar.last_update_time-bar.start_time))
        
        if counter != N_not_over:
            print("Error: incomplete interpolation. The target function has not been computed for some non overlapping points\n")
        
        return fXF
