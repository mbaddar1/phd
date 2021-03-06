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
import logging
import itertools

import numpy as np
from scipy.special import comb

import SpectralToolbox.Spectral1D

def MultiIndex(d,N):
    """
    MultiIndex(): generates the multi index ordering for the construction of multidimensional Generalized Vandermonde matrices
    
    Syntax:
        ``IDX = MultiIndex(d,N)``
    
    Input:
        * ``d`` = (int) dimension of the simplex
        * ``N`` = (int) the maximum value of the sum of the indeces
    
    OUTPUT:
        * ``IDX`` = (2d-array,int) array containing the ordered multi indeces        
    """
    
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

class PolyND:
    
    logger = logging.getLogger(__name__)
    logger.propagate = False
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    def __init__(self, polys):
        """
        Initialization of the N-dimensional Polynomial instance
        
        Syntax:
            ``p = PolyND(polys)``
        
        Input:
            * ``polys`` = (list,Spectral1D.Poly1D) list of polynomial instances of the class ``Spectral1D.Poly1D``
        
        .. seealso:: Spectral1D.Poly1D
        
        """
        self.polys = polys
        self.DIM = len(self.polys)

    def Quadrature(self, Ns, quadTypes=None, norm=True, warnings=True):
        """
        GaussQuadrature(): computes the tensor product of the Guass Points and weights
        
        Syntax:
            ``(x,w) = GaussQuadrature(Ns, [quadTypes=None], [norm=True],[warnings=True])``
        
        Input:
            * ``Ns`` = (list,int) n-dimensional list with the order of approximation of each polynomial
            * ``quadTypes`` = (list,``Spectral1D.AVAIL_QUADPOINTS``) n-dimensional list of quadrature point types chosen among Gauss, Gauss-Radau, Gauss-Lobatto (using the definition in ``Spectral1D``). If None, Gauss points will be generated by default
            * ``norm`` = (optional,boolean) whether the weights will be normalized or not
            * ``warnings`` = (optional,boolean) set whether to ask for confirmation when it is required to allocate more then 100Mb of memory
        
        Output:
            * ``x`` = tensor product of the collocation points
            * ``w`` = tensor product of the weights
        
        .. warning:: The lengths of ``Ns`` has to be conform to the number of polynomials with which you have instantiated ``PolyND``
        
        """
        
        # Memory allocation for which the user will get a warning message (Mb)
        warningMem = 100.0
        
        if self.DIM != len(Ns) :
            print("The number of elements in Ns is not consistent")
            return

        if quadTypes == None:
            quadTypes = [Spectral1D.GAUSS for i in range(self.DIM)]

        if self.DIM != len(quadTypes):
            print("The number of elements in quadTypes is not consistent")
            return
        
        # #######################
        # # Estimate memory usage
        # Ncoll = np.prod(np.asarray(Ns) + 1)
        # SDOUBLE = sys.getsizeof(0.0)
        # SARRAY = sys.getsizeof(np.asarray([]))
        # xMem = self.DIM * Ncoll * SDOUBLE + SARRAY
        # wMem = Ncoll * SDOUBLE + SARRAY
        # totMem = xMem + wMem
        # # Print out information
        # self.logger.debug("\n" +
        #                   "Memory usage information:\n" +
        #                   "\t X Points: %10.2f Mb \n" % (xMem * 1e-6) +
        #                   "\t Weights: %10.2f Mb \n" % (wMem * 1e-6) +
        #                   "Total Memory: %10.2f Mb \n" % (totMem * 1e-6) +
        #                   "N of collocation points: %d " % (Ncoll))
        
        # if warnings and totMem * 1e-6 > warningMem:
        #     opt = 'a'
        #     while (opt != 'c' and opt != 'b' and opt != 'q'):
        #         self.logger.warning("\n" +
        #                             "The total memory that will be allocated exceed %10.2fMb. Chose one , of the following options:\n" % (warningMem) +
        #                             "\t [c]: continue\n" +
        #                             "\t [q]: exit" )
        #         opt = sys.stdin.read(1)
        #     if (opt ==  'q'):
        #         return        
        
        x,w = self.polys[0].Quadrature(Ns[0],quadType=quadTypes[0],norm=norm)
        wKron = w
        xs = [x]
        for i in range(1,self.DIM):
            x,w = self.polys[i].Quadrature(Ns[i],quadType=quadTypes[i],norm=norm)
            wKron = np.kron(wKron, w)
            xs.append(x)
        xKron = np.asarray(list(itertools.product(*xs)))
        
        return (xKron, wKron)
    
    def GaussQuadrature(self, Ns, norm=True, warnings=True):
        """
        GaussQuadrature(): computes the tensor product of the Guass Points and weights
        
        Syntax:
            ``(x,w) = GaussQuadrature(Ns, [norm=True],[warnings=True])``
        
        Input:
            * ``Ns`` = (list,int) n-dimensional list with the order of approximation of each polynomial
            * ``norm`` = (optional,boolean) whether the weights will be normalized or not
            * ``warnings`` = (optional,boolean) set whether to ask for confirmation when it is required to allocate more then 100Mb of memory
        
        Output:
            * ``x`` = tensor product of the collocation points
            * ``w`` = tensor product of the weights
        
        .. warning:: The lengths of ``Ns`` has to be conform to the number of polynomials with which you have instantiated ``PolyND``
        
        """
        
        # Memory allocation for which the user will get a warning message (Mb)
        warningMem = 100.0
        
        if self.DIM != len(Ns) :
            print("The number of elements in Ns is not consistent")
            return

        # #######################
        # # Estimate memory usage
        # Ncoll = np.prod(np.asarray(Ns) + 1)
        # SDOUBLE = sys.getsizeof(0.0)
        # SARRAY = sys.getsizeof(np.asarray([]))
        # xMem = self.DIM * Ncoll * SDOUBLE + SARRAY
        # wMem = Ncoll * SDOUBLE + SARRAY
        # totMem = xMem + wMem
        # # Print out information
        # self.logger.debug("\n" +
        #                   "Memory usage information:\n" +
        #                   "\t X Points: %10.2f Mb \n" % (xMem * 1e-6) +
        #                   "\t Weights: %10.2f Mb \n" % (wMem * 1e-6) +
        #                   "Total Memory: %10.2f Mb \n" % (totMem * 1e-6) +
        #                   "N of collocation points: %d " % (Ncoll))
        
        # if warnings and totMem * 1e-6 > warningMem:
        #     opt = 'a'
        #     while (opt != 'c' and opt != 'b' and opt != 'q'):
        #         self.logger.warning("\n" +
        #                             "The total memory that will be allocated exceed %10.2fMb. Chose one , of the following options:\n" % (warningMem) +
        #                             "\t [c]: continue\n" +
        #                             "\t [q]: exit" )
        #         opt = sys.stdin.read(1)
        #     if (opt ==  'q'):
        #         return
                
        x,w = self.polys[0].GaussQuadrature(Ns[0],norm=norm)
        wKron = w
        xs = [x]
        for i in range(1,self.DIM):
            x,w = self.polys[i].GaussQuadrature(Ns[i],norm=norm)
            wKron = np.kron(wKron, w)
            xs.append(x)
        xKron = np.asarray(list(itertools.product(*xs)))
        
        return (xKron, wKron)
    
    def GaussLobattoQuadrature(self, Ns, norm=True, warnings=True):
        """
        GaussLobattoQuadrature(): computes the tensor product of the Guass Lobatto Points and weights
        
        Syntax:
            ``(x,w) = GaussLobattoQuadrature(Ns,[norm=True],[warnings=True])``
        
        Input:
            * ``Ns`` = (list,int) n-dimensional list with the order of approximation of each polynomial
            * ``norm`` = (optional,boolean) whether the weights will be normalized or not
            * ``warnings`` = (optional,boolean) set whether to ask for confirmation when it is required to allocate more then 100Mb of memory
        
        Output:
            * ``x`` = tensor product of the collocation points
            * ``w`` = tensor product of the weights
        
        .. warning:: The lengths of ``Ns`` has to be conform to the number of polynomials with which you have instantiated ``PolyND``
        
        """
        
        # Memory allocation for which the user will get a warning message (Mb)
        warningMem = 100.0
        
        if self.DIM != len(Ns) :
            print("The number of elements in Ns is not consistent")
            return
        
        # #######################
        # # Estimate memory usage
        # Ncoll = np.prod(np.asarray(Ns) + 1)
        # SDOUBLE = sys.getsizeof(0.0)
        # SARRAY = sys.getsizeof(np.asarray([]))
        # xMem = self.DIM * Ncoll * SDOUBLE + SARRAY
        # wMem = Ncoll * SDOUBLE + SARRAY
        # totMem = xMem + wMem
        # # Print out information
        # self.logger.debug("\n" +
        #                   "Memory usage information:\n" +
        #                   "\t X Points: %10.2f Mb \n" % (xMem * 1e-6) +
        #                   "\t Weights: %10.2f Mb \n" % (wMem * 1e-6) +
        #                   "Total Memory: %10.2f Mb \n" % (totMem * 1e-6) +
        #                   "N of collocation points: %d " % (Ncoll))
        
        # if warnings and totMem * 1e-6 > warningMem:
        #     opt = 'a'
        #     while (opt != 'c' and opt != 'b' and opt != 'q'):
        #         self.logger.warning("\n" +
        #                             "The total memory that will be allocated exceed %10.2fMb. Chose one , of the following options:\n" % (warningMem) +
        #                             "\t [c]: continue\n" +
        #                             "\t [q]: exit" )
        #         opt = sys.stdin.read(1)
        #     if (opt ==  'q'):
        #         return

        x,w = self.polys[0].GaussLobattoQuadrature(Ns[0])
        wKron = w
        xs = [x]
        for i in range(1,self.DIM):
            x,w = self.polys[i].GaussLobattoQuadrature(Ns[i])
            wKron = np.kron(wKron, w)
            xs.append(x)
        xKron = np.asarray(list(itertools.product(*xs)))
        
        return (xKron, wKron)
    
    def GradVandermonde(self,rs,Ns,ks=None,norms=None,usekron=True,output=True,warnings=True):
        """
        GradVandermonde(): initialize the tensor product of the k-th gradient of the modal basis.
        
        Syntax:
            ``V = GradVandermonde(r,N,k,[norms=None],[usekron=True],[output=True],[warnings=True])``
        
        Input:
            * ``rs`` = (list of 1d-array,float) ``n``-dimensional list of set of points on which to evaluate the polynomials (by default they are not the kron product of the points. See ``usekron`` option)
            * ``Ns`` = (list,int) n-dimensional list with the maximum orders of approximation of each polynomial
            * ``ks`` = (list,int) n-dimensional list with derivative orders [default=0]
            * ``norms`` = (default=None,list,boolean) n-dimensional list of boolean, True -> orthonormal, False -> orthogonal, None -> all orthonormal
            * ``usekron`` = (optional,boolean) set whether to apply the kron product of the single dimensional Vandermonde matrices or to multiply column-wise. kron(rs) and usekron==False is equal to rs and usekron==True
            * ``output`` = (optional,boolean) set whether to print out information about memory allocation
            * ``warnings`` = (optional,boolean) set whether to ask for confirmation when it is required to allocate more then 100Mb of memory
        
        OUTPUT:
            * ``V`` = Tensor product of the Generalized Vandermonde matrices
        
        .. warning:: The lengths of ``Ns`` , ``rs`` , ``ks`` , ``norms`` has to be conform to the number of polynomials with which you have instantiated ``PolyND``
        
        """
        
        # Memory allocation for which the user will get a warning message (Mb)
        warningMem = 100.0

        if norms == None:
            norms = [True] * self.DIM
        if ks == None:
            ks = [0]*self.DIM
        
        if usekron:
            if (self.DIM != len(rs) or self.DIM != len(Ns) or self.DIM != len(ks) or self.DIM != len(norms)) :
                print("The number of elements in rs, Ns, ks and norms is not consistent")
                return
        else:
            if ( not (type(rs) == np.ndarray and rs.shape[1] == self.DIM) or self.DIM != len(Ns) or self.DIM != len(ks) or self.DIM != len(norms)) :
                print("The number of elements in rs, Ns, ks and norms is not consistent")
                return
                
        # #######################
        # # Estimate memory usage
        # if usekron:
        #     Ncolls = np.zeros(self.DIM)
        #     for i in range(0,self.DIM):
        #         Ncolls[i] = len(rs[i])
        #     SDOUBLE = sys.getsizeof(0.0)
        #     SARRAY = sys.getsizeof(np.asarray([]))
        #     VMem = np.prod((np.asarray(Ns)+1) * Ncolls) * SDOUBLE + SARRAY
        #     totMem = VMem
        # else:
        #     Ncolls = rs.shape[0]
        #     SDOUBLE = sys.getsizeof(0.0)
        #     SARRAY = sys.getsizeof(np.asarray([]))
        #     VMem = np.prod(np.asarray(Ns)+1) * Ncolls * SDOUBLE + SARRAY
        #     totMem = VMem
        # # Print out information
        # self.logger.debug("\n" +
        #                   "Memory usage information:\n" +
        #                   "\t Tensor Basis: %10.2f Mb \n" % (VMem * 1e-6) +
        #                   "Total Memory: %10.2f Mb " % (totMem * 1e-6) )

        # if output and warnings and totMem * 1e-6 > warningMem:
        #     opt = 'a'
        #     while (opt != 'c' and opt != 'b' and opt != 'q'):
        #         self.logger.warning("%n" +
        #                             "The total memory that will be allocated exceed %10.2fMb. Chose one , of the following options:\n" % (warningMem) +
        #                             "\t [c]: continue\n" +
        #                             "\t [q]: exit")
        #         opt = sys.stdin.read(1)
        #     if (opt ==  'q'):
        #         return

        if usekron:
            VKron = self.polys[0].GradVandermonde(rs[0],Ns[0],ks[0],norms[0])
            for i in range(1,self.DIM):
                VKron = np.kron(VKron, self.polys[i].GradVandermonde(rs[i],Ns[i],ks[i],norms[i]))
        else:
            VKron = np.ones((rs.shape[0],1))
            for i in range(0,self.DIM):
                VKronNew = np.zeros((VKron.shape[0],VKron.shape[1] * (Ns[i]+1)))
                V = self.polys[i].GradVandermonde(rs[:,i],Ns[i],ks[i],norms[i])
                for col in range(0,VKron.shape[1]):
                    VKronNew[:,col*(Ns[i]+1):(col+1)*(Ns[i]+1)] = np.tile(VKron[:,col],(Ns[i]+1,1)).T * V
                VKron = VKronNew
        
        return VKron

    def GradVandermondePascalSimplex(self,rs,N,ks=None,norms=None,usekron=True,output=True,warnings=True):
        """
        GradVandermondePascalSimplex(): initialize k-th gradient of the modal basis up to the total order N
        
        Syntax:
            ``V = GradVandermonde(r,N,k,[norms=None],[output=True],[warnings=True])``
        
        Input:
            * ``rs`` = (list of 1d-array,float) ``n``-dimensional list of set of points on which to evaluate the polynomials (by default they are not the kron product of the points. See ``usekron`` option)
            * ``N`` = (int) the maximum orders of the polynomial basis
            * ``ks`` = (list,int) n-dimensional list with derivative orders [default=0]
            * ``norms`` = (default=None,list,boolean) n-dimensional list of boolean, True -> orthonormal, False -> orthogonal, None -> all orthonormal
            * ``usekron`` = (optional,boolean) set whether to apply the kron product of the single dimensional Vandermonde matrices or to multiply column-wise. kron(rs) and usekron==False is equal to rs and usekron==True
            * ``output`` = (optional,boolean) set whether to print out information about memory allocation
            * ``warnings`` = (optional,boolean) set whether to ask for confirmation when it is required to allocate more then 100Mb of memory
        
        OUTPUT:
            * ``V`` = Generalized Vandermonde matrix up to the N-th order
        
        .. warning:: The lengths of ``rs`` , ``ks`` , ``norms`` has to be conform to the number of polynomials with which you have instantiated ``PolyND``
        
        """
        
        # Memory allocation for which the user will get a warning message (Mb)
        warningMem = 100.0

        if norms == None:
            norms = [True] * self.DIM
        if ks == None:
            ks = [0]*self.DIM
        
        if usekron:
            if (self.DIM != len(rs) or self.DIM != len(ks) or self.DIM != len(norms)) :
                print("The number of elements in rs, ks and norms is not consistent")
                return
        else:
            if ( not (type(rs) == np.ndarray and rs.shape[1] == self.DIM) or self.DIM != len(ks) or self.DIM != len(norms)) :
                print("The number of elements in rs, ks and norms is not consistent")
                return
        
        # # Estimate memory usage
        # if usekron:
        #     Ncolls = 1
        #     for i in range(0,self.DIM):
        #         Ncolls = Ncolls * len(rs[i])
        #     # Number of basis computed using the pascal simplex formula
        #     Nbasis = 0
        #     for i in range(0,N+1):
        #         Nbasis = Nbasis + comb( i+(self.DIM-1),self.DIM-1,True)
        #     SDOUBLE = sys.getsizeof(0.0)
        #     SARRAY = sys.getsizeof(np.asarray([]))
        #     VMem = Nbasis * Ncolls * SDOUBLE + SARRAY
        #     totMem = VMem
        # else:
        #     Ncolls = rs.shape[0]
        #     # Number of basis computed using the pascal simplex formula
        #     Nbasis = 0
        #     for i in range(0,N+1):
        #         Nbasis = Nbasis + comb( i+(self.DIM-1),self.DIM-1,True)
        #     SDOUBLE = sys.getsizeof(0.0)
        #     SARRAY = sys.getsizeof(np.asarray([]))
        #     VMem = Nbasis * Ncolls * SDOUBLE + SARRAY
        #     totMem = VMem
        # # Print out information
        # self.logger.debug("\n" +
        #                   "Memory usage information:\n" +
        #                   "\t Tensor Basis: %10.2f Mb \n" % (VMem * 1e-6) +
        #                   "Total Memory: %10.2f Mb " % (totMem * 1e-6) )
            
        # if output and warnings and totMem * 1e-6 > warningMem:
        #     opt = 'a'
        #     while (opt != 'c' and opt != 'b' and opt != 'q'):
        #         self.logger.warning("%n" +
        #                             "The total memory that will be allocated exceed %10.2fMb. Chose one , of the following options:\n" % (warningMem) +
        #                             "\t [c]: continue\n" +
        #                             "\t [q]: exit")
        #         opt = sys.stdin.read(1)
        #     if (opt ==  'q'):
        #         return
        
        if usekron:
            # Compute combinations of collocation points
            xKron = np.asarray(list(itertools.product(*rs)))
        else:
            xKron = rs
        
        # Compute single Generalized Vandermonde Matrices
        Vs = []
        for i in range(0,self.DIM):
            Vs.append(self.polys[i].GradVandermonde(xKron[:,i],N,ks[i],norms[i]))
        
        # Make space for the Vandermonde matrix
        V = np.ones((Ncolls,Nbasis))
        
        # Compute the Pascal's Simplex of the basis functions
        IDX = MultiIndex(self.DIM,N)
        
        for i in range(0,np.size(IDX,0)):
            for j in range(0,np.size(IDX,1)):
                V[:,i] = V[:,i] * Vs[j][:,IDX[i,j]]
        
        return V
        
