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
import orthpol_light
import polymod
try:
    import orthpol
    ORTHPOL_SUPPORT = True
except ImportError:
    ORTHPOL_SUPPORT = False

from SpectralToolbox.Spectral1D.Constants import *
from SpectralToolbox.Spectral1D.AbstractClasses import *

__all__ = ['Poly1D', 'QUADS', 'nestedlobatto', 'nestedgauss',
           'fej', 'cc', 'kpn', 'kpu', 'gqn', 'gqu']
        
class Poly1D(OrthogonalPolynomial):
    """
    Initialization of the Polynomial instance.

    This method generates an instance of the Poly1D class,
    to be used in order to generate
    orthogonal basis of the polynomial type selected.
    Avaliable polynomial types can be
    selected using their string name or by predefined attributes
      * :ref:`ref_jacobi`: 'Jacobi' or ``Spectral1D.JACOBI``
      * Hermite Physicist: 'HermiteP' or ``Spectral1D.HERMITEP``
      * Hermite Function: 'HermiteF' or ``Spectral1D.HERMITEF``
      * Hermite Probabilistic: 'HermitePprob' or ``Spectral1D.HERMITEP_PROB``
      * Hermite Function Probabilistic: 'HermiteFprob' or ``Spectral1D.HERMITEF_PROB``
      * Laguerre Polynomial: 'LaguerreP' or ``Spectral1D.LAGUERREP``
      * Laguerre Function: 'LaguerreF' or ``Spectral1D.LAGUERREF``
      * General orthogonal polynomial: 'ORTHPOL' or ``Spectral1D.ORTHPOL``
      * Fourier: 'Fourier' or ``Spectral1D.FOURIER``
    
    Additional parameters are required for some polynomials.

        +--------------+--------------+
        | Polynomial   | Parameters   |
        +==============+==============+
        | Jacobi       | (alpha,beta) |
        +--------------+--------------+
        | HermiteP     | None         |
        +--------------+--------------+
        | HermiteF     | None         |
        +--------------+--------------+
        | HermitePprob | None         |
        +--------------+--------------+
        | LaguerreP    | alpha        |
        +--------------+--------------+
        | LaguerreF    | alpha        |
        +--------------+--------------+
        | ORTHPOL      | see notes    |
        +--------------+--------------+
        | Fourier      | None         |
        +--------------+--------------+

    Available quadrature rules (related to selected polynomials):
      * Gauss or ``Spectral1D.GAUSS``
      * Gauss-Lobatto or ``Spectral1D.GAUSSLOBATTO``
      * Gauss-Radau or ``Spectral1D.GAUSSRADAU``

    Available quadrature rules (without polynomial selection):
      * Kronrod-Patterson on the real line or ``Spectral1D.KPN`` (function ``Spectral1D.kpn(n)``)
      * Kronrod-Patterson uniform or ``Spectral1D.KPU`` (function ``Spectral1D.kpu(n)``)
      * Clenshaw-Curtis or ``Spectral1D.CC`` (function ``Spectral1D.cc(n)``)
      * Fejer's or ``Spectral1D.FEJ`` (function ``Spectral1D.fej(n)``)
    
    Args:
      poly (string): The orthogonal polynomial type desired
      params (list): The parameters needed by the selected polynomial
      sdout (stream): output stream for logging

    .. note:: The ORTHPOL polynomials are built up using the "Multiple-Component Discretization Procedure" described in :cite:`Gautschi1994`. The following parameters describing the measure function are required in order to use the procedure for finding the recursion coefficients (alpha,beta) and have to be provided at construction time:

            * 0 ``ncapm``: (int) maximum integer N0 (default = 500)
            * 1 ``mc``: (int) number of component intervals in the continuous part of the spectrum
            * 2 ``mp``: (int) number of points in the discrete part of the spectrum. If the measure has no discrete part, set mp=0
            * 3 ``xp``, 4 ``yp``: (Numpy 1d-array) of dimension mp, containing the abscissas and the jumps of the point spectrum
            * 5 ``mu``: (function) measure function that returns the mass (float) with arguments: ``x`` (float) absissa, ``i`` (int) interval number in the continuous part
            * 6 ``irout``: (int) selects the routine for generating the recursion coefficients from the discrete inner product; ``irout=1`` selects the routine ``sti``, ``irout!=1`` selects the routine ``lancz``
            * 7 ``finl``, 8 ``finr``: (bool) specify whether the extreme left/right interval is finite (false for infinite)
            * 9 ``endl``, 10 ``endr``: (Numpy 1d-array) of dimension ``mc`` containing the left and right endpoints of the component intervals. If the first of these extends to -infinity, endl[0] is not being used by the routine.
        Parameters ``iq``, ``quad``, ``idelta`` in :cite:`Gautschi1994` are suppressed. Instead the routine ``qgp`` of ORTHPOL :cite:`Gautschi1994` is used by default (``iq=0`` and ``idelta=2``)

    .. deprecated:: 0.2.0
       Use :class:`OrthogonalPolynomial` and its sub-classes instead.

    Raises:
       DeprecationWarnin: use :class:`OrthogonalPolynomial` and its sub-classes instead.
    
    """
    
    def __init__(self,poly,params,sdout=sys.stderr):

        warnings.warn("Use the class OrthogonalPolynomial and its " +
                      "subclasses instead", DeprecationWarning)

        #####################
        # List of attributes
        self.poly = None
        self.params = None
        self.sdout = None
        #####################
        
        self.sdout = sdout
        
        # Check consistency of polynomial types and parameters
        if (poly in AVAIL_POLY):
            if (poly == JACOBI):
                if len(params) != 2:
                    raise AttributeError("The number of parameters inserted for the polynomial of type '%s' is not correct" % poly)
                    return
            if ((poly == LAGUERREP) or (poly == LAGUERREF)):
                if len(params) != 1:
                    raise AttributeError("The number of parameters inserted for the polynomial of type '%s' is not correct" % poly)
                    return
            if ((poly == ORTHPOL)):
                raise NotImplemented("ORTHPOL support for the Poly1D class has " +
                                     "been removed since version 0.2.0. The " +
                                     "class GenericOrthogonalPolynomial " +
                                     "provides all the ORTHPOL functionalities.")
                if not ORTHPOL_SUPPORT:
                    raise ImportError("The orthpol package is not installed on this machine.")
                if len(params) != 11:
                    raise AttributeError("The number of parameters inserted for the polynomial of type '%s' is not correct" % poly)
                self.orthpol_alphabeta = None
        else:
            raise AttributeError("The inserted type of polynomial is not available.")
        self.poly = poly
        self.params = params
        # Define the base span for each polynomial type
        if self.poly == JACOBI:
            self.base_span = [-1.,1.]
        elif self.poly in [HERMITEP, HERMITEF, HERMITEP_PROB, HERMITEF_PROB]:
            self.base_span = [-np.inf,np.inf]
        elif self.poly == LAGUERREP or self.poly == LAGUERREF:
            self.base_span = [0.,np.inf]
        elif self.poly == ORTHPOL:
            self.base_span = [self.params[9], self.params[10]]
        elif self.poly == FOURIER:
            self.base_span = [0.,2.*np.pi]
            
    
    def orthpol_rec_coeffs(self,N):
        """
        Compute the recursion coefficients for polynomials up to the N-th order
        """
        if not ORTHPOL_SUPPORT:
            raise ImportError("The orthpol package is not installed on this machine.")

        # Precompute alpha and beta parameters
        eps = orthpol.d1mach(3)

        if (self.orthpol_alphabeta == None) or (N > self.orthpol_alphabeta['N']):
            # Unpack self.params
            ncapm = self.params[0]
            mc =    self.params[1]
            mp =    self.params[2]
            xp =    self.params[3]
            yp =    self.params[4]
            mu =    self.params[5]
            irout = self.params[6]
            finl =  self.params[7]
            finr =  self.params[8]
            endl =  self.params[9]
            endr =  self.params[10]
            # Default values
            iq = 0
            idelta = 2
            # Compute the alpha and beta coeff.
            (alphaCap, betaCap, ncapCap, 
             kountCap, ierrCap, ieCap) = orthpol.dmcdis(N+1, ncapm, mc, mp, xp,
                                                        yp, mu, eps, iq, idelta, 
                                                        irout, finl, finr, endl, endr )
            if self.orthpol_alphabeta == None: self.orthpol_alphabeta = {}
            self.orthpol_alphabeta = {'N': N,
                                      'alpha': alphaCap,
                                      'beta': betaCap}
        else:
            alphaCap = self.orthpol_alphabeta['alpha']
            betaCap = self.orthpol_alphabeta['beta']
        
        return (alphaCap,betaCap)

    def __JacobiGQ(self,N):
        """
        Purpose: Compute the N'th order Gauss quadrature points, x, 
                 and weights, w, associated with the Jacobi 
                 polynomial, of type (alpha,beta) > -1 ( <> -0.5).
        """
        if (self.poly != JACOBI):
            print("The method __JacobiGQ cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,-0.5) or AlmostEqual(beta,-0.5) ):
                return self.__JacobiCGQ(N)
            
            x = np.zeros((N+1))
            w = np.zeros((N+1))
            
            if (N == 0):
                x[0] = -(alpha-beta)/(alpha+beta+2)
                w[0] = 2
                return (x,w)
            
            J = np.zeros((N+1,N+1))
            h1 = 2.* np.arange(0.,N+1)+alpha+beta
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                J = np.diag(-1./2.*(alpha**2.-beta**2.)/(h1+2.)/h1) +  np.diag( 2./(h1[0:N] + 2.) * np.sqrt( np.arange(1.,N+1) * (np.arange(1.,N+1)+alpha+beta) * (np.arange(1.,N+1)+alpha) * (np.arange(1.,N+1)+beta) / (h1[0:N] + 1.) / (h1[0:N] + 3.) ) , 1) 
            if (alpha + beta < 10.*np.finfo(np.float64).eps): J[0,0] = 0.0
            J = J + np.transpose(J)
            
            # Compute quadrature by eigenvalue solve
            vals,vecs = LA.eigh(J)
            perm = np.argsort(vals)
            x = vals[perm]
            vecs = vecs[:,perm]
            w = np.power(np.transpose(vecs[0,:]),2.) * 2**(alpha+beta+1.)/(alpha+beta+1.)*gammaF(alpha+1.)*gammaF(beta+1.)/gammaF(alpha+beta+1.)
            return (np.asarray(x),np.asarray(w))
        
    def __JacobiGL(self,N):
        """
        x = JacobiGL
        Purpose: Compute the N'th order Gauss Lobatto quadrature points, x, 
                 and weights, w, associated with the Jacobi 
                 polynomial, of type (alpha,beta) > -1 ( <> -0.5).
        
        .. note:: For the computation of the weights for generic Jacobi polynomials, see :cite:`Shi-jun2002`
        """
        if (self.poly != JACOBI):
            print("The method __JacobiGL cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,0.0) or AlmostEqual(beta,0.0) ):
                return self.__JacobiLGL(N)
            elif ( AlmostEqual(alpha,-0.5) or AlmostEqual(beta,-0.5) ):
                return self.__JacobiCGL(N)
            
            x = np.mat(np.zeros((N+1)))
            if ( N == 1 ): 
                x[0] = -1.
                x[1] = 1.
                return x
                
            [xint,wint] = self.__JacobiGQ(N-2)
            x = np.concatenate(([-1.], xint, [1.]))
            w0 = ( 2.**(alpha+beta+1.) * gammaF(alpha+2) * gammaF(beta+1) / gammaF(alpha+beta+3) ) * SPcomb(N+alpha,N-1)/ (SPcomb(N+beta,N-1) * SPcomb(N+alpha+beta+1,N-1))
            wint *= 1./(1.-xint**2.)
            w = np.concatenate(([w0],wint,[w0]))
            return (x,w)
        
    def __JacobiLGL(self,N):
        """
        x,w = JacobiLGL(N)
        Compute the Legendre Gauss Lobatto points and weights for polynomials up
        to degree N
        Algorithm (25) taken from :cite:`Kopriva2009`
        """
        if (self.poly != JACOBI):
            print("The method __JacobiLGL cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else: 
            # Unpack parameters
            alpha,beta = self.params
            if ( not AlmostEqual(alpha,0.0) or not AlmostEqual(beta,0.0) ):
                print("The method can be called only for Legendre Polynomials. Actual values of alpha and beta: %f, %f" % (alpha,beta))
            else:
                maxNewtonIter = 1000
                NewtonTOL = 1e-12
                x = np.zeros((N+1))
                w = np.zeros((N+1))
                if ( N == 1 ):
                    x[0] = -1.
                    x[1] = 1.
                    w[0] = 1.
                    w[1] = 1.
                else:
                    x[0] = -1.
                    w[0] = 2./(N * (N + 1.))
                    x[N] = 1.
                    w[N] = w[0]
                    for j in range(1,int(np.floor((N+1.)/2.)-1) + 1):
                        x[j] = -np.cos( (j + 1./4.)*np.pi/N - 3./(8.*N*np.pi) * 1./(j+1./4.) )
                        # Newton iteratio for getting the point
                        k = 0
                        delta = 10. * NewtonTOL
                        while ( (k < maxNewtonIter) and (abs(delta) > NewtonTOL * abs(x[j])) ):
                            k = k + 1
                            q,dq,LN = self.__qLEvaluation(N,x[j])
                            delta = -q/dq
                            x[j] = x[j] + delta
                        q,dq,LN = self.__qLEvaluation(N,x[j])
                        x[N-j] = -x[j]
                        w[j] = 2./(N*(N+1)*LN**2)
                        w[N-j] = w[j]
                if ( np.remainder(N,2) == 0 ):
                    q,dq,LN = self.__qLEvaluation(N,0.)
                    x[N/2] = 0
                    w[N/2] = 2./(N*(N+1)*LN**2)
                return (np.asarray(x),np.asarray(w))
        
    def __JacobiCGQ(self,N):
        """
        Compute the Chebyshev Gauss points and weights for polynomials up
        to degree N
        Algorithm (26) taken from :cite:`Kopriva2009`
        """
        if (self.poly != JACOBI):
            print("The method __JacobiCGQ cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( not AlmostEqual(alpha,-0.5) or not AlmostEqual(beta,-0.5) ):
                print("The method can be called only for Chebyshev Polynomials. Actual values of alpha and beta: %f, %f" % (alpha,beta))
            else:
                x = np.zeros((N+1))
                w = np.zeros((N+1))
                for j in range(0,N+1):
                    x[j] = -np.cos( (2.*j + 1.)/(2.*N + 2.) * np.pi )
                    w[j] = np.pi / (N + 1.)
                return (np.asarray(x),np.asarray(w))
        
    def __JacobiCGL(self,N):
        """
        x,w = JacobiCL(N)
        Compute the Chebyshev Gauss Lobatto points and weights for polynomials up
        to degree N
        Algorithm (27) taken from :cite:`Kopriva2009`
        """
        if (self.poly != JACOBI):
            print("The method __JacobiCGL cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( not AlmostEqual(alpha,-0.5) or not AlmostEqual(beta,-0.5) ):
                print("The method can be called only for Chebyshev Polynomials. Actual values of alpha and beta: %f, %f" % (alpha,beta))
            else:
                x = np.zeros((N+1))
                w = np.zeros((N+1))
                for j in range(0,N+1):
                    x[j] = -np.cos(float(j)/float(N) * np.pi)
                    w[j] = np.pi/float(N)
                w[0] = w[0]/2.
                w[N] = w[N]/2.
                return (np.asarray(x),np.asarray(w))
        
    def __HermitePGQ(self,N):
        """
        Compute the Hermite-Gauss quadrature points and weights for Hermite Physicists 
        polynomials up to degree N
        For further details see :cite:`Shen2009a`
        """
        if (self.poly != HERMITEP):
            print("The method __HermitePGQ cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            j = np.asarray(range(1,N+1))
            b = np.sqrt(j / 2.)
            D = np.diag(b,1)
            D = D + D.T
            x,vec = np.linalg.eig(D)
            x = np.sort(x)
            hp = self.__HermiteP(x,N)
            w = np.sqrt(np.pi) * 2.**N * factorial(N) / ((N+1) * hp**2.)
            return (x,w)
        
    def __HermiteFGQ(self,N):
        """
        Compute the Hermite-Gauss quadrature points and weights for the Hermite 
        functions up to degree N
        For further details see :cite:`Shen2009a`
        """
        if (self.poly not in [HERMITEF, HERMITEF_PROB]):
            print("The method __HermiteFGQ cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            j = np.asarray(range(1,N+1))
            b = np.sqrt(j / 2.)
            D = np.diag(b,1)
            D = D + D.T
            x,vec = np.linalg.eig(D)
            x = np.sort(x)
            hf = self.__HermiteF(x,N)
            w = np.sqrt(np.pi) / ((N+1) * hf**2.)
            return (x,w)
    
    def __HermiteP_Prob_GQ(self,N):
        """
        Compute the Hermite-Gauss quadrature points and weights for Hermite 
        Probabilistic polynomials up to degree N
        For further details see Golub-Welsh algorithm in [3]
        """
        if (self.poly != HERMITEP_PROB):
            print("The method __HermiteP_Prob_GQ cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            j = np.asarray(range(1,N+1))
            b = np.sqrt(j)
            D = np.diag(b,1)
            D = D + D.T
            x,vec = np.linalg.eig(D)
            x = np.sort(x)
            hp = self.__HermiteP_Prob(x,N)
            w = factorial(N) / ((N+1) * hp**2.)
            return (x,w)
    
    def __LaguerrePGQ(self,N,alpha=None):
        """
        __LaguerrePGQ(): Compute the Laguerre-Gauss quadrature points and weights for Laguerre polynomials up to degree N
        
        Syntax:
            ``(x,w) = __LaguerrePGQ(N,[alpha=None])``
        
        Input:
            * ``N`` = (int) Order of polynomial accuracy
            * ``alpha`` = (optional,float) Laguerre constant
        
        Output:
            * ``x`` = set of Laguerre-Gauss quadrature points
            * ``w`` = set of Laguerre-Gauss quadrature weights
        
        .. note:: For further details see :cite:`Shen2009a`
        """
        if (self.poly != LAGUERREP):
            print("The method __LaguerrePGQ cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            if (alpha is None):
                (alpha,) = self.params
            # Compute points and weights
            j = np.asarray(range(0,N+1))
            a = 2. * j + alpha + 1.
            b = - np.sqrt( j[1:] * (j[1:] + alpha) )
            D = np.diag(b,1)
            D = D + D.T
            D = D + np.diag(a,0)
            x,vec = np.linalg.eig(D)
            x = np.sort(x)
            lp = self.__LaguerreP(x,N).ravel()
            w = gammaF(N+alpha+1.)/ ((N+alpha+1.) * factorial(N+1)) * ( x / lp**2. )
            return (x,w)
    
    def __LaguerreFGQ(self,N,alpha=None):
        """
        __LaguerreFGQ(): Compute the Laguerre-Gauss quadrature points and weights for Laguerre functions up to degree N
        
        Syntax:
            ``(x,w) = __LaguerreFGQ(N,[alpha=None])``
        
        Input:
            * ``N`` = (int) Order of polynomial accuracy
            * ``alpha`` = (optional,float) Laguerre constant
        
        Output:
            * ``x`` = set of Laguerre-Gauss quadrature points
            * ``w`` = set of Laguerre-Gauss quadrature weights
        
        .. note:: For further details see :cite:`Shen2009a`
        """
        if (self.poly != LAGUERREF):
            print("The method __LaguerreFGQ cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            if (alpha is None):
                (alpha,) = self.params
            # Compute points and weights
            j = np.asarray(range(0,N+1))
            a = 2. * j + alpha + 1.
            b = - np.sqrt( j[1:] * (j[1:] + alpha) )
            D = np.diag(b,1)
            D = D + D.T
            D = D + np.diag(a,0)
            x,vec = np.linalg.eig(D)
            x = np.sort(x)
            lf = self.__LaguerreF(x,N).ravel()
            w = gammaF(N+alpha+1.)/ ((N+alpha+1.) * factorial(N+1)) * ( x / lf**2. )
            return (x,w)
    
    def __LaguerrePGR(self,N,alpha=None):
        """
        __LaguerrePGR(): Compute the Laguerre-Gauss-Radau quadrature points and weights for Laguerre polynomials up to degree N
        
        Syntax:
            ``(x,w) = __LaguerrePGR(N,[alpha=None])``
        
        Input:
            * ``N`` = (int) Order of polynomial accuracy
            * ``alpha`` = (optional,float) Laguerre constant
        
        Output:
            * ``x`` = set of Laguerre-Gauss-Radau quadrature points
            * ``w`` = set of Laguerre-Gauss-Radau quadrature weights
        
        .. note:: For further details see :cite:`Shen2009a`
        """
        if (self.poly != LAGUERREP):
            print("The method __LaguerrePGR cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            if (alpha is None):
                (alpha,) = self.params
            # Compute points and weights x1...xN, w1...wN
            j = np.asarray(range(0,N))
            a = 2. * j + (alpha+1.) + 1.
            b = - np.sqrt( j[1:] * (j[1:] + (alpha+1.) ) )
            D = np.diag(b,1)
            D = D + D.T
            D = D + np.diag(a,0)
            x,vec = np.linalg.eig(D)
            x = np.sort(x)
            lp = self.__LaguerreP(x,N).ravel()
            w = gammaF(N+alpha+1.)/ ((N+alpha+1.) * factorial(N)) * ( 1. / lp**2. )
            # Add x0 and w0
            x = np.hstack((0.0,x))
            w0 = (alpha+1.) * gammaF(alpha+1.)**2. * gammaF(N+1) / gammaF(N+alpha+2.)
            w = np.hstack((w0,w))
            return (x,w)
    
    def __LaguerreFGR(self,N,alpha=None):
        """
        __LaguerreFGR(): Compute the Laguerre-Gauss-Radau quadrature points and weights for Laguerre functions up to degree N
        
        Syntax:
            ``(x,w) = __LaguerreFGR(N,[alpha=None])``
        
        Input:
            * ``N`` = (int) Order of polynomial accuracy
            * ``alpha`` = (optional,float) Laguerre constant
        
        Output:
            * ``x`` = set of Laguerre-Gauss-Radau quadrature points
            * ``w`` = set of Laguerre-Gauss-Radau quadrature weights
        
        .. note:: For further details see :cite:`Shen2009a`
        """
        if (self.poly != LAGUERREF):
            print("The method __LaguerreFGR cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            if (alpha is None):
                (alpha,) = self.params
            # Compute points and weights x1...xN, w1...wN
            j = np.asarray(range(0,N))
            a = 2. * j + (alpha+1.) + 1.
            b = - np.sqrt( j[1:] * (j[1:] + (alpha+1.) ) )
            D = np.diag(b,1)
            D = D + D.T
            D = D + np.diag(a,0)
            x,vec = LA.eig(D)
            x = np.sort(x)
            lp = self.__LaguerreF(x,N).ravel()
            w = gammaF(N+alpha+1.)/ ((N+alpha+1.) * factorial(N)) * ( 1. / lp**2. )
            # Add x0 and w0
            x = np.hstack((0.0,x))
            w0 = (alpha+1.) * gammaF(alpha+1.)**2. * gammaF(N+1) / gammaF(N+alpha+2.)
            w = np.hstack((w0,w))
            return (x,w)
    
    def __ORTHPOL_GQ(self,N):
        """
        __ORTHPOL_GQ(): Compute the ORTHPOL Gauss quadrature points and weights for a generic measure functions up to degree N
        
        Syntax:
            ``(x,w,ierr) = __ORTHPOL_GQ(N)``
        
        Input:
            * ``N`` = (int) Order of polynomial accuracy
        
        Output:
            * ``x`` = set of Gauss quadrature points
            * ``w`` = set of Gauss quadrature weights
            * ``ierr`` = (int) error flag equal to 0 on normal return, equal to ``i`` if the QR algorithm does not converge within 30 iterations on evaluating the ``i``-th eigenvalue, equal to -1 if ``N`` is not in range and equal to -2 if one of the ``beta`` is negative.
        
        .. note:: For further details see :cite:`Gautschi1994`
        """
        eps = orthpol.d1mach(3)
        (alphaCap,betaCap) = self.orthpol_rec_coeffs(N)
        
        # Compute Gauss quadrature points and weights
        (xg,wg,ierr) = orthpol.dgauss(N+1,alphaCap,betaCap,eps);
        
        return (xg,wg,ierr)
    
    def __ORTHPOL_GL(self,N,left,right):
        """
        __ORTHPOL_GL(): Compute the ORTHPOL Gauss-Lobatto quadrature points and weights for a generic measure functions up to degree N
        
        Syntax:
            ``(x,w,ierr) = __ORTHPOL_GL(N,left,right)``
        
        Input:
            * ``N`` = (int) Order of polynomial accuracy
            * ``left`` = (optional,float) containing the left endpoint
            * ``right`` = (optional,float) containing the right endpoint
        
        Output:
            * ``x`` = set of Gauss-Lobatto quadrature points
            * ``w`` = set of Gauss-Lobatto quadrature weights
            * ``ierr`` = (int) error flag equal to 0 on normal return, equal to ``i`` if the QR algorithm does not converge within 30 iterations on evaluating the ``i``-th eigenvalue, equal to -1 if ``N`` is not in range and equal to -2 if one of the ``beta`` is negative.
        
        .. note:: For further details see :cite:`Gautschi1994`
        """
        eps = orthpol.d1mach(3)
        (alphaCap,betaCap) = self.orthpol_rec_coeffs(N)

        # Compute Gauss quadrature points and weights
        (xg,wg,ierr) = orthpol.dlob(N, alphaCap, betaCap, left, right);
        
        return (xg,wg,ierr)
    
    
    def __ORTHPOL_GR(self,N,end):
        """
        __ORTHPOL_GR(): Compute the ORTHPOL Gauss-Radau quadrature points and weights for a generic measure functions up to degree N
        
        Syntax:
            ``(x,w,ierr) = __ORTHPOL_GR(N,end)``
        
        Input:
            * ``N`` = (int) Order of polynomial accuracy
            * ``end`` = (optional,float) containing the endpoint
        
        Output:
            * ``x`` = set of Gauss-Lobatto quadrature points
            * ``w`` = set of Gauss-Lobatto quadrature weights
            * ``ierr`` = (int) error flag equal to 0 on normal return, equal to ``i`` if the QR algorithm does not converge within 30 iterations on evaluating the ``i``-th eigenvalue, equal to -1 if ``N`` is not in range and equal to -2 if one of the ``beta`` is negative.
        
        .. note:: For further details see :cite:`Gautschi1994`
        """
        eps = orthpol.d1mach(3)
        (alphaCap,betaCap) = self.orthpol_rec_coeffs(N)

        # Compute Gauss quadrature points and weights
        (xg,wg,ierr) = orthpol.dradau(N, alphaCap, betaCap, end);
        
        return (xg,wg,ierr)

    def __FourierGQ(self, N):
        """
        Compute the Fourier quadrature points and weights for Fourier interpolation/projection up to degree N
        
        Syntax:
            ``(x,w) = __FourierGQ(N)``

        :param int N: order of accuracy
        :return: tuple :py:data:`(x,w)`

           * :py:data:`x` (np.ndarray): quadrature points
           * :py:data:`w` (np.ndarray): quadrature weights
        """
        x = np.linspace(0., 2.*np.pi, N+2)[:-1]
        w = 2. * np.pi * np.ones(N+1)/float(N+1)
        return (x,w)
    

    def Quadrature(self, N, quadType=None, normed=False, left=None, right=None, end=None):
        """
        Quadrature(): Generates list of nodes and weights for the ``quadType`` quadrature rule using the selected Polynomial basis
        
        Syntax:
            ``(x,w) = Quadrature(N, [quadType=None], [normed=False], [left=None], [right=None], [end=None])``
        
        Input:
            * ``N`` = (int) accuracy level required
            * ``quadType`` = (``AVAIL_QUADPOINTS``) type of quadrature to be used. Default is Gauss quadrature rule.
            * ``normed`` = (optional,bool) whether the weights will be normalized or not
            * ``left`` = (optional,float) containing the left endpoint (used by ORTHPOL Gauss-Lobatto rules)
            * ``right`` = (optional,float) containing the right endpoint (used by ORTHPOL Gauss-Lobatto rules)
            * ``end`` = (optional,float) containing the endpoint (used by ORTHPOL Gauss-Radau rules)
        
        Output:
            * ``x`` = (1d-array,float) containing the nodes
            * ``w`` = (1d-array,float) containing the weights
        """
        if (quadType == None) or (quadType == GAUSS):
            return self.GaussQuadrature(N, normed)
        elif quadType == GAUSSLOBATTO:
            return self.GaussLobattoQuadrature(N,normed,left,right)
        elif quadType == GAUSSRADAU:
            return self.GaussRadauQuadrature(N,normed,end)
        elif quadType in AVAIL_QUADPOINTS:
            if self.poly == HERMITEP_PROB and (quadType in [GQN,KPN]):
                if quadType == GQN: return gqn(N)
                if quadType == KPN: return kpn(N)
            if self.poly == JACOBI and (quadType in [GQU,KPU,CC,FEJ]):
                if quadType == GQU: return gqu(N,norm=normed)
                if quadType == KPU: return kpu(N,norm=normed)
                if quadType == CC:  return cc(N,norm=normed)
                if quadType == FEJ: return fej(N,norm=normed)
                
        else:
            raise AttributeError("The selected type of quadrature rule is not available")
    
    def GaussQuadrature(self,N,normed=False):
        """
        GaussQuadrature(): Generates list of nodes and weights for the Gauss quadrature rule using the selected Polynomial basis
        
        Syntax:
            ``(x,w) = GaussQuadrature(N,[normed=False])``
        
        Input:
            * ``N`` = (int) accuracy level required
            * ``normed`` = (optional,bool) whether the weights will be normalized or not
        
        Output:
            * ``x`` = (1d-array,float) containing the nodes
            * ``w`` = (1d-array,float) containing the weights
        """
        if (self.poly == JACOBI):
            (x,w) = self.__JacobiGQ(N)
        elif (self.poly == HERMITEP):
            (x,w) = self.__HermitePGQ(N)
        elif (self.poly == HERMITEF):
            (x,w) = self.__HermiteFGQ(N)
        elif (self.poly == HERMITEP_PROB):
            (x,w) = self.__HermiteP_Prob_GQ(N)
        elif (self.poly == HERMITEF_PROB):
            (x,w) = self.__HermiteFGQ(N)
            x *= np.sqrt(2)
        elif (self.poly == LAGUERREP):
            (x,w) = self.__LaguerrePGQ(N)
        elif (self.poly == LAGUERREF):
            (x,w) = self.__LaguerreFGQ(N)
        elif (self.poly == ORTHPOL):
            (x,w,ierr) = self.__ORTHPOL_GQ(N)
            if ierr != 0:
                print("Error in ORTHPOL GaussQuadrature.")
        elif (self.poly == FOURIER):
            (x,w) = self.__FourierGQ(N)
        
        if normed:
            w = w / np.sum(w)
        
        return (x,w.flatten())

    def GaussLobattoQuadrature(self,N,normed=False,left=None,right=None):
        """
        GaussLobattoQuadrature(): Generates list of nodes for the Gauss-Lobatto quadrature rule using selected Polynomial basis
        
        Syntax:
            ``x = GaussLobattoQuadrature(N,[normed=False],[left=None],[right=None])``
        
        Input:
            * ``N`` = (int) accuracy level required
            * ``normed`` = (optional,bool) whether the weights will be normalized or not
            * ``left`` = (optional,float) containing the left endpoint (used by ORTHPOL)
            * ``right`` = (optional,float) containing the right endpoint (used by ORTHPOL)            
        
        Output:
            * ``x`` = (1d-array,float) containing the nodes
            * ``w`` = (1d-array,float) containing the weights
        
        .. note:: Available only for Jacobi Polynomials and ORTHPOL
        """
        if (self.poly == JACOBI):
            (x,w) = self.__JacobiGL(N)
        elif (self.poly == ORTHPOL):
            (x,w,ierr) = self.__ORTHPOL_GL(N,left,right)
            if ierr != 0:
                print("Error in ORTHPOL GaussLobattoQuadrature.")
        else:
            print("Gauss Lobatto quadrature does not apply to the selected Polynomials/Function.")
        
        if normed:
            w = w / np.sum(w)
        
        return (x,w.flatten())
        
    
    def GaussRadauQuadrature(self,N,normed=False,end=None):
        """
        GaussRadauQuadrature(): Generates list of nodes for the Gauss-Radau quadrature rule using selected Polynomial basis
        
        Syntax:
            ``x = GaussRadauQuadrature(N,[normed=False],[end=None])''
        
        Input:
            * ``N'' = (int) accuracy level required
            * ``normed`` = (optional,bool) whether the weights will be normalized or not
            * ``end`` = (optional,float) containing the endpoint (used by ORTHPOL)
        
        Output:
            * ``x'' = (1d-array,float) containing the nodes
            * ``w'' = (1d-array,float) weights
        
        .. note:: Available only for Laguerre Polynomials/Functions and ORTHPOL
        """
        if (self.poly == LAGUERREP):
            (x,w) = self.__LaguerrePGR(N)
        elif (self.poly == LAGUERREF):
            (x,w) = self.__LaguerreFGR(N)
        elif (self.poly == ORTHPOL):
            (x,w,ierr) = self.__ORTHPOL_GR(N,end)
            if ierr != 0:
                print("Error in ORTHPOL GaussRadauQuadrature.")
        else:
            print("Gauss Radau quadrature does not apply to the selected Polynomials/Function.")
        
        if normed:
            w = w / np.sum(w)
        
        return (x,w.flatten())
        
    def __qLEvaluation(self,N,x):
        """
        q,dq,LN = qLEvaluation(N,x)
        Evaluate Legendre Polynomial LN and 
        q = L_N+1 - L_N-1
        q' = L'_N+1 - L'_N-1
        at point x.
        Algorithm (24) taken from :cite:`Kopriva2009`
        """
        L = np.zeros((N+2))
        DL = np.zeros((N+2))
        L[0] = 1.
        L[1] = x
        DL[0] = 0.
        DL[1] = 1.
        for k in range(2,N+2):
            L[k] = (2.*k-1.)/k * x * L[k-1] - (k-1.)/k * L[k-2]
            DL[k] = DL[k-2] + (2.*k-1.) * L[k-1]
        q = L[N+1] - L[N-1]
        dq = DL[N+1] - DL[N-1]
        return (q,dq,L[N])

    def __JacobiP(self,x,N,alpha=None,beta=None, 
                  isConstructingVandermonde=False, recursion_storage=None):
        """
        Returns an 1d-array with the Jacobi polynomial of order N at points r

        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param alpha: alpha parameter for Jacobi polynomial
        :type alpha: float
        :param beta: beta parameter for Jacobi polynomial
        :type beta: float
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly != JACOBI):
            print("The method __JacobiP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if (alpha is None) and (beta is None):
                # Unpack parameters
                alpha,beta = self.params
            if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
                return self.__ChebyshevP(x,N,isConstructingVandermonde,recursion_storage)
            else:
                if ORTHPOL_SUPPORT and not isConstructingVandermonde:
                    (al,be,ierr) = orthpol.drecur(N+1,6,alpha,beta)
                    v = orthpol.polyeval(x,N,al,be)
                    return v
                else:
                    # In the recursion storages, PL[:,0] is the oldest (N-2), PL[:,1] is the newer (N-1)
                    if isConstructingVandermonde:
                        gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gammaF(alpha+1)*gammaF(beta+1)/gammaF(alpha+beta+1)
                        if (N==0):
                            Pout = 1.0/np.sqrt(gamma0)
                            recursion_storage[:,0] = Pout
                            return Pout
                        gamma1 = (alpha+1.)*(beta+1.)/(alpha+beta+3.)*gamma0
                        if (N==1):
                            Pout = ((alpha+beta+2.)*x/2. + (alpha-beta)/2.)/np.sqrt(gamma1)
                            recursion_storage[:,1] = Pout
                            return Pout
                        h = np.zeros([2])
                        a = np.zeros([2])
                        b = np.zeros([2])
                        for idx,i in enumerate(range(N-1,N+1)):
                            fi = float(i)
                            h[idx] = 2. * (fi-1.) + alpha + beta
                            a[idx] = 2. / (h[idx] + 2.) * np.sqrt( (fi*(fi+alpha+beta)*(fi+alpha)*(fi+beta))/((h[idx]+1.)*(h[idx]+3.)) )
                            if idx==1:
                                b[idx] = - (alpha**2. - beta**2.) / (h[idx]*(h[idx]+2.))
                        Pout = 1.0/a[1] * ( -a[0] * recursion_storage[:,0] + (x-b[1]) * recursion_storage[:,1] )
                        recursion_storage[:,0] = recursion_storage[:,1]
                        recursion_storage[:,1] = Pout
                    else:
                        PL = np.zeros((len(x),2))
                        gamma0 = 2**(alpha+beta+1)/(alpha+beta+1)*gammaF(alpha+1)*gammaF(beta+1)/gammaF(alpha+beta+1)
                        PL[:,0] = 1.0/np.sqrt(gamma0)
                        if (N == 0): return PL[:,0]
                        gamma1 = (alpha+1.)*(beta+1.)/(alpha+beta+3.)*gamma0
                        PL[:,1] = ((alpha+beta+2.)*x/2. + (alpha-beta)/2.)/np.sqrt(gamma1)
                        if (N == 1): return PL[:,1]
                        # Recurrence
                        aold = 2./(2.+alpha+beta)*np.sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.))
                        for i in range(1,N):
                            h1 = 2.*i+alpha+beta
                            anew = 2./(h1+2.)*np.sqrt( (i+1.)*(i+1.+alpha+beta)*(i+1.+alpha)*(i+1.+beta)/(h1+1.)/(h1+3.) )
                            bnew = - (alpha**2. - beta**2.)/h1/(h1+2.)
                            Pout = 1.0 / anew*( -aold*PL[:,0] + np.multiply((x-bnew),PL[:,1]) )
                            PL[:,0] = PL[:,1]
                            PL[:,1] = Pout
                            aold = anew
                    
                    return Pout
        
    def __ChebyshevP(self,r, N, isConstructingVandermonde=False, recursion_storage=None):
        """
        Returns an 1d-array with the Chebyshev (first type) polynomial
        of order N at points r
        Algorithm (21) taken from :cite:`Kopriva2009`

        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly != JACOBI):
            print("The method __ChebyshevP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
                # shape r
                Ks = 50
                rShape = r.shape
                if ( N == 0 ): 
                    T = np.ones(rShape)
                    if isConstructingVandermonde:
                        recursion_storage[:,0] = T
                    return T
                if ( N == 1 ): 
                    T = r
                    if isConstructingVandermonde:
                        recursion_storage[:,1] = T
                    return T
                if ( N <= Ks):
                    if isConstructingVandermonde:
                        T = 2 * r * recursion_storage[:,1] - recursion_storage[:,0]
                        recursion_storage[:,0] = recursion_storage[:,1]
                        recursion_storage[:,1] = T
                    else:
                        T2 = np.ones(rShape)
                        T1 = r
                        for j in range(2,N+1):
                            T = 2 * r *  T1 - T2
                            T2 = T1
                            T1 = T
                else:
                    T = np.cos(N * np.arccos(r) )
                return T
        
    def __HermiteP(self,r, N, isConstructingVandermonde=False, recursion_storage=None):
        """
        Returns the N-th Hermite Physicist polynomial using the recurrence relation
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly not in [HERMITEP, HERMITEF, HERMITEF_PROB]):
            print("The method __HermiteP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if (N == 0):
                old2 = np.ones( len(r) )
                if isConstructingVandermonde:
                    recursion_storage[:,0] = old2
                return old2
            if (N == 1):
                old1 = 2. * r
                if isConstructingVandermonde:
                    recursion_storage[:,1] = old1
                return old1
            if isConstructingVandermonde:
                out = 2. * r * recursion_storage[:,1] - \
                      2. * (N-1) * recursion_storage[:,0]
                recursion_storage[:,0] = recursion_storage[:,1]
                recursion_storage[:,1] = out
            else:
                old2 = np.ones( len(r) )
                old1 = 2. * r
                out = 2. * r * old1 - 2. * old2
                for i in range(3,N+1):
                    old2 = old1
                    old1 = out
                    out = 2. * r * old1 - 2. * (i-1) * old2
            return out
            
    def __HermiteF(self,r, N, isConstructingVandermonde=False, recursion_storage=None):
        """
        Returns the N-th Hermite function using the recurrence relation
        Reference: [2]
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly not in [HERMITEF, HERMITEF_PROB]):
            print("The method __HermiteF cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            new = (2.**N * factorial(N) * np.sqrt(np.pi))**(-.5) * np.exp(-(r**2.)/2.) * \
                  self.__HermiteP(r, N, isConstructingVandermonde, recursion_storage)
            return new
    
    def __HermiteP_Prob(self,r, N, isConstructingVandermonde=False, recursion_storage=None):
        """
        Returns the N-th Hermite Probabilistic polynomial using the recurrence relation
        Use the Probabilistic Hermite Polynomial
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly not in [HERMITEP_PROB, HERMITEF, HERMITEF_PROB]):
            print("The method __HermiteP_Prob cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if (N == 0):
                old2 = np.ones( len(r) )
                if isConstructingVandermonde:
                    recursion_storage[:,0] = old2
                return old2
            if (N == 1):
                old1 = r.copy()
                if isConstructingVandermonde:
                    recursion_storage[:,1] = old1
                return old1
            if isConstructingVandermonde:
                out = r * recursion_storage[:,1] - (N-1) * recursion_storage[:,0]
                recursion_storage[:,0] = recursion_storage[:,1]
                recursion_storage[:,1] = out
            else:
                old2 = np.ones( len(r) )
                old1 = r.copy()
                out = r * old1 - old2
                for i in range(3,N+1):
                    old2 = old1
                    old1 = out
                    out = r * old1 - (i-1) * old2
            return out
        
    def __LaguerreP(self,r, N, alpha=None, isConstructingVandermonde=False, recursion_storage=None):
        """
        Generates the N-th Laguerre polynomial using the recurrence relation
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param alpha: alpha parameter for Laguerre polynomial
        :type alpha: float
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        
        if (self.poly != LAGUERREP):
            print("The method __LaguerreP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            if (alpha == None):
                (alpha,) = self.params
            # Recurrence relation
            if (N == 0):
                old2 = np.ones( len(r) )
                if isConstructingVandermonde:
                    recursion_storage[:,0] = old2
                return old2
            if (N == 1):
                old1 = alpha + 1. - r
                if isConstructingVandermonde:
                    recursion_storage[:,1] = old1
                return old1
            if isConstructingVandermonde:
                n = N-1.
                out = ( (2*n + alpha + 1. - r) * recursion_storage[:,1] - \
                        (n + alpha) * recursion_storage[:,0] ) / (n + 1.)
                recursion_storage[:,0] = recursion_storage[:,1]
                recursion_storage[:,1] = out
            else:
                old2 = np.ones( len(r) )
                old1 = alpha + 1. - r
                out = ( (3. + alpha - r) * old1 - (1. + alpha) * old2 ) / 2.0
                for i in range(3,N+1):
                    n = i - 1.
                    old2 = old1
                    old1 = out
                    out = ( (2*n + alpha + 1. - r) * old1 - (n + alpha) * old2 ) / (n + 1.)
            return out

    def __LaguerreF(self,r,N,alpha=None, isConstructingVandermonde=False, recursion_storage=None):
        """
        Generates the N-th Laguerre function using the recurrence relation
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param alpha: alpha parameter for Laguerre polynomial
        :type alpha: float
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        
        if (self.poly != LAGUERREF):
            print("The method __LaguerreF cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            if (alpha == None):
                (alpha,) = self.params
            # Recurrence relation
            if (N == 0):
                old2 = np.exp(-r/2.)
                if isConstructingVandermonde:
                    recursion_storage[:,0] = old2
                return old2
            if (N == 1):
                old1 = (alpha + 1. - r) * np.exp(-r/2.)
                if isConstructingVandermonde:
                    recursion_storage[:,1] = old1
                return old1
            if isConstructingVandermonde:
                n = N-1.
                out = ( (2*n + alpha + 1. - r) * recursion_storage[:,1] - \
                        (n + alpha) * recursion_storage[:,0] ) / (n + 1.)
                recursion_storage[:,0] = recursion_storage[:,1]
                recursion_storage[:,1] = out
            else:
                old2 = np.exp(-r/2.)
                old1 = (alpha + 1. - r) * np.exp(-r/2.)
                out = ( (3. + alpha - r) * old1 - (1. + alpha) * old2 ) / 2.0
                for i in range(3,N+1):
                    n = i - 1.
                    old2 = old1
                    old1 = out
                    out = ( (2*n + alpha + 1. - r) * old1 - (n + alpha) * old2 ) / (n + 1.)
            return out
    
    def __GradLaguerreP(self,r,N,k,alpha=None, 
                        isConstructingVandermonde=False, recursion_storage=None):
        """
        Generates the k-th derivative of the N-th Laguerre polynomial using the recurrence relation
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param k: order of the derivative
        :type k: int
        :param alpha: alpha parameter for Laguerre polynomial
        :type alpha: float
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        
        if (self.poly != LAGUERREP):
            print("The method __LaguerreP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            if (alpha == None):
                (alpha,) = self.params
            if (k == 0):
                dP = self.__LaguerreP(r,N,alpha, isConstructingVandermonde, recursion_storage)
            else:
                if (N == 0):
                    dP = np.zeros(r.shape)
                else:
                    dP = - self.__GradLaguerreP(r,N-1,k-1,alpha+1.,
                                                isConstructingVandermonde, 
                                                recursion_storage)
            return dP
    
    def __GradLaguerreF(self,r,N,k,alpha=None, 
                        isConstructingVandermonde=False, recursion_storage=None):
        """
        Generates the k-th derivative of the N-th Laguerre function using the recurrence relation
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param k: order of the derivative
        :type k: int
        :param alpha: alpha parameter for Laguerre polynomial
        :type alpha: float
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        
        if (self.poly != LAGUERREF):
            print("The method __LaguerreF cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            if (alpha == None):
                (alpha,) = self.params
            if (k == 0):
                if alpha == self.params[0]:
                    dP = self.__LaguerreF(r, N, alpha, isConstructingVandermonde, recursion_storage)
                else:
                    dP = self.__LaguerreF(r,N,alpha)
            else:
                if (N == 0):
                    dP = -0.5 * self.__GradLaguerreF(r,N,k-1,alpha)
                else:
                    dP = - self.__GradLaguerreF(r,N-1,k-1,alpha+1.) - 0.5 * self.__GradLaguerreF(r,N,k-1,alpha)
            return dP
    
    def __GradHermiteP(self,r,N,k, isConstructingVandermonde=False, recursion_storage=None):
        """
        Compute the first derivative of the N-th Hermite Physicist Polynomial 
        using the recursion relation in [2]
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param k: order of the derivative
        :type k: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly != HERMITEP):
            print("The method __GradHermiteP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if (N-k < 0):
                dP = np.zeros(r.shape)
            else:
                fact = np.exp( gammalnF(N+1) - gammalnF(N-k+1) )
                dP = 2.**k * fact * self.__HermiteP(r,N-k, isConstructingVandermonde, recursion_storage)
            return dP
        
    def __GradHermiteF(self,r,N,k, isConstructingVandermonde=False, recursion_storage=None):
        """
        Compute the first derivative of the N-th Hermite Function using the
        recursion relation in [2]
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param k: order of the derivative
        :type k: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly not in [HERMITEF, HERMITEF_PROB]):
            print("The method __GradHermiteF cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if k==0:
                dP = self.__HermiteF(r, N, isConstructingVandermonde, recursion_storage)
            else:
                # This part does not exploit the storage of previously computed polynomials
                # in case the Vandermonde matrix is under construction
                if (N-k < 0):
                    dP = np.zeros(r.shape)
                else:
                    dP = np.zeros(r.shape)
                    for i in range(k+1):
                        fact1 = np.exp( gammalnF(k+1) - gammalnF(i+1) - gammalnF(k-i+1) )
                        fact2 = np.exp( gammalnF(N+1) - gammalnF(N-k+i+1) )
                        dP += fact1 * (-1.)**i * 2.**((k-i)/2.) * np.sqrt(fact2) * \
                              self.__HermiteF(r,N-k+i) * self.__HermiteP_Prob(r,i)
            return dP
    
    def __GradHermiteP_Prob(self,r,N,k, isConstructingVandermonde=False, recursion_storage=None):
        """
        Compute the k-th derivative of the N-th Hermite Probabilistic Polynomial
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param k: order of the derivative
        :type k: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly != HERMITEP_PROB):
            print("The method __GradHermiteP_Prob cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if (N-k < 0):
                dP = np.zeros(r.shape)
            else:
                fact = np.exp( gammalnF(N+1) - gammalnF(N-k+1) )
                dP = fact * self.__HermiteP_Prob(r,N-k, isConstructingVandermonde, recursion_storage)
            return dP
    
    def __GradJacobiP(self,r, N, k, 
                      isConstructingVandermonde=False,
                      recursion_storage=None):
        """
        Evaluate the kth-derivative of the Jacobi polynomial of type (alpha,beta)>-1,
        at points r for order N and returns dP[1:length(r))]

        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param k: order of the derivative
        :type k: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly != JACOBI):
            print("The method __GradJacobiP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
                return self.__GradChebyshevP(r, N, k, 0, 
                                             isConstructingVandermonde,
                                             recursion_storage)
            else:
                r = np.array(r)
                if (N >= k):
                    dP = gammaF(alpha+beta+N+1.+k)/(2.**k * gammaF(alpha+beta+N+1.)) * \
                         np.sqrt(
                             self.__GammaJacobiP(N-k,alpha+k,beta+k)/ \
                             self.__GammaJacobiP(N,alpha,beta)) * \
                             self.__JacobiP(r,N-k,alpha+k,beta+k,
                                            isConstructingVandermonde,
                                            recursion_storage)
                else:
                    dP = np.zeros(r.shape)
                return dP
    
    def __GradChebyshevP(self,r, N, k, method=0, 
                         isConstructingVandermonde=False,
                         recursion_storage=None):
        """
        Returns the k-th derivative of the Chebyshev polynomial of order N at
        points r.

        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param k: order of the derivative
        :type k: int
        :param method: 0 -> Matrix multiplication, 1 -> Fast Chebyshev Transform
        :type method: int
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly != JACOBI):
            print("The method __GradChebyshevP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
                dP = np.zeros((N+1))
                if ( k == 0 ):
                    dP = self.__ChebyshevP(r,N, isConstructingVandermonde, recursion_storage)
                elif ( method == 0 ):
                    D = PolynomialDerivativeMatrix(r,k)
                    P = self.__ChebyshevP(r,N, isConstructingVandermonde, recursion_storage)
                    dP = np.dot(D,P)
                return dP
    
    def __ORTHPOL(self,r,N,k=0, old1=None, old2=None):
        """
        dP = GradORTHPOL(r,N,k)
        Returns the 0-th derivative of the N-th orthogonal polynomial defined for the supplied measure
        
        .. note:: For further information see :cite:`Gautschi1994`
        """
        (alphaCap,betaCap) = self.orthpol_rec_coeffs(N)

        # Evaluate Polynomial
        rs = np.reshape(r,(len(r),1))
        if N >= 0:
            out = np.ones(rs.shape)
        if N >= 1:
            old1 = out
            out = (rs - alphaCap[0])
        for i in range(2,N+1):
            old2 = old1
            old1 = out
            out = (rs - alphaCap[i-1]) * old1 - betaCap[i-1] * old2
        
        return out
    
    def __FourierP(self, x, N, old1=None, old2=None):
        """
        Construct the N-th Fourier function (1,cos(x),sin(x),cos(2*x),...)
        
        :param ndarray x: points between [0,2*pi] where to evaluate the function
        :param int N: function number
        :return: ndarray :py:data:`P` function evaluated at the point
        """
        if N == 0:
            P = np.ones( x.shape )
        elif N % 2 == 0:
            P = -np.sin( N/2 * x )
        else:
            P = np.cos( (N+1)/2 * x )
        return P
    
    def __GradFourierP(self, x, N, k, old1=None, old2=None):
        """
        Construct the k-th derivative of the N-th Fourier function (1,cos(x),sin(x),cos(2*x),...)
        
        :param ndarray x: points between [0,2*pi] where to evaluate the function
        :param int N: function number
        :param int k: derivative order
        :return: ndarray :py:data:`P` function evaluated at the point
        """
        if k == 0:
            P = self.__FourierP(x, N)
        else:
            if N == 0:
                P = np.zeros( x.shape )
            elif N % 2 == 0:
                P = -1 * ( - N/2. )**k * self.__FourierP(x, N - (k % 2))
            else:
                P = ( - (N+1)/2. )**k * self.__FourierP(x, N + (k % 2))
        return P

    def GradEvaluate(self, r, N, k=0, norm=True, 
                     isConstructingVandermonde=False,
                     recursion_storage=None):
        """
        Evaluate the ``k``-th derivative of the ``N``-th order polynomial at points ``r``
        
        :param r: set of points on which to evaluate the polynomial
        :type r: 1d-array or float
        :param N: order of the polynomial
        :type N: int
        :param k: order of the derivative [default=0]
        :type k: int
        :param norm: whether to return normalized (True) or non normalized (False) polynomials
        :type norm: bool
        :param isConstructingVandermonde: whether to store values in the recursion relation for the construciton of the Vandermonde matrix
        :type isConstructingVandermonde: bool
        :param recursion_storage: storage needed for the recursion whether one is computing the Vandermonde matrix (``isConstructingVandermonde==True``)
        :type recursion_storage: ndarray len(r) x 2
        :return: polynomial evaluated on ``r``
        :rtype: 1d-ndarray
        """
        if (self.poly == JACOBI):
            p = self.__GradJacobiP(r, N, k, 
                                   isConstructingVandermonde=isConstructingVandermonde, 
                                   recursion_storage=recursion_storage)
        elif (self.poly == HERMITEP):
            p = self.__GradHermiteP(r, N, k, 
                                   isConstructingVandermonde=isConstructingVandermonde, 
                                   recursion_storage=recursion_storage)
        elif (self.poly == HERMITEF):
            p = self.__GradHermiteF(r, N, k, 
                                   isConstructingVandermonde=isConstructingVandermonde, 
                                   recursion_storage=recursion_storage)
        elif (self.poly == HERMITEP_PROB):
            p = self.__GradHermiteP_Prob(r, N, k, 
                                         isConstructingVandermonde=isConstructingVandermonde, 
                                         recursion_storage=recursion_storage)
        elif (self.poly == HERMITEF_PROB):
            p = self.__GradHermiteF(r/np.sqrt(2.), N, k,
                                    isConstructingVandermonde=isConstructingVandermonde, 
                                    recursion_storage=recursion_storage)
        elif (self.poly == LAGUERREP):
            p = self.__GradLaguerreP(r, N, k, 
                                     isConstructingVandermonde=isConstructingVandermonde, 
                                     recursion_storage=recursion_storage)
        elif (self.poly == LAGUERREF):
            p = self.__GradLaguerreF(r, N, k, 
                                     isConstructingVandermonde=isConstructingVandermonde, 
                                     recursion_storage=recursion_storage)
        elif (self.poly == ORTHPOL):
            if k != 0:
                print("Spectral1D: Error. Derivatives of Polynomials obtained using ORTHPOL package are not implemented")
                return
            p = self.__ORTHPOL(r,N).flatten()
        elif (self.poly == FOURIER):
            p = self.__GradFourierP(r,N,k)
        
        if (self.poly == JACOBI):
            if not norm:
                p *= math.sqrt(self.Gamma(N))
        else:
            if norm:
                p /= math.sqrt(self.Gamma(N))
        
        return p
    
    def __GammaLaguerreF(self,N,alpha=None):
        """
        __GammaLaguerreF(): evaluate the normalization constant for the ``N``-th order Laguerre function
        
        Syntax:
            ``g = __GammaLaguerreF(N,[alpha=None])``
        
        Input:
            * ``N`` = (int) order of the polynomial
            * ``alpha'' = (optional,float) Laguerre constant
        
        Output:
            * ``g`` = Normalization constant
        """
        if (self.poly != LAGUERREF):
            print("The method __GammaLaguerreF cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if (alpha is None):
                # Unpack parameters
                (alpha,) = self.params
            g = gammaF(N+alpha+1.)/gammaF(N+1)
            return g
    
    def __GammaLaguerreP(self,N,alpha=None):
        """
        __GammaLaguerreP(): evaluate the normalization constant for the ``N``-th order Laguerre polynomial
        
        Syntax:
            ``g = __GammaLaguerreP(N,[alpha=None])``
        
        Input:
            * ``N`` = (int) order of the polynomial
            * ``alpha`` = (optional,float) Laguerre constant
        
        Output:
            * ``g`` = Normalization constant
        """
        if (self.poly != LAGUERREP):
            print("The method __GammaLaguerreP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if (alpha is None):
                # Unpack parameters
                (alpha,) = self.params
            g = gammaF(N+alpha+1.)/gammaF(N+1)
            return g
        
    def __GammaJacobiP(self,N,alpha=None,beta=None):
        """
        gamma = GammaJacobiP(alpha,beta,N)
        Generate the normalization constant for the
        Jacobi Polynomial (alpha,beta) of order N.
        """
        if (self.poly != JACOBI):
            print("The method __GammaJacobiP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            if (alpha is None) and (beta is None):
                # Unpack parameters
                alpha,beta = self.params
            g = 2**(alpha+beta+1.) * (gammaF(N+alpha+1.)*gammaF(N+beta+1.)) / (factorial(N,exact=True) * (2.*N + alpha + beta + 1.)*gammaF(N+alpha+beta+1.))
            return g
        
    def __GammaHermiteP(self,N):
        """
        Returns the normalization contant for the Probabilistic Hermite Physicist
        polynomial of order N
        """
        if (self.poly != HERMITEP):
            print("The method __GammaHermiteP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            return math.sqrt(np.pi) * 2.**N * factorial(N,exact=True)
        
    def __GammaHermiteF(self,N):
        """
        Returns the normalization contant for the Hermite function of order N
        """
        if (self.poly not in  [HERMITEF, HERMITEF_PROB]):
            print("The method __GammaHermiteF cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            return np.sqrt(np.pi)
    
    def __GammaHermiteP_Prob(self,N):
        """
        Returns the normalization contant for the Probabilistic Hermite 
        Probabilistic polynomial of order N
        """
        if (self.poly != HERMITEP_PROB):
            print("The method __GammaHermiteP_Prob cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            return factorial(N,exact=True)
    
    def __GammaORTHPOL(self,N):
        """
        Returns the normalization constant for the generic ORTHPOL polynomial of order N.
        The computation is performed using Gauss Quadrature.
        """
        if (self.poly != ORTHPOL):
            print("The method __GammaORTHPOL cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            (x,w,ierr) = self.__ORTHPOL_GQ(N)
            if ierr != 0:
                print("Error in ORTHPOL GaussQuadrature.")
            P = self.__ORTHPOL(x,N)
            return np.dot(P.T**2.,w)

    def __GammaFourier(self, N):
        """
        Returns the normalization constant for the N-th Fourier function
        """
        if N == 0:
            return 2.*np.pi
        else:
            return np.pi
    
    def Gamma(self,N):
        """
        Gamma(): returns the normalization constant for the N-th polynomial
        
        Syntax:
            ``g = Gamma(N)``
        
        Input:
            * ``N`` = polynomial order
        
        Output:
            * ``g`` = normalization constant
        """
        if (self.poly == JACOBI):
            return self.__GammaJacobiP(N)
        elif (self.poly == HERMITEP):
            return self.__GammaHermiteP(N)
        elif (self.poly == HERMITEF):
            return self.__GammaHermiteF(N)
        elif (self.poly == HERMITEP_PROB):
            return self.__GammaHermiteP_Prob(N)
        elif (self.poly == HERMITEF_PROB):
            return self.__GammaHermiteF(N) # Unsure about it...
        elif (self.poly == LAGUERREP):
            return self.__GammaLaguerreP(N)
        elif (self.poly == LAGUERREF):
            return self.__GammaLaguerreF(N)
        elif (self.poly == ORTHPOL):
            return self.__GammaORTHPOL(N)
        elif (self.poly == FOURIER):
            return self.__GammaFourier(N)
        else:
            print("[Spectral1D]: Gamma function not implemented yet for the selected polynomial %s" % self.poly)
    
    def __GradJacobiVandermondeORTHPOL(self,r,N,k,norm):
        if (self.poly != JACOBI):
            raise ImportError( "The method __GradJacobiP cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly )
        else:
            if not ORTHPOL_SUPPORT:
                raise ImportError("The orthpol package is not installed on this machine.")
            # Unpack parameters
            alpha,beta = self.params
            
            (al,be,ierr) = orthpol.drecur(N+1,6,alpha,beta)
            dV = orthpol.vandermonde(r,N,al,be)
            
            if k > 0:
                for n in range(N+1):
                    if (n >= k):
                        dV[:,n] = gammaF(alpha+beta+N+1.+k)/(2.**k * gammaF(alpha+beta+N+1.)) * np.sqrt(self.__GammaJacobiP(N-k,alpha+k,beta+k)/self.__GammaJacobiP(N,alpha,beta)) * dV[:,n]
                    else:
                        dV[:,n] = 0.
            return dV

    def GradVandermonde(self,r,N,k=0,norm=True):
        return self.GradVandermonde1D(r,N,k,norm)
    
    def GradVandermonde1D(self,r,N,k=0,norm=True):
        """
        GradVandermonde1D(): Initialize the ``k``-th gradient of the modal basis ``N`` at ``r``
        
        Syntax:
            ``V = GradVandermonde1D(r,N,k,[norm])``
        
        Input:
            * ``r`` = (1d-array,float) set of ``M`` points on which to evaluate the polynomials
            * ``N`` = (int) maximum order in the vanermonde matrix
            * ``k`` = (int) derivative order [default=0]
            * ``norm`` = (optional,boolean) True -> orthonormal polynomials, False -> non orthonormal polynomials
        
        Output:
            * ``V`` = (2d-array(``MxN``),float) Generalized Vandermonde matrix
        """

        recursion_storage = np.zeros((r.shape[0],2))

        DVr = np.zeros((r.shape[0],N+1))
        for i in range(0,N+1):
            DVr[:,i] = self.GradEvaluate(r, i, k, norm, 
                                         isConstructingVandermonde=True,
                                         recursion_storage=recursion_storage)
        
        return DVr
        
    def AssemblyDerivativeMatrix(self, x, N, k):
        """
        AssemblyDerivativeMatrix(): Assemble the k-th derivative matrix using polynomials of order N.
        
        Syntax:
            ``Dk = AssemblyDerivativeMatrix(x,N,k)``
        
        Input:
            * x = (1d-array,float) Set of points on which to evaluate the polynomials
            * N = (int) maximum order in the vanermonde matrix
            * k = (int) derivative order
        
        Output:
            * Dk = Derivative matrix
        
        Description:
            This function performs ``D = linalg.solve(V.T, Vx.T)`` where ``V`` and ``Vx`` are a Generalized Vandermonde Matrix and its derivative respectively.
        
        Notes:
            For Chebyshev Polynomial, this function refers to the recursion form implemented in ``PolynomialDerivativeMatrix``
        """
        # Unpack parameters
        if (self.poly == JACOBI):
            alpha,beta = self.params
        
        if (self.poly == JACOBI) and ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
            return self.PolynomialDerivativeMatrix(x,k)
        else:
            V = self.GradVandermonde1D(x, N, 0)
            Vx = self.GradVandermonde1D(x, N ,1)
            D = LA.solve(V.T, Vx.T)
            D = D.T
            Dk = np.asarray(np.mat(D)**k)
            return Dk

    def interpolate(self, x, f, xi, order):
        """
        Interpolates function values ``f`` from points ``x`` to points ``xi`` using Forward and Backward Polynomial Transform

        :param 1d-array,float x: set of ``N`` original points where ``f`` is evaluated
        :param 1d-array,float f: set of ``N`` function values
        :param 1d-array,float xi: set of ``M`` points where the function is interpolated
        :param int order: order of polynomial interpolation
        :returns: ``fi`` (1d-array,float) set of ``M`` function values

        ..note:: this is the same of calling :func:`Ploy1D.PolyInterp`
        .. seealso:: PolyInterp
        """
        return self.PolyInterp(x, f, xi, order)
            
    def PolyInterp(self, x, f, xi, order):
        """
        PolyInterp(): Interpolate function values ``f`` from points ``x`` to points ``xi`` using Forward and Backward Polynomial Transform
        
        Syntax:
            ``fi = PolyInterp(x, f, xi)``
        
        Input:
            * ``x`` = (1d-array,float) set of ``N`` original points where ``f`` is evaluated
            * ``f`` = (1d-array,float) set of ``N`` function values
            * ``xi`` = (1d-array,float) set of ``M`` points where the function is interpolated
            * ``order`` = (integer) order of polynomial interpolation
        
        Output:
            * ``fi`` = (1d-array,float) set of ``M`` function values
        
        Notes:
            
        """
        fhat = self.DiscretePolynomialTransform(x,f,order)
        return self.InverseDiscretePolynomialTransform(xi,fhat,order)
        
    def LegendreDerivativeCoefficients(self,fhat):
        """
        LegendreDerivativeCoefficients(): computes the Legendre coefficients of the derivative of a function
        
        Syntax:
            ``dfhat = LegendreDerivativeCoefficients(fhat)``
        
        Input:
            * ``fhat`` = (1d-array,float) list of Legendre coefficients of the original function
        
        Output:
            * ``dfhat`` = (1d-array,float) list of Legendre coefficients of the derivative of the original function
        
        Notes:
            Algorithm (4) from :cite:`Kopriva2009`
        """
        if (self.poly != JACOBI):
            print("The method cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,0.) and AlmostEqual(beta,0.) ):
                N = fhat.shape[0]-1
                dfhat = np.zeros((N+1))
                dfhat[N-1] = (2.*N - 1.) * fhat[N]
                for k in range(N-2, -1, -1):
                    dfhat[k] = (2.*k + 1.) * (fhat[k+1] + dfhat[k+2]/(2.*k + 5.) )
                return dfhat
    
    def ChebyshevDerivativeCoefficients(self,fhat):
        """
        ChebyshevDerivativeCoefficients(): computes the Chebyshev coefficients of the derivative of a function
        
        Syntax:
            ``dfhat = ChebyshevDerivativeCoefficients(fhat)``
        
        Input:
            * ``fhat`` = (1d-array,float) list of Chebyshev coefficients of the original function
        
        Output:
            * ``dfhat`` = (1d-array,float) list of Chebyshev coefficients of the derivative of the original function
        
        Notes:
            Algorithm (5) from :cite:`Kopriva2009`
        """
        if (self.poly != JACOBI):
            print("The method cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
                N = fhat.shape[0]-1
                dfhat = np.zeros((N+1))
                dfhat[N-1] = (2.*N) * fhat[N]
                for k in range(N-2, 0, -1):
                    dfhat[k] = 2. * (k + 1.) * fhat[k+1] + dfhat[k+2]
                dfhat[0] = fhat[1] + dfhat[2]/2.
                return dfhat

    def project(self, r, f, N):
        """
        Computes the Discrete Polynomial Transform of function values f, i.e. project on the space spanned by the selected polynomials up to order ``N``.
                
        :param r: set of points on which to the polynomials are evaluated
        :type r: 1d-array or float
        :param f: function values
        :type f: 1d-array or float
        :param int N: maximum order in the generalized vanermonde matrix
        :return: projection coefficients 
        :rtype: 1d-ndarray containing the projection coefficients. If ``len(r) == N+1`` exact projection is used. If ``len(r) != N+1`` the least square solution is obtained by ``numpy.linalg.lstsq``.
        
        .. note:: If the Chebyshev polynomials are chosen and ``r`` contains Chebyshev-Gauss-Lobatto points, the Fast Chebyshev Transform is used. Otherwise uses the Generalized Vandermonde Matrix in order to transform from physical space to transform space.
        .. note:: this is the same of calling :func:`DiscretePolynomialTransform`
        
        .. seealso:: DiscretePolynomialTransform
        .. seealso:: numpy.linalg.lstsq
        """
        return self.DiscretePolynomialTransform(r, f, N)

    def DiscretePolynomialTransform(self,r, f, N):
        """
        Computes the Discrete Polynomial Transform of function values f, i.e. project on the space spanned by the selected polynomials up to order ``N``.
                
        :param r: set of points on which to the polynomials are evaluated
        :type r: 1d-array or float
        :param f: function values
        :type f: 1d-array or float
        :param int N: maximum order in the generalized vanermonde matrix
        :return: projection coefficients 
        :rtype: 1d-ndarray containing the projection coefficients. If ``len(r) == N+1`` exact projection is used. If ``len(r) != N+1`` the least square solution is obtained by ``numpy.linalg.lstsq``.
        
        .. note:: If the Chebyshev polynomials are chosen and ``r`` contains Chebyshev-Gauss-Lobatto points, the Fast Chebyshev Transform is used. Otherwise uses the Generalized Vandermonde Matrix in order to transform from physical space to transform space.
        .. note:: If the Fourier expansion is chosen, no check on ``r`` and ``N`` is done. Though the periodic function is assumed to be evaluated over ``N+1 == len(f)`` equidistant points over [0, 2pi-2pi/(N+2)]. The returned coefficients ``a`` are ordered such that the following expansion is obtained: :math:`f(x)=a_0 + \sum_{i=1}^M a_{2i-1} cos( (2i-1) x) - \sum_{i=1}^M a_{2i} sin( 2i x )`, where :math:`M=\lfloor N/2 \rfloor + 1` if ``N`` is even, :math:`M=(N+1)/2` if ``N`` is odd.
        .. note:: this is the same of calling :py:method:`DiscretePolynomialTransform`
        
        .. seealso:: FastChebyshevTransform
        .. seealso:: numpy.linalg.lstsq
        """
        # Unpack parameters
        if (self.poly == JACOBI):
            alpha,beta = self.params
        
        if (self.poly == JACOBI) and ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
            # Chebyshev case
            rr,w = self.__JacobiCGL(N)
            equal = True
            for i in range(0,N+1):
                equal = AlmostEqual(r[i],rr[i])
                if (not equal) :
                    break
            if equal:
                # Chebyshev-Gauss-Lobatto points. Use FCS.
                return self.FastChebyshevTransform(f)
        elif (self.poly == FOURIER):
            N = len(f)
            fhatc = FFT.rfft( f ) # Complex
            fhatc = np.vstack( (fhatc.real, fhatc.imag) ).T
            fhat = 2. * np.sqrt(np.pi) / float(N) * np.hstack( ( fhatc[0,0]/np.sqrt(2), fhatc[1:,:].flatten()) )
            return fhat[:N]
        # Use the generalized vandermonde matrix
        V = self.GradVandermonde1D(r, N, 0)
        if ( V.shape[0] == V.shape[1] ):
            fhat = LA.solve(V, f)
        else:
            (fhat, residues, rank, s) = LA.lstsq(V, f)
        return fhat
        
    def InverseDiscretePolynomialTransform(self, r, fhat, N):
        """
        Computes the nodal values from the modal form fhat.
                
        :param x: set of points on which to the polynomials are evaluated
        :type x: 1d-array or float
        :param fhat: list of Polynomial coefficients
        :type fhat: 1d-array or float
        :param N: maximum order in the generalized vanermonde matrix
        :type N: int
        :return: function values
        :rtype: 1d-array or float
            
        .. note:: If the Chebyshev polynomials are chosen and r contains Chebyshev-Gauss-Lobatto points, the Inverse Fast Chebyshev Transform is used. Otherwise uses the Generalized Vandermonde Matrix in order to transform from transform space to physical space.
        
        .. seealso:: InverseFastChebyshevTransform
        
        """
        # Unpack parameters
        if (self.poly == JACOBI):
            alpha,beta = self.params
        
        if (self.poly == JACOBI) and ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
            # Chebyshev case
            rr,w = self.__JacobiCGL(N)
            equal = True
            for i in range(0,N+1):
                equal = AlmostEqual(r[i],rr[i])
                if (not equal) :
                    break
            if equal:
                # Chebyshev-Gauss-Lobatto points. Use FCS.
                return self.InverseFastChebyshevTransform(fhat)
        # Use the generalized vandermonde matrix
        V = self.GradVandermonde1D(r, N, 0)
        f = np.dot(V,fhat)
        return f
        
    def FastChebyshevTransform(self,f):
        """
        FastChebyshevTransform(): Returns the coefficients of the Fast Chebyshev Transform.
        
        Syntax:
            ``fhat = FastChebyshevTransform(f)``
        
        Input:
            * ``f`` = (1d-array,float) function values
        
        Output:
            * ``fhat`` = (1d-array,float) list of Polynomial coefficients
        
        .. warning:: It is assumed that the values f are computed at Chebyshev-Gauss-Lobatto points.
        
        .. note:: If f is odd, the vector is interpolated to even Chebyshev-Gauss-Lobatto points.
        .. note:: Modification of algorithm (29) from :cite:`Kopriva2009`
        """
        if (self.poly != JACOBI):
            print("The method cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
                N = f.shape[0]-1
                # Create Even function
                fbar = np.hstack([f[::-1], f[1:N]])
                # Transform
                fhat = FFT.ifft(fbar)
                fhat = np.hstack([fhat[0], 2*fhat[1:N], fhat[N]])
                return fhat
        
    def InverseFastChebyshevTransform(self,fhat):
        """
        InverseFastChebyshevTransform(): Returns the coefficients of the Inverse Fast Chebyshev Transform.
        
        Syntax:
            ``f = InverseFastChebyshevTransform(fhat)``
        
        Input:
            * ``fhat`` = (1d-array,float) list of Polynomial coefficients
        
        Output:
            * ``f`` = (1d-array,float) function values
        
        .. note:: If f is odd, the vector is padded with a zero value (highest freq.)
        .. note:: Modification of algorithm (29) from :cite:`Kopriva2009`
        """
        if (self.poly != JACOBI):
            print("The method cannot be called with the actual type of polynomials. Actual type: '%s'" % self.poly)
        else:
            # Unpack parameters
            alpha,beta = self.params
            if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
                N = fhat.shape[0]
                # Sort values out for FFT
                fhat = np.hstack([fhat[0], np.hstack([fhat[1:N-1], fhat[N-1]*2, fhat[-2:0:-1] ])*0.5 ])
                f = FFT.fft(fhat)
                f = f[N-1::-1]
                f = np.real(f)
                return f
    
    def GramSchmidt(self, p, N, w):
        """
        GramSchmidt(): creates a Generalized Vandermonde Matrix of orthonormal polynomials with respect to the weights ``w``
        
        Syntax:
            ``V = GramSchmidt(p, N, w)``
        
        Input:
            * ``p`` = (1d-array,float) points at which to evaluate the new polynomials
            * ``N`` = (int) the maximum order of the polynomials
            * ``w`` = (1d-array,float) weights to be used for the orthogonoalization
        
        Output:
            * ``V`` = Generalized Vandermonde Matrix containing the new orthogonalized polynomials
        
        Description:
            Takes the points where the polynomials have to be evaluated and computes a Generalized Gram Schmidth procedure, where a weighted projection is used. If ``w==1`` then the usual inner product is used for the orthogonal projection.
        """
        # Evaluate Vandermonde matrix 
        V = np.vander(p,N+1)
        V  = V[:,::-1]
        
        # Evaluate Gram-Shmidt orthogonalization
        gs = np.zeros(N+1) # Vector of gamma values for the new polynomials
        for k in range(0,N+1):
            for j in range(0,N+1):
                for i in range(0,j):
                    # Use numerical quadrature to evaluate the projection
                    V[:,j] = V[:,j] - np.dot(V[:,j] * V[:,i], w) / np.sqrt(gs[i]) * V[:,i]
                # Compute the gamma value for the new polynomial
                gs[j] = np.dot(V[:,j]*V[:,j],w)
                # Normalize the vector if required
                V[:,j] = V[:,j]/np.sqrt(gs[j])
        
        return V

    
def AlmostEqual(a,b):
    """
    b = AlmostEqual(a,b)
    Test equality of two floating point numbers. Returns a boolean.
    """
    eps = np.finfo(np.float64).eps
    if ((a == 0) or (b == 0)):
        if (abs(a-b) <= 2*eps):
            return True
        else:
            return False
    else:
        if ( (abs(a-b) <= eps*abs(a)) and (abs(a-b) <= eps*abs(b)) ):
            return True
        else:
            return False

def gqu(N,norm=True):
    """
    GQU(): function for generating 1D Gaussian quadrature rules for unweighted integral over [-1,1] (Gauss-Legendre)

    Note:: Max ``N'' is 25

    Syntax:
        (n,w) = GQU(l)
    Input:
        l = level of accuracy of the quadrature rule
    Output:
        n = nodes
        w = weights
    """
    (x,w) = SG.GQU(N)
    if (N % 2 == 0):
        x = np.asarray(x)
        x = np.hstack([1-x[::-1],x])
        w = np.asarray(w)
        w = np.hstack([w[::-1],w])
    else:
        x = np.asarray(x)
        x = np.hstack([1-x[1:][::-1],x])
        w = np.asarray(w)
        w = np.hstack([w[1:][::-1],w])
    
    x = x*2. - 1.
    if not norm: w *= 2.
    return (x,w)

def gqn(N):
    """
    GQN(): function for generating 1D Gaussian quadrature for integral with Gaussian weight (Gauss-Hermite)

    Note:: Max ``N'' is 25

    Syntax:
        (n,w) = GQU(l)
    Input:
        l = level of accuracy of the quadrature rule
    Output:
        n = nodes
        w = weights
    """
    (x,w) = SG.GQN(N)
    if (N % 2 == 0):
        x = np.asarray(x)
        x = np.hstack([-x[::-1],x])
        w = np.asarray(w)
        w = np.hstack([w[::-1],w])
    else:
        x = np.asarray(x)
        x = np.hstack([-x[1:][::-1],x])
        w = np.asarray(w)
        w = np.hstack([w[1:][::-1],w])
    return (x,w)

def kpu(N,norm=True):
    """
    KPU(): function for generating 1D Nested rule for unweighted integral over [-1,1]

    Note:: Max ``N'' is 25

    Syntax:
        (n,w) = GQU(l)
    Input:
        l = level of accuracy of the quadrature rule
    Output:
        n = nodes
        w = weights
    """
    (x,w) = SG.KPU(N)
    x = np.asarray(x)
    x = np.hstack([1-x[1:][::-1],x])
    x = x*2. - 1.
    w = np.asarray(w)
    w = np.hstack([w[1:][::-1],w])
    if not norm: w *= 2.
    return (x,w)

def kpn(N):
    """
    KPN(): function for generating 1D Nested rule for integral with Gaussian weight

    Note:: Max ``N'' is 25

    Syntax:
        (n,w) = GQU(l)
    Input:
        l = level of accuracy of the quadrature rule
    Output:
        n = nodes
        w = weights
    """
    (x,w) = SG.KPN(N)
    x = np.asarray(x)
    x = np.hstack([-x[1:][::-1],x])
    w = np.asarray(w)
    w = np.hstack([w[1:][::-1],w])
    return (x,w)

def cc(N,norm=True):
    """
    cc(): function for generating 1D Nested Clenshaw-Curtis [-1,1]

    Syntax:
        (n,w) = cc(N)
    Input:
        N = level of accuracy of the quadrature rule
    Output:
        n = nodes
        w = weights
    """
    (x,w) = SG.CC(N)
    x = np.hstack([-x[1:][::-1],x])
    w = np.hstack([w[1:][::-1],w])
    if norm: w /= np.sum(w)
    return (x,w)

def fej(N,norm=True):
    """
    fej(): function for generating 1D Nested Fejer's rule [-1,1]

    Syntax:
        (n,w) = fej(N)
    Input:
        N = level of accuracy of the quadrature rule
    Output:
        n = nodes
        w = weights
    """
    (x,w) = SG.FEJ(N)
    x = np.hstack([-x[1:][::-1],x])
    w = np.hstack([w[1:][::-1],w])
    if norm: w /= np.sum(w)
    return (x,w)

def nestedgauss(N,norm=True):
    """
    nestedgauss(): function for generating 1D Nested rule for integral with Uniform weight with 2**l scaling

    Syntax:
        (n,w) = nestedgauss(N)
    Input:
        N = level of accuracy of the quadrature rule
    Output:
        n = nodes
        w = weights
    """
    (x,w) = SG.NESTEDGAUSS(N)
    x = np.array(x)
    w = np.array(w)
    if norm: w /= np.sum(w)
    return (x,w)

def nestedlobatto(N,norm=True):
    """
    nestedlobatto(): function for generating 1D Nested rule for integral with Uniform weight with 2**l scaling

    Syntax:
        (n,w) = nestedlobatto(N)
    Input:
        N = level of accuracy of the quadrature rule
    Output:
        n = nodes
        w = weights
    """
    (x,w) = SG.NESTEDLOBATTO(N)
    x = np.array(x)
    w = np.array(w)
    if norm: w /= np.sum(w)
    return (x,w)

QUADS = {GQU: gqu,
         GQN: gqn,
         KPU: kpu,
         KPN: kpn,
         CC: cc,
         FEJ: fej,
         NESTEDGAUSS: nestedgauss,
         NESTEDLOBATTO: nestedlobatto}
