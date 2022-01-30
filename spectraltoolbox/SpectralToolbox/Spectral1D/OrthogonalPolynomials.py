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

import numpy as np
from numpy import linalg as LA
from numpy import fft as FFT
import math
from scipy.special import gamma as gammaF
from scipy.special import gammaln as gammalnF
from scipy.special import factorial

import orthpol_light
import polymod
try:
    import orthpol
    ORTHPOL_SUPPORT = True
except ImportError:
    ORTHPOL_SUPPORT = False

from SpectralToolbox.Spectral1D.Constants import *
from SpectralToolbox.Spectral1D.AbstractClasses import *

__all__ = ['JacobiPolynomial', 'LaguerrePolynomial', 'HermitePhysicistsPolynomial',
           'HermiteProbabilistsPolynomial', 'GenericOrthogonalPolynomial']

class JacobiPolynomial(OrthogonalPolynomial):
    r""" Construction of Jacobi polynomials

    Args:
      alpha (float): parameter :math:`\alpha > -1`
      beta (float): parameter :math:`\beta > -1`.
      span (list): span of the domain
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.
    """

    eps = orthpol_light.d1mach(3)
    
    def __init__(self, alpha, beta, span=None, normalized=None):
        super(JacobiPolynomial,self).__init__(normalized)
        if alpha <= -1 or beta <= -1:
            raise ValueError("Values alpha and/or beta out of range")
        self.alpha = alpha
        self.beta = beta
        self.bspan = [-1.,1.]
        if span == None:
            self.span = self.bspan
        elif len(span) != 2:
            raise ValueError("Span must be a list of len 2")
        elif np.any( np.isinf( span ) ):
            raise ValueError("Infite not allowed in span for Jacobi polynomial")
        else:
            self.span = span

    def RecursionCoeffs(self, N, alpha=None, beta=None):
        r""" Get coefficients for the Jacobi polynomials up to order ``N``

        .. seealso:: :func:`OrthogonalPolynomial.RecursionCoeffs`
        """
        if alpha == None and beta == None:
            # Unpack parameters
            alpha = self.alpha
            beta = self.beta
        elif None in [alpha,beta]:
            raise ValueError("alpha and beta must be either both None or "+
                             "both initialized")
        if alpha <= -1. or beta <= -1.:
            raise ValueError("alpha and beta must be bigger than -1")
        if ( np.isclose(alpha, 0.0, rtol=self.eps, atol=self.eps) and
             np.isclose(beta, 0.0, rtol=self.eps, atol=self.eps)):
            # Legendre polynomials
            (a,b,ierr) = orthpol_light.drecur(N+1, 1, 0, 0)
        elif ( np.isclose(alpha, -.5, rtol=self.eps, atol=self.eps) and
               np.isclose(beta, -.5, rtol=self.eps, atol=self.eps)):
            # Chebyshev first kind
            (a,b,ierr) = orthpol_light.drecur(N+1, 3, 0, 0)
        elif ( np.isclose(alpha, .5, rtol=self.eps, atol=self.eps) and
               np.isclose(beta, .5, rtol=self.eps, atol=self.eps)):
            # Chebyshev second kind
            (a,b,ierr) = orthpol_light.drecur(N+1, 4, 0, 0)
        else:
            # Jacobi polynomials
            (a,b,ierr) = orthpol_light.drecur(N+1, 6, alpha, beta)
        if ierr != 0: raise RuntimeError("ORTHPOL drecur: error flag %d" % ierr)
        return (a,b)
    
    def GaussQuadrature(self, N, norm=False):
        r""" Jacobi Gauss quadrature points

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        (x,w,ierr) = orthpol_light.dgauss(a, b)
        if ierr != 0: raise RuntimeError("ORTHPOL dgauss: error flag %d" % ierr)
        # Normalization and rescaling
        w /= np.sum(w)
        if not norm:
            w *= self.Gamma(0)
        return (x,w)

    def GaussLobattoQuadrature(self, N, norm=False):
        r""" Jacobi Gauss Lobatto quadrature points

        .. seealso:: :func:`OrthogonalPolynomial.GaussLobattoQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        (x,w,ierr) = orthpol_light.dlob(a, b, -1., 1.)
        if ierr != 0: raise RuntimeError("ORTHPOL dlob: error flag %d" % ierr)
        # Normalization and rescaling
        w /= np.sum(w)
        if not norm:
            w *= self.Gamma(0)
        return (x,w)

    def GaussRadauQuadrature(self, N, norm=False):
        r""" Jacobi Gauss Radau quadrature points

        .. seealso:: :func:`OrthogonalPolynomial.GaussRadauQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        (x,w,ierr) = orthpol_light.dradau(a, b, -1.)
        if ierr != 0: raise RuntimeError("ORTHPOL dradau: error flag %d" % ierr)
        # Normalization and rescaling
        w /= np.sum(w)
        if not norm:
            w *= self.Gamma(0)
        return (x,w)

    def Evaluate(self, x, N, alpha=None, beta=None, norm=True):
        r""" Evaluate the ``N``-th order Jacobi polynomial

        Args:
          x ((:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomial
          N (int): order of the polynomial

        Returns:
          (:class:`ndarray<ndarray>` [``m``]) -- polynomial evaluated on the
            ``x`` points.
        """
        if self.normalized is not None: norm = self.normalized
        (a,b) = self.RecursionCoeffs(N, alpha, beta)
        p = polymod.polyeval(x, N, a, b, norm)
        if not norm:
            pend = polymod.polyeval(np.array([1.]), N, a, b, norm)
            p *= (gammaF(N+alpha+1)/(gammaF(N+1)*gammaF(alpha+1))) / pend
        return p

    def GradEvaluate(self, r, N, k=0, norm=True):
        r""" ``k``-th derivative of the ``N``-th order Jacobi polynomial.
        
        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        # Unpack parameters
        if self.normalized is not None: norm = self.normalized
        alpha = self.alpha
        beta = self.beta
        r = np.array(r)
        if (N >= k):
            num = gammaF(alpha+beta+N+1.+k)
            den = (2.**k * gammaF(alpha+beta+N+1.))
            if num == np.inf and den == np.inf:
                der_mul = 1.
            else:
                der_mul = num / den
            dP = der_mul * self.Evaluate(r, N-k, alpha+k, beta+k, norm)
            if norm:
                dP *= np.sqrt( self.Gamma(N-k,alpha+k,beta+k) / \
                               self.Gamma(N,alpha,beta) )
        else:
            dP = np.zeros(r.shape)
        return dP

    def Gamma(self,N,alpha=None,beta=None):
        r""" Return the normalization constant for the ``N``-th Jacobi polynomial.

        .. seealso:: :func:`OrthogonalPolynomial.Gamma`
        """
        if (alpha is None) and (beta is None):
            # Unpack parameters
            alpha = self.alpha
            beta = self.beta
        if (N == 0) and (alpha == -0.5) and (beta == -0.5):
            g = np.pi
        else:
            g = 2**(alpha+beta+1.) * (gammaF(N+alpha+1.)*gammaF(N+beta+1.)) / (factorial(N,exact=True) * (2.*N + alpha + beta + 1.)*gammaF(N+alpha+beta+1.))
        return g

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
        # Unpack parameters
        alpha = self.alpha
        beta = self.beta
        if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
            N = f.shape[0]-1
            # Create Even function
            fbar = np.hstack([f[::-1], f[1:N]])
            # Transform
            fhat = FFT.ifft(fbar)
            fhat = np.hstack([fhat[0], 2*fhat[1:N], fhat[N]])
            return np.real(fhat)
        
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
        # Unpack parameters
        alpha = self.alpha
        beta = self.beta
        if ( AlmostEqual(alpha,-0.5) and AlmostEqual(beta,-0.5) ):
            N = fhat.shape[0]
            # Sort values out for FFT
            fhat = np.hstack([fhat[0], np.hstack([fhat[1:N-1], fhat[N-1]*2, fhat[-2:0:-1] ])*0.5 ])
            f = FFT.fft(fhat)
            f = f[N-1::-1]
            f = np.real(f)
            return f

class LaguerrePolynomial(OrthogonalPolynomial):
    r""" Construction of Laguerre polynomials

    Args:
      alpha (float): parameter :math:`\alpha`.
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.
    """

    eps = orthpol_light.d1mach(3)

    def __init__(self, alpha, normalized=None):
        super(LaguerrePolynomial,self).__init__(normalized)
        if alpha <= -1:
            raise ValueError("Value alpha out of range")
        self.alpha = alpha

    def RecursionCoeffs(self, N, alpha=None):
        r""" Get coefficients for the Laguerre polynomials up to order ``N``

        .. seealso:: :func:`OrthogonalPolynomial.RecursionCoeffs`
        """
        if alpha is None:
            alpha = self.alpha
        (a,b,ierr) = orthpol_light.drecur(N+1, 7, alpha, 0)
        if ierr != 0: raise RuntimeError("ORTHPOL drecur: error flag %d" % ierr)
        return (a,b)

    def GaussQuadrature(self, N, norm=False):
        r""" Laguerre Gauss quadrature points

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        (x,w,ierr) = orthpol_light.dgauss(a, b)
        if ierr != 0: raise RuntimeError("ORTHPOL dgauss: error flag %d" % ierr)
        if norm:
            w /= np.sum(w)
        return (x,w)

    def GaussRadauQuadrature(self, N, norm=False):
        r""" Laguerre Gauss Radau quadrature points

        .. seealso:: :func:`OrthogonalPolynomial.GaussRadauQuadrature`
        """
        x = np.zeros(N+1)
        w = np.zeros(N+1)
        j = np.asarray(range(0,N))
        a = 2. * j + (self.alpha+1.) + 1.
        b = - np.sqrt( j[1:] * (j[1:] + (self.alpha+1.) ) )
        D = np.diag(b,1)
        D = D + D.T
        D = D + np.diag(a,0)
        x[1:],vec = np.linalg.eig(D)
        x[1:] = np.sort(x[1:])
        lp = self.Evaluate(x[1:],N,norm=False)
        w[1:] = gammaF(N+self.alpha+1.)/ ((N+self.alpha+1.) * factorial(N)) * \
                ( 1. / lp**2. )
        # Add x0 and w0
        w[0] = (self.alpha+1.) * gammaF(self.alpha+1.)**2. * gammaF(N+1) / \
               gammaF(N+self.alpha+2.)
        if norm:
            w /= np.sum(w)
        return (x,w)

    def Evaluate(self, x, N, alpha=None, norm=True):
        r""" Evaluate the ``N``-th order Laguerre polynomial

        Args:
          x ((:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomial
          N (int): order of the polynomial

        Returns:
          (:class:`ndarray<ndarray>` [``m``]) -- polynomial evaluated on the
            ``x`` points.
        """
        if self.normalized is not None: norm = self.normalized
        if alpha is None:
            alpha = self.alpha
        (a,b) = self.RecursionCoeffs(N, alpha)
        p = polymod.polyeval(x, N, a, b, norm)
        if not norm:
            pend = polymod.polyeval(np.array([0.]), N, a, b, norm)
            p *= (gammaF(N+alpha+1)/(gammaF(N+1)*gammaF(alpha+1))) / np.abs(pend)
        return p

    def GradEvaluate(self, r, N, k=0, norm=True):
        r""" ``k``-th derivative of the ``N``-th order Laguerre polynomial.
        
        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        # Unpack parameters
        if self.normalized is not None: norm = self.normalized
        alpha = self.alpha
        if N >= k:
            dP = self.Evaluate(r, N-k, alpha+k, norm)
            if norm:
                dP *= np.sqrt( self.Gamma(N-k, alpha+k) / \
                               self.Gamma(N, alpha) )
        else:
            dP = np.zeros(r.shape)
        return dP

    def Gamma(self,N,alpha=None,beta=None):
        r""" Return the normalization constant for the ``N``-th Laguerre polynomial.

        .. seealso:: :func:`OrthogonalPolynomial.Gamma`
        """
        if alpha is None:
            # Unpack parameters
            alpha = self.alpha
        g = gammaF(N+alpha+1.) / gammaF(N+1.)
        return g

class HermitePhysicistsPolynomial(OrthogonalPolynomial):
    r""" Construction of the Hermite Physicists' polynomials

    Args:
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.
    """

    def __init__(self, normalized=None):
        super(HermitePhysicistsPolynomial,self).__init__(normalized)

    def RecursionCoeffs(self, N):
        r""" Get coefficients for the Hermite Physicists' polynomials up to order ``N``

        .. seealso:: :func:`OrthogonalPolynomial.RecursionCoeffs`
        """
        (a,b,ierr) = orthpol_light.drecur(N+1, 8, 0, 0)
        if ierr != 0: raise RuntimeError("ORTHPOL drecur: error flag %d" % ierr)
        return (a,b)

    def MonomialCoeffs(self, N, norm=True):
        r""" Generate the first ``N`` monomial coefficients.

        These coefficients :math:`\{c_i\}` define the polynomials:
        :math:`P_{n}(x) = \sum_{i=0}^n c_i x^i`

        Args:
          N (int): number of monomial coefficients
          norm (bool): whether to normalize the weights

        Returns:
          (:class:`ndarray<ndarray>` [N+1]) --
            monomial coefficients ``c``

        .. seealso: :func:`OrthogonalPolynomial.MonomialCoeffs`
        """
        c = super(HermitePhysicistsPolynomial, self).MonomialCoeffs(N, norm)
        if (self.normalized is None and not norm) or \
           (self.normalized is not None and not self.normalized):
            c *= 2.**N
        return c

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Physicists' Gauss quadrature points

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        (x,w,ierr) = orthpol_light.dgauss(a, b)
        if ierr != 0: raise RuntimeError("ORTHPOL dgauss: error flag %d" % ierr)
        if norm:
            w /= np.sum(w)
        return (x,w)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order Hermite Physicists' polynomial

        Args:
          x ((:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomial
          N (int): order of the polynomial

        Returns:
          (:class:`ndarray<ndarray>` [``m``]) -- polynomial evaluated on the
            ``x`` points.
        """
        if self.normalized is not None: norm = self.normalized
        (a,b) = self.RecursionCoeffs(N)
        p = polymod.polyeval(x, N, a, b, norm)
        if not norm:
            p *= 2.**N
        return p

    def GradEvaluate(self, r, N, k=0, norm=True):
        r""" ``k``-th derivative of the ``N``-th order Hermite Physicists' polynomial.
        
        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        if self.normalized is not None: norm = self.normalized
        if N >= k:
            fact = np.exp( gammalnF(N+1) - gammalnF(N-k+1) )
            dP = 2.**k * fact * self.Evaluate(r, N-k, norm)
            if norm:
                dP *= np.sqrt( self.Gamma(N-k) / \
                               self.Gamma(N) )
        else:
            dP = np.zeros(r.shape)
        return dP

    def Gamma(self,N,alpha=None,beta=None):
        r""" Return the normalization constant for the ``N``-th Hermite Physicists' polynomial.

        .. seealso:: :func:`OrthogonalPolynomial.Gamma`
        """
        return math.sqrt(np.pi) * 2.**N * factorial(N,exact=True)

class HermiteProbabilistsPolynomial(OrthogonalPolynomial):
    r""" Construction of the Hermite Probabilists polynomials

    Args:
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.
    """

    def __init__(self, normalized=None):
        super(HermiteProbabilistsPolynomial,self).__init__(normalized)

    def RecursionCoeffs(self, N):
        r""" Get coefficients for the Hermite Probabilists' polynomials up to order ``N``

        .. seealso:: :func:`OrthogonalPolynomial.RecursionCoeffs`
        """
        a = np.zeros(N+1)
        b = np.hstack([[np.sqrt(2.*np.pi)],np.arange(1,N+1,dtype=float)])
        return (a,b)

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Probabilists' Gauss quadrature points

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        (x,w,ierr) = orthpol_light.dgauss(a, b)
        if ierr != 0: raise RuntimeError("ORTHPOL dgauss: error flag %d" % ierr)
        if norm:
            w /= np.sum(w)
        return (x,w)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order Hermite Probabilists' polynomial

        Args:
          x ((:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomial
          N (int): order of the polynomial

        Returns:
          (:class:`ndarray<ndarray>` [``m``]) -- polynomial evaluated on the
            ``x`` points.
        """
        if self.normalized is not None: norm = self.normalized
        (a,b) = self.RecursionCoeffs(N)
        p = polymod.polyeval(x, N, a, b, norm)
        return p

    def GradEvaluate(self, r, N, k=0, norm=True):
        r""" ``k``-th derivative of the ``N``-th order Hermite Probabilists' polynomial.
        
        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        if self.normalized is not None: norm = self.normalized
        if N >= k:
            dP = np.exp( gammalnF(N+1) - gammalnF(N-k+1) ) * \
                 self.Evaluate(r, N-k, norm)
            if norm:
                dP *= np.sqrt( self.Gamma(N-k) / self.Gamma(N) )
        else:
            dP = np.zeros(r.shape)
        return dP

    def Gamma(self,N,alpha=None,beta=None):
        r""" Return the normalization constant for the ``N``-th Hermite Probabilists' polynomial.

        .. seealso:: :func:`OrthogonalPolynomial.Gamma`
        """
        return math.sqrt(2.*np.pi) * factorial(N,exact=True)

    @staticmethod
    def from_xml_element(node):
        normalized = node.attrib.get('normalized', None)
        if normalized is not None: normalized = bool(normalized)
        return HermiteProbabilistsPolynomial(normalized=normalized)

class GenericOrthogonalPolynomial(OrthogonalPolynomial):
    r""" Construction of polynomials orthogonal with respect to a generic measure

    Args:
      mu (float (float x, int i)): function returning the mass value at the
        point ``x`` of interval ``i`` of the continuous spectrum.
      endl (:class:`ndarray<numpy.ndarray>` [``mc``]): left endpoints of the
        intervals in the continuous spectrum.
      endr (:class:`ndarray<numpy.ndarray>` [``mc``]): right endpoints of the
        intervals in the continuous spectrum.
      mc (int): number of component intervals in the continuous part of the
        spectrum.
      mp (int): number of points in the discrete part of the spectrum. If the
        measure has no discrete part, set mp=0.
      xp (:class:`ndarray<numpy.ndarray>` [``mp``]): abscissas of the points
        in the discrete spectrum.
      yp (:class:`ndarray<numpy.ndarray>` [``mp``]): jumps of the points
        in the discrete spectrum.
      ncapm (int): maximum integer N0
      irout (int): selects the routine for generating the recursion coefficients
        from the discrete inner product; ``irout=1`` selects the routine ``sti``,
        ``irout!=1`` selects the routine ``lancz``.
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.

    .. note:: Parameters ``iq``, ``quad``, ``idelta`` in :cite:`Gautschi1994` are suppressed.
       Instead the routine ``qgp`` of ORTHPOL :cite:`Gautschi1994` is used by default
       (``iq=0`` and ``idelta=2``)
    """

    eps = orthpol_light.d1mach(3)

    def __init__(self, mu, endl, endr, mc=1, mp=0, xp=None, yp=None,
                 ncapm=500, irout=1, normalized=None):
        if not ORTHPOL_SUPPORT:
            raise ImportError("The orthpol package is not available. " + \
                              "(only its light verions orthpol_light is installed.")
        super(GenericOrthogonalPolynomial,self).__init__(normalized)
        self.mu = mu
        self.endl = endl if (endl[0] != -np.inf) else endl[1:]
        self.endr = endr if (endr[-1] != np.inf) else endr[:-1]
        self.finl = (endl[0] != -np.inf)
        self.finr = (endr[-1] != np.inf)
        self.ncapm = ncapm
        self.mc = mc
        self.mp = mp
        self.xp = np.array([]) if xp == None else xp
        self.yp = np.array([]) if yp == None else yp
        self.irout = irout
        # Default values
        self.iq = 0
        self.idelta = 2

        # Caching space for recurrence coefficients
        self.cached_alphabeta = None

    def RecursionCoeffs(self, N):
        r""" Get the recursion coefficients up to order ``N``

        .. seealso:: :func:`OrthogonalPolynomial.RecursionCoeffs`
        """
        if (self.cached_alphabeta == None) or (N > self.cached_alphabeta['N']):
            # Compute alpha and beta coefficients.
            (alphaCap, betaCap,
             ncapCap, kountCap,
             ierrCap, ieCap) = orthpol.dmcdis(N+1, self.ncapm, self.mc, self.mp,
                                              self.xp, self.yp, self.mu,
                                              self.eps, self.iq, self.idelta, 
                                              self.irout, self.finl, self.finr,
                                              self.endl, self.endr )
            self.cached_alphabeta = {'N': N,
                                      'alpha': alphaCap,
                                      'beta': betaCap}
        else:
            alphaCap = self.cached_alphabeta['alpha']
            betaCap = self.cached_alphabeta['beta']
        return (alphaCap,betaCap)

    def GaussQuadrature(self, N, norm=False):
        r""" Gauss quadrature points.

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        (x,w,ierr) = orthpol.dgauss(N+1, a, b)
        if ierr != 0: raise RuntimeError("ORTHPOL dgauss: error flag %d" % ierr)
        if norm:
            w /= np.sum(w)
        return (x,w)

    def GaussLobattoQuadrature(self, N, norm=False):
        r""" Gauss Lobatto quadrature points.

        .. seealso:: :func:`OrthogonalPolynomial.GaussLobattoQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        (x,w,ierr) = orthpol.dlob(N-1, a, b, self.endl[0], self.endr[-1])
        if ierr != 0: raise RuntimeError("ORTHPOL dlob: error flag %d" % ierr)
        # Normalization and rescaling
        if norm:
            w /= np.sum(w)
        return (x,w)

    def GaussRadauQuadrature(self, N, norm=False):
        r""" Gauss Radau quadrature points.

        .. seealso:: :func:`OrthogonalPolynomial.GaussRadauQuadrature`
        """
        (a,b) = self.RecursionCoeffs(N)
        end = self.endl[0] if (self.endl[0] != -np.inf) else self.endr[-1]
        (x,w,ierr) = orthpol.dradau(N, a, b, end)
        if ierr != 0: raise RuntimeError("ORTHPOL dlob: error flag %d" % ierr)
        # Normalization and rescaling
        if norm:
            w /= np.sum(w)
        return (x,w)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order polynomial

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if self.normalized is not None: norm = self.normalized
        (a,b) = self.RecursionCoeffs(N)
        p = orthpol.polyeval(x, N, a, b, norm)
        return p

    def GradEvaluate(self, r, N, k=0, norm=True):
        r""" ``k``-th derivative of the ``N``-th order polynomial.

        .. warning:: no derivatives are implemented for this type of
           polynomials. Works only with ``k==0``.
        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        if self.normalized is not None: norm = self.normalized
        if k > 0:
            raise NotImplemented("Not implemented for this class")
        else:
            return self.Evaluate(r, N, norm)

    def GradVandermonde(self, r, N, k=0, norm=True):
        r"""
        .. seealso:: :func:`OrthogonalPolynomial.GradVandermonde`
        """
        if self.normalized is not None: norm = self.normalized
        self.RecursionCoeffs(N) # Pre-cache recursion coefficients
        return super(GenericOrthogonalPolynomial,
                     self).GradVandermonde(r, N, k, norm)

    def Gamma(self, N, alpha=None, beta=None):
        r""" Return the normalization constant for the ``N``-th polynomial.

        .. seealso:: :func:`OrthogonalPolynomial.Gamma`
        """
        (a,b) = self.RecursionCoeffs(N)
        g = orthpol.numeric_gamma(N, a, b)
        return g