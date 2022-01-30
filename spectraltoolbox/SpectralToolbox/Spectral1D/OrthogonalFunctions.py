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

from SpectralToolbox.Spectral1D.Constants import *
from SpectralToolbox.Spectral1D.AbstractClasses import *
from SpectralToolbox.Spectral1D.OrthogonalPolynomials import *

__all__ = ['HermitePhysicistsFunction', 'HermiteProbabilistsFunction',
           'LaguerreFunction', 'Fourier']

class HermitePhysicistsFunction(OrthogonalPolynomial):
    r""" Construction of the Hermite Physiticists' functions

    Args:
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.
    """

    def __init__(self, normalized=None):
        super(HermitePhysicistsFunction,self).__init__(normalized)
        self.H = HermitePhysicistsPolynomial()
        self.He = HermiteProbabilistsPolynomial()

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Physicists' function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        (x,w) = self.H.GaussQuadrature(N)
        hf = self.Evaluate(x,N,norm=False)
        w = np.sqrt(np.pi) * 2.**N * factorial(N) / ((N+1) * hf**2.)
        if norm:
            w /= np.sum(w)
        return (x,w)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order Hermite Physicists' function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if self.normalized is not None: norm = self.normalized
        p = self.H.Evaluate(x, N, norm)
        p *= np.exp( -x**2./2. )
        return p

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th order Hermite Physicists' function

        One can write it as

        .. math::

           \psi_n^{(k)} = \sum_{i=0}^k (-1)^i F_{k,i} \psi_i H_n^{(k-i)}

        where :math:`F_{k,i}` is the :math:`i`-th component of the :math:`k`-th
        row of the Pascal's triangle, :math:`psi_i` is the :math:`i`-th
        Hermite Physicists' function and :math:`H_n^{(n-i)}` is the
        :math:`(k-1)`-th derivative of the :math:`n`-th Hermite Physicists' 
        polynomial.

        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        if self.normalized is not None: norm = self.normalized
        F = 1
        dP = self.H.GradEvaluate(x, N, k, norm=False) * \
             self.Evaluate(x, 0, norm=False)
        for i in range(1,k+1):
            F = F * (k-i+1)/i
            dP += (-1)**i * F * np.exp( -x**2./2. ) * \
                  self.He.Evaluate(x,i,norm=False) * \
                  self.H.GradEvaluate(x, N, k-i, norm=False)
        if norm:
            dP /= np.sqrt( self.Gamma(N) )
        return dP

    def Gamma(self, N):
        r""" Return the normalization constant for the ``N``-th Hermite Physiticsts' function.

        .. seealso:: :func:`OrthogonalPolynomial.Gamma`
        """
        return math.sqrt(np.pi) * 2.**N * factorial(N,exact=True)

class HermiteProbabilistsFunction(OrthogonalPolynomial):
    r""" Construction of the Hermite Probabilists' functions

    This are rescaling of :math:`\sqrt{2}` of the Hermite Physicists'
    functions.

    Args:
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.
    """

    def __init__(self, normalized=None):
        super(HermiteProbabilistsFunction,self).__init__(normalized)
        self.Hf = HermitePhysicistsFunction()

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Probabilists' function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        (x,w) = self.Hf.GaussQuadrature(N, norm)
        x *= np.sqrt(2.)
        w *= np.sqrt(2.)
        return (x,w)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order Hermite Probabilists' function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        return self.GradEvaluate(x, N, k=0, norm=norm)

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th order Hermite Probabilists' function

        .. seealso:: :func:`HermitePhysicistsFunction.GradEvaluate`
        """
        if self.normalized is not None: norm = self.normalized
        dP = np.sqrt(2.)**(-k) * self.Hf.GradEvaluate(x/np.sqrt(2.), N, k, norm)
        if norm:
            dP /= np.sqrt(np.sqrt(2.))
        return dP

    def Gamma(self, N):
        r""" Return the normalization constant for the ``N``-th Hermite Probabilists' function.

        .. seealso:: :func:`OrthogonalPolynomial.Gamma`
        """
        return np.sqrt(2.) * self.Hf.Gamma(N)

    @staticmethod
    def from_xml_element(node):
        opt_node = node.find('basis_options')
        if opt_node is None:
            return HermiteProbabilistsFunction()
        else:
            normalized = opt_node.attrib.get('normalized', None)
            if normalized is not None: normalized = bool(normalized)
            return HermiteProbabilistsFunction(normalized=normalized)

class LaguerreFunction(OrthogonalPolynomial):
    r""" Construction of the Laguerre functions

    Args:
      alpha (float): parameter :math:`\alpha`.
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.
    """

    def __init__(self, alpha, normalized=None):
        super(LaguerreFunction,self).__init__(normalized)
        self.alpha = alpha
        self.L = LaguerrePolynomial(alpha)

    def GaussQuadrature(self, N, norm=False):
        r""" Laguerre function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        (x,w) = self.L.GaussQuadrature(N)
        w *= np.exp(x)
        if norm:
            w /= np.sum(w)
        return (x,w)

    def GaussRadauQuadrature(self, N, norm=False):
        (x,w) = self.L.GaussRadauQuadrature(N)
        w *= np.exp(x)
        if norm:
            w /= np.sum(w)
        return (x,w)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order Laguerre function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if self.normalized is not None: norm = self.normalized
        p = self.L.Evaluate(x, N, norm)
        p *= np.exp(-x/2)
        return p

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th order Laguerre function

        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        if self.normalized is not None: norm = self.normalized
        F = 1
        dP = self.L.GradEvaluate(x, N, k, norm=False) * np.exp(-x/2.)
        for i in range(1,k+1):
            F = F * (k-i+1)/i
            dP += (-0.5)**i * F * np.exp(-x/2.) * \
                  self.L.GradEvaluate(x, N, k-i, norm=False)
        if norm:
            dP /= np.sqrt( self.Gamma(N) )
        return dP

    def Gamma(self, N):
        return self.L.Gamma(N)


class Fourier(OrthogonalBasis):

    def __init__(self):
        self.bspan = [0., 2.*np.pi]

    def Quadrature(self, N, norm=False):
        r""" Generate quadrature points for the Fourier series

        .. seealso:: :func:`OrthogonalBase.Quadrature`
        """
        x = np.linspace(0., 2.*np.pi, N+3)[:-1]
        w = 2. * np.pi * np.ones(N+2)/float(N+2)
        if norm:
            w /= 2. * np.pi
        return (x,w)

    def Evaluate(self, x, N):
        r""" Evaluate the ``N``-th Fourier basis

        .. seealso:: :func:`OrthogonalBase.Evaluate`
        """
        if N == 0:
            P = np.ones( x.shape )
        elif N % 2 == 0:
            P = - np.sin( N/2 * x )
        else:
            P = np.cos( (N+1)/2 * x )
        return P

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th Fourier basis

        .. seealso:: :func:`OrthogonalBase.GradEvaluate`
        """
        if k == 0:
            P = self.Evaluate(x, N)
        else:
            if N == 0:
                P = np.zeros( x.shape )
            elif N % 2 == 0:
                P = (-1.)**((k+1)//2) * (N/2.)**k * self.Evaluate(x, N - (k % 2))
            else:
                P = (-1.)**(k//2) * ((N+1)/2.)**k * self.Evaluate(x, N + (k % 2))
        if norm:
            P /= np.sqrt( self.Gamma(N) )
        return P

    def Gamma(self, N):
        r""" Evaluate the ``N``-th normalization constant for the Fourier basis

        .. seealso:: :func:`OrthogonalBase.Gamma`
        """
        if N == 0:
            return 2.*np.pi
        else:
            return np.pi

    def project(self, r, f, N):
        r""" Project ``f`` onto the span of Fourier basis up to order ``N``

        .. seealso:: :func:`OrthogonalBase.project`
        """
        if len(r) != len(f):
            raise ValueError("r and f should have the same size")
        elif len(r) == N+2:
            x,w = self.Quadrature(N)
            if np.allclose(x,r,atol=1e-12):
                fhatc = FFT.rfft( f ) # Complex
                fhatc = np.vstack( (fhatc.real, fhatc.imag) ).T
                fhat = 2. * np.sqrt(np.pi) / float(N+2) * \
                       np.hstack( ( fhatc[0,0]/np.sqrt(2),
                                    fhatc[1:,:].flatten()) )
                return fhat[:N+2]
        return super(Fourier,self).project(r, f, N+1)

    def interpolate(self, x, f, xi, order):
        r""" Interpolate function values ``f`` from points ``x`` to points ``xi``.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` original points
          f (:class:`ndarray<ndarray>` [``m``]): function values at points ``x``
          xi (:class:`ndarray<ndarray>` [``n``]): set of ``n`` interpolation 
            points
          order (int): polynomial order

        Returns:
          (:class:`ndarray<ndarray>` [``n``]) -- interpolated function values at
            points ``xi``
        """
        fhat = self.project(x, f, order-1)
        # Use the generalized vandermonde matrix
        V = self.GradVandermonde(xi, order, 0)
        fi = np.dot(V,fhat)
        return fi
