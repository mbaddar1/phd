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

from SpectralToolbox.Spectral1D.Constants import *
from SpectralToolbox.Spectral1D.AbstractClasses import *
from SpectralToolbox.Spectral1D.OrthogonalFunctions import *
from SpectralToolbox.Spectral1D.OrthogonalPolynomials import *

__all__ = ['ConstantExtendedHermiteProbabilistsFunction',
           'LinearExtendedHermiteProbabilistsFunction',
           'ConstantExtendedHermitePhysicistsFunction',
           'HermiteProbabilistsRadialBasisFunction',
           'ConstantExtendedHermiteProbabilistsRadialBasisFunction',
           'LinearExtendedHermiteProbabilistsRadialBasisFunction']

class ConstantExtendedHermiteProbabilistsFunction(Basis):
    r""" Construction of the Hermite Probabilists' functions extended with the constant basis

    The basis is defined by:

    .. math::

       \phi_0(x) = 1 \qquad \phi_i(x) = \psi_{i-1}(x) \quad \text{for } i=1\ldots

    where :math:`\psi_j` are the Hermite Probabilists' functions.

    Args:
      normalized (bool): whether to normalize the underlying polynomials.
        Default=``None`` which leaves the choice at evaluation time.
    """
    def __init__(self, normalized=None):
        self.hpf = HermiteProbabilistsFunction(normalized)

    def Quadrature(self, N, quadType=None, norm=False):
        r""" Generate quadrature rules of the selected type.

        .. seealso:: :func:`OrthogonalPolynomial.Quadrature`
        """
        if quadType in [None, GAUSS]:
            return self.GaussQuadrature(N, norm)
        else:
            raise ValueError("quadType=%s not available" % quadType)

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Probabilists' function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        return self.hpf.GaussQuadrature(N, norm)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order constant extended Hermite Probabilists' function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if N > 0:
            p = self.hpf.Evaluate(x, N-1, norm)
        else:
            p = np.ones(x.shape[0])
        return p

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th order constant extended Hermite Probabilists' function

        .. seealso:: :func:`HermitePhysicistsFunction.GradEvaluate`
        """
        if N > 0:
            dp = self.hpf.GradEvaluate(x, N-1, k, norm)
        else:
            dp = np.ones(x.shape[0]) if k == 0 else np.zeros(x.shape[0])
        return dp

    def GradVandermonde(self, r, N, k=0, norm=True):
        r""" Generate the ``k``-th derivative of the ``N``-th order Vandermoned matrix.

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum polynomial order
          k (int): order of the derivative
          norm (bool): whether to return normalized (``True``) or unnormalized
            (``False``) polynomial. The parameter is ignored if the ``normalized``
            parameter is provided at construction time.

        Returns:
          (:class:`ndarray<ndarray>` [``m``,``N+1``]) -- polynomials evaluated
            at the ``r`` points.
        """
        DVr = np.zeros((r.shape[0],N+1))
        for i in range(0,N+1):
            DVr[:,i] = self.GradEvaluate(r, i, k, norm)
        return DVr

    @staticmethod
    def from_xml_element(node):
        normalized = node.attrib.get('normalized', None)
        if normalized is not None: normalized = bool(normalized)
        return ConstantExtendedHermiteProbabilistsFunction(normalized=normalized)

class LinearExtendedHermiteProbabilistsFunction(Basis):
    r""" Construction of the Hermite Probabilists' functions extended with the constant and linear basis

    The basis is defined by:

    .. math::

       \phi_0(x) = 1 \qquad \phi_1(x) = x \qquad \phi_i(x) = \psi_{i-1}(x) \quad \text{for } i=1\ldots

    where :math:`\psi_j` are the Hermite Probabilists' functions.

    Args:
      normalized (bool): whether to normalize the underlying polynomials.
        Default=``None`` which leaves the choice at evaluation time.
    """
    def __init__(self, normalized=None):
        self.hpf = HermiteProbabilistsFunction(normalized)

    def Quadrature(self, N, quadType=None, norm=False):
        r""" Generate quadrature rules of the selected type.

        .. seealso:: :func:`OrthogonalPolynomial.Quadrature`
        """
        if quadType in [None, GAUSS]:
            return self.GaussQuadrature(N, norm)
        else:
            raise ValueError("quadType=%s not available" % quadType)

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Probabilists' function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        return self.hpf.GaussQuadrature(N, norm)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order constant extended Hermite Probabilists' function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if N == 0:
            p = np.ones(x.shape[0])
        elif N == 1:
            p = x.flatten()
        else:
            p = self.hpf.Evaluate(x, N-2, norm)
        return p

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th order constant extended Hermite Probabilists' function

        .. seealso:: :func:`HermitePhysicistsFunction.GradEvaluate`
        """
        if N == 0:
            dp = np.ones(x.shape[0]) if k == 0 else np.zeros(x.shape[0])
        elif N == 1:
            if k == 0:
                dp = x.flatten()
            elif k == 1:
                dp = np.ones(x.shape[0])
            elif k > 1:
                dp = np.zeros(x.shape[0])
        else:
            dp = self.hpf.GradEvaluate(x, N-2, k, norm)
        return dp

    def GradVandermonde(self, r, N, k=0, norm=True):
        r""" Generate the ``k``-th derivative of the ``N``-th order Vandermoned matrix.

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum polynomial order
          k (int): order of the derivative
          norm (bool): whether to return normalized (``True``) or unnormalized
            (``False``) polynomial. The parameter is ignored if the ``normalized``
            parameter is provided at construction time.

        Returns:
          (:class:`ndarray<ndarray>` [``m``,``N+1``]) -- polynomials evaluated
            at the ``r`` points.
        """
        DVr = np.zeros((r.shape[0],N+1))
        for i in range(0,N+1):
            DVr[:,i] = self.GradEvaluate(r, i, k, norm)
        return DVr

    @staticmethod
    def from_xml_element(node):
        normalized = node.attrib.get('normalized', None)
        if normalized is not None: normalized = bool(normalized)
        return LinearExtendedHermiteProbabilistsFunction(normalized=normalized)
        
class ConstantExtendedHermitePhysicistsFunction(Basis):
    r""" Construction of the Hermite Physicists' functions extended with the constant basis

    The basis is defined by:

    .. math::

       \phi_0(x) = 1 \qquad \phi_i(x) = \psi_{i-1}(x) \quad \text{for } i=1\ldots

    where :math:`\psi_j` are the Hermite Physicists' functions.

    Args:
      normalized (bool): whether to normalize the underlying polynomials.
        Default=``None`` which leaves the choice at evaluation time.
    """
    def __init__(self, normalized=None):
        self.hpf = HermitePhysicistsFunction(normalized)

    def Quadrature(self, N, quadType=None, norm=False):
        r""" Generate quadrature rules of the selected type.

        .. seealso:: :func:`OrthogonalPolynomial.Quadrature`
        """
        if quadType in [None, GAUSS]:
            return self.GaussQuadrature(N, norm)
        else:
            raise ValueError("quadType=%s not available" % quadType)

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Probabilists' function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        return self.hpf.GaussQuadrature(N, norm)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th order constant extended Hermite Probabilists' function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if N > 0:
            p = self.hpf.Evaluate(x, N-1, norm)
        else:
            p = np.ones(x.shape[0])
        return p

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th order constant extended Hermite Probabilists' function

        .. seealso:: :func:`HermitePhysicistsFunction.GradEvaluate`
        """
        if N > 0:
            dp = self.hpf.GradEvaluate(x, N-1, k, norm)
        else:
            dp = np.ones(x.shape[0]) if k == 0 else np.zeros(x.shape[0])
        return dp

    def GradVandermonde(self, r, N, k=0, norm=True):
        r""" Generate the ``k``-th derivative of the ``N``-th order Vandermoned matrix.

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum polynomial order
          k (int): order of the derivative
          norm (bool): whether to return normalized (``True``) or unnormalized
            (``False``) polynomial. The parameter is ignored if the ``normalized``
            parameter is provided at construction time.

        Returns:
          (:class:`ndarray<ndarray>` [``m``,``N+1``]) -- polynomials evaluated
            at the ``r`` points.
        """
        DVr = np.zeros((r.shape[0],N+1))
        for i in range(0,N+1):
            DVr[:,i] = self.GradEvaluate(r, i, k, norm)
        return DVr

    @staticmethod
    def from_xml_element(node):
        normalized = node.attrib.get('normalized', None)
        if normalized is not None: normalized = bool(normalized)
        return ConstantExtendedHermitePhysicistsFunction(normalized=normalized)

        
class HermiteProbabilistsRadialBasisFunction(Basis):
    r""" Construction of the Hermite Probabilists' Radial Basis Functions

    For the set :math:`\left\{x_i\right\}_{i=1}^N` of Gauss-Hermite points,
    the basis are defined by:

    .. math::

       \phi_i(x) = \exp\left( -\frac{(x-x_i)^2}{2\sigma^2_{i}} \right) 

    where :math:`\sigma_i=(x_{i+1} - x_{i-1})/2.`, :math:`\sigma_0=\sigma_1` and
    :math:`\sigma_N=\sigma_{N-1}`

    Args:
      nbasis (int): number of knots points :math:`x_i`
      scale (float): scaling for the badwidth :math:`\sigma`.
    """

    def __init__(self, nbasis, scale=1.):
        if nbasis < 0:
            raise ValueError("Range error. nbasis >= 0 must hold")
        self.nknots = nbasis+1
        self.hp = HermiteProbabilistsPolynomial()
        self.xknots, self.wknots = self.hp.GaussQuadrature(self.nknots-1)
        self.sigma = np.zeros(self.nknots)
        if self.nknots == 1:
            self.sigma[:] = 1.
        elif self.nknots == 2:
            self.sigma[0] = self.xknots[1] - self.xknots[0]
            self.sigma[1] = self.sigma[0]
        else:
            self.sigma[1:-1] = (self.xknots[2:] - self.xknots[:-2])/2.
            self.sigma[0] = self.sigma[1]
            self.sigma[-1] = self.sigma[-2]
        self.sigma *= scale

    def Quadrature(self, N, quadType=None, norm=False):
        r""" Generate quadrature rules of the selected type.

        .. seealso:: :func:`OrthogonalPolynomial.Quadrature`
        """
        if quadType in [None, GAUSS]:
            return self.GaussQuadrature(N, norm)
        else:
            raise ValueError("quadType=%s not available" % quadType)

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Probabilists' function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        return self.hp.GaussQuadrature(N, norm)

    def Evaluate(self, x, N, norm=True, extended_output=False):
        r""" Evaluate the ``N``-th Hermite Probabilists' Radial Basis Function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if N > self.nknots:
            raise ValueError("N must be <= than the number of knots")
        diff = x - self.xknots[N]
        out = np.exp( - diff**2./(2.*self.sigma[N]**2.) )
        if extended_output:
            return (out, diff)
        else:
            return out

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th Hermite Probabilists' Radial Basis Function

        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        (out, diff) = self.Evaluate(x, N, norm, extended_output=True)
        if k == 0:
            return out
        elif k == 1:
            out *= -diff/self.sigma[N]**2.
        elif k == 2:
            out *= ( -1./self.sigma[N]**2. + diff**2./self.sigma[N]**4. )
        elif k == 3:
            out *= diff * (3.*self.sigma[N]**2. - diff**2.) / self.sigma[N]**6.
        else:
            raise ValueError("%d-th derivative not defined yet" % k)
        return out

    def GradVandermonde(self, r, N, k=0, norm=True):
        r""" Generate the ``k``-th derivative of the ``N``-th order Vandermoned matrix.

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum polynomial order
          k (int): order of the derivative
          norm (bool): whether to return normalized (``True``) or unnormalized
            (``False``) polynomial. The parameter is ignored if the ``normalized``
            parameter is provided at construction time.

        Returns:
          (:class:`ndarray<ndarray>` [``m``,``N+1``]) -- polynomials evaluated
            at the ``r`` points.
        """
        DVr = np.zeros((r.shape[0],N+1))
        for i in range(0,N+1):
            DVr[:,i] = self.GradEvaluate(r, i, k, norm)
        return DVr

    @staticmethod
    def from_xml_element(node):
        nbasis = int( node.attrib['nbasis'] )
        scale = float( node.attrib.get('scale', 1.) )
        return HermiteProbabilistsRadialBasisFunction(nbasis, scale=scale)

class ConstantExtendedHermiteProbabilistsRadialBasisFunction(Basis):
    r""" Construction of the Hermite Probabilists' Radial Basis Functions

    For the set :math:`\left\{x_i\right\}_{i=1}^N` of Gauss-Hermite points,
    the basis :math:`\{\phi_i\}_{i=0}^M` are defined by:

    .. math::

       \phi_0(x) = 1 \\
       \phi_i(x) = \exp\left( -\frac{(x-x_i)^2}{2\sigma^2_{i-1}} \right) 

    where :math:`\sigma_i=x_{i+1} - x_{i}`, :math:`\sigma_0=\sigma_1` and
    :math:`\sigma_N=\sigma_{N-1}`

    Args:
      nbasis (int): number of basis :math:`M`
      scale (float): scaling for the badwidth :math:`\sigma`.
    """

    def __init__(self, nbasis, scale=1.):
        self.nbasis = nbasis
        if nbasis >= 1:
            self.hprbf = HermiteProbabilistsRadialBasisFunction(nbasis-1, scale=scale)

    def Quadrature(self, N, quadType=None, norm=False):
        r""" Generate quadrature rules of the selected type.

        .. seealso:: :func:`OrthogonalPolynomial.Quadrature`
        """
        return self.hprbf.Quadrature(N, quadType, norm)

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Probabilists' function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        return self.hprbf.GaussQuadrature(N, norm)

    def Evaluate(self, x, N, norm=True):
        r""" Evaluate the ``N``-th Hermite Probabilists' Radial Basis Function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if N > self.nbasis:
            raise ValueError("N must be <= than the number of basis")
        if N == 0:
            out = np.ones(x.shape[0])
        else:
            nrbf = N-1
            out = self.hprbf.Evaluate(x, nrbf, norm)
        return out

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th Hermite Probabilists' Radial Basis Function

        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        if N == 0:
            if k == 0:
                return np.ones(x.shape[0])
            elif k > 0:
                return np.zeros(x.shape[0])
        else:
            nrbf = N-1
            return self.hprbf.GradEvaluate(x, nrbf, k, norm)

    def GradVandermonde(self, r, N, k=0, norm=True):
        r""" Generate the ``k``-th derivative of the ``N``-th order Vandermoned matrix.

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum polynomial order
          k (int): order of the derivative
          norm (bool): whether to return normalized (``True``) or unnormalized
            (``False``) polynomial. The parameter is ignored if the ``normalized``
            parameter is provided at construction time.

        Returns:
          (:class:`ndarray<ndarray>` [``m``,``N+1``]) -- polynomials evaluated
            at the ``r`` points.
        """
        DVr = np.zeros((r.shape[0],N+1))
        for i in range(0,N+1):
            DVr[:,i] = self.GradEvaluate(r, i, k, norm)
        return DVr

    @staticmethod
    def from_xml_element(node):
        nbasis = int( node.attrib['nbasis'] )
        scale = float( node.attrib.get('scale', 1.) )
        return ConstantExtendedHermiteProbabilistsRadialBasisFunction(nbasis, scale=scale)

class LinearExtendedHermiteProbabilistsRadialBasisFunction(Basis):
    r""" Construction of the Hermite Probabilists' Radial Basis Functions

    For the set :math:`\left\{x_i\right\}_{i=1}^N` of Gauss-Hermite points,
    the basis :math:`\{\phi_i\}_{i=0}^M` are defined by:

    .. math::

       \phi_0(x) = 1 \\
       \phi_1(x) = x \\
       \phi_i(x) = \exp\left( -\frac{(x-x_i)^2}{2\sigma^2_{i-1}} \right) 

    where :math:`\sigma_i=x_{i+1} - x_{i}`, :math:`\sigma_0=\sigma_1` and
    :math:`\sigma_N=\sigma_{N-1}`

    Args:
      nbasis (int): number of basis :math:`M`
      scale (float): scaling for the badwidth :math:`\sigma`.
    """

    def __init__(self, nbasis, scale=1.):
        self.nbasis = nbasis
        if nbasis >= 2:
            self.hprbf = HermiteProbabilistsRadialBasisFunction(nbasis-2, scale=scale)

    def Quadrature(self, N, quadType=None, norm=False):
        r""" Generate quadrature rules of the selected type.

        .. seealso:: :func:`OrthogonalPolynomial.Quadrature`
        """
        return self.hprbf.Quadrature(N, quadType, norm)

    def GaussQuadrature(self, N, norm=False):
        r""" Hermite Probabilists' function Gauss quadratures

        .. seealso:: :func:`OrthogonalPolynomial.GaussQuadrature`
        """
        return self.hprbf.GaussQuadrature(N, norm)

    def Evaluate(self, x, N, norm=True, extended_output=False):
        r""" Evaluate the ``N``-th Hermite Probabilists' Radial Basis Function

        .. seealso:: :func:`OrthogonalPolynomial.Evaluate`
        """
        if N > self.nbasis:
            raise ValueError("N must be <= than the number of basis")
        if N == 0:
            out = np.ones(x.shape[0])
        elif N == 1:
            out = x.flatten()
        else:
            nrbf = N-2
            out = self.hprbf.Evaluate(x, nrbf, norm)
        return out

    def GradEvaluate(self, x, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th Hermite Probabilists' Radial Basis Function

        .. seealso:: :func:`OrthogonalPolynomial.GradEvaluate`
        """
        if N == 0:
            if k == 0:
                return np.ones(x.shape[0])
            elif k > 0:
                return np.zeros(x.shape[0])
        elif N == 1:
            if k == 0:
                return x.flatten()
            elif k == 1:
                return np.ones(x.shape[0])
            elif k > 1:
                return np.zeros(x.shape[0])
        else:
            nrbf = N-2
            return self.hprbf.GradEvaluate(x, nrbf, k, norm)

    def GradVandermonde(self, r, N, k=0, norm=True):
        r""" Generate the ``k``-th derivative of the ``N``-th order Vandermoned matrix.

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum polynomial order
          k (int): order of the derivative
          norm (bool): whether to return normalized (``True``) or unnormalized
            (``False``) polynomial. The parameter is ignored if the ``normalized``
            parameter is provided at construction time.

        Returns:
          (:class:`ndarray<ndarray>` [``m``,``N+1``]) -- polynomials evaluated
            at the ``r`` points.
        """
        DVr = np.zeros((r.shape[0],N+1))
        for i in range(0,N+1):
            DVr[:,i] = self.GradEvaluate(r, i, k, norm)
        return DVr

    @staticmethod
    def from_xml_element(node):
        nbasis = int( node.attrib['nbasis'] )
        scale = float( node.attrib.get('scale', 1.) )
        return LinearExtendedHermiteProbabilistsRadialBasisFunction(nbasis, scale=scale)