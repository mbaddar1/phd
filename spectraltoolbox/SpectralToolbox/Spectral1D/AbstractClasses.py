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

import polymod
from SpectralToolbox.Spectral1D.Constants import *

__all__ = ['Basis', 'OrthogonalBasis', 'OrthogonalPolynomial']

class Basis(object):
    """ This is an abstract class for 1-d basis

    Raises:
      NotImplementedError: if not overridden.
    """
    def __init__(self):
        raise NotImplementedError("Not implemented or undefined for this class.")

    def Evaluate(self, r, N):
        r""" Evaluate the ``N``-th order polynomial at points ``r``

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomial
          N (int): order of the polynomial

        Returns:
          (:class:`ndarray<ndarray>` [``m``]) -- polynomial evaluated on
             the ``r`` points.

        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")
    
    def GradEvaluate(self, r, N, k=0):
        r""" Evaluate the ``k``-th derivative of the ``N``-th order polynomial at points ``r``

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomial
          N (int): order of the polynomial
          k (int): order of the derivative

        Returns:
          (:class:`ndarray<ndarray>` [``m``]) -- ``k``-th derivative of the
            polynomial evaluated on the ``r`` points.
        
        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")

    def GradVandermonde(self, r, N, k=0):
        r""" Generate the ``k``-th derivative of the ``N``-th order Vandermoned matrix.

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum polynomial order
          k (int): order of the derivative

        Returns:
          (:class:`ndarray<ndarray>` [``m``,``N+1``]) -- polynomials evaluated
            at the ``r`` points.
        """
        DVr = np.zeros((r.shape[0],N+1))
        for i in range(0,N+1):
            DVr[:,i] = self.GradEvaluate(r, i, k, norm)
        return DVr

    def AssemblyDerivativeMatrix(self, x, N, k):
        r""" Assemble the ``k``-th derivative matrix using polynomials of order ``N``.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum polynomial order
          k (int): order of the derivative

        Returns:
          (:class:`ndarray<ndarray>` [``m``,``m``]) -- derivative matrix
        """
        V = self.GradVandermonde(x, N, 0)
        Vx = self.GradVandermonde(x, N ,1)
        D = LA.solve(V.T, Vx.T)
        D = D.T
        Dk = np.asarray(np.mat(D)**k)
        return Dk

    def DerivativeMatrix(self, x, N, k):
        return self.AssemblyDerivativeMatrix(x, N, k)

    def project(self, r, f, N):
        r""" Project ``f`` onto the span of the polynomial up to order ``N``.

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` evaluation points
          f (:class:`ndarray<ndarray>` [``m``]): function value at the ``m``
            evaluation points
          N (int): maximum polynomial order

        Returns:
          (:class:`ndarray<ndarray>` [``N``]) -- projection coefficients
        """
        # The standard approach is to use the Vandermonde matrix
        V = self.GradVandermonde(r, N, 0)
        if ( V.shape[0] == V.shape[1] ):
            fhat = LA.solve(V, f)
        else:
            (fhat, residues, rank, s) = LA.lstsq(V, f)
        return fhat
        
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
        fhat = self.project(x, f, order)
        # Use the generalized vandermonde matrix
        V = self.GradVandermonde(xi, order, 0)
        fi = np.dot(V,fhat)
        return fi

    @staticmethod
    def from_xml_element(node):
        r""" Given an xml node instantiate and return the corresponding object.
        """
        import SpectralToolbox.Spectral1D as S1D
        basis_node = node.find(S1D.XML_NAMESPACE + 'HermiteProbabilistsPolynomial')
        if basis_node is not None:
            return S1D.HermiteProbabilistsPolynomial.from_xml_element(basis_node)
        basis_node = node.find(S1D.XML_NAMESPACE + 'ConstantExtendedHermiteProbabilistsFunction')
        if basis_node is not None:
            return S1D.ConstantExtendedHermiteProbabilistsFunction.from_xml_element(basis_node)
        basis_node = node.find(S1D.XML_NAMESPACE + 'HermiteProbabilistsRadialBasisFunction')
        if basis_node is not None:
            return S1D.HermiteProbabilistsRadialBasisFunction.from_xml_element(basis_node)
        basis_node = node.find(S1D.XML_NAMESPACE + \
                               'ConstantExtendedHermiteProbabilistsRadialBasisFunction')
        if basis_node is not None:
            return S1D.ConstantExtendedHermiteProbabilistsRadialBasisFunction.from_xml_element(
                basis_node)
        basis_node = node.find(S1D.XML_NAMESPACE + \
                               'LinearExtendedHermiteProbabilistsRadialBasisFunction')
        if basis_node is not None:
            return S1D.LinearExtendedHermiteProbabilistsRadialBasisFunction.from_xml_element(
                basis_node)
        raise NotImplementedError("The from_xml_element method is not implemented for " + \
                                  "the selected basis")

class OrthogonalBasis(Basis):
    r""" This is an abstract class for 1-d orthogonal basis

    Args:
      normalized (bool): whether to normalize the polynomials. Default=``None``
        which leaves the choice at evaluation time.

    """

    def __init__(self, normalized=None):
        self.normalized = normalized
    
    def Evaluate(self, r, N, norm=True):
        r""" Evaluate the ``N``-th order polynomial at points ``r``

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomial
          N (int): order of the polynomial
          norm (bool): whether to return normalized (``True``) or unnormalized
            (``False``) polynomial. The parameter is ignored if the ``normalized``
            parameter is provided at construction time.

        Returns:
          (:class:`ndarray<ndarray>` [``m``]) -- polynomial evaluated on
             the ``r`` points.

        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")
    
    def GradEvaluate(self, r, N, k=0, norm=True):
        r""" Evaluate the ``k``-th derivative of the ``N``-th order polynomial at points ``r``

        Args:
          r (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomial
          N (int): order of the polynomial
          k (int): order of the derivative
          norm (bool): whether to return normalized (``True``) or unnormalized
            (``False``) polynomial. The parameter is ignored if the ``normalized``
            parameter is provided at construction time.

        Returns:
          (:class:`ndarray<ndarray>` [``m``]) -- ``k``-th derivative of the
            polynomial evaluated on the ``r`` points.
        
        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")

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

    def Gamma(self, N):
        r""" Return the normalization constant for the ``N``-th polynomial.

        Args:
          N (int): polynomial order

        Returns:
          g (float): normalization constant

        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")

    def Quadrature(self, N, quadType=None, norm=False):
        r""" Generate quadrature rules of the selected type.

        Args:
          N (int): polynomial order
          quadType (string): quadrature type
          norm (bool): whether to normalize the weights

        Returns:
          (:class:`tuple<tuple>` of :class:`ndarray<ndarray>`) -- format
            ``(x,w)`` where ``x`` and ``w`` are ``N+1`` nodes and weights

        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")
    
class OrthogonalPolynomial(OrthogonalBasis):
    """ This is an abstract class for 1-d polynomials
    """

    def RecursionCoeffs(self, N):
        r""" Generate the first ``N`` recursion coefficients.

        These coefficients define the polynomials:
        :math:`P_{n+1}(x) = (x - a_n) P_n(x) - b_n P_{n-1}(x)`

        Args:
          N (int): number of recursion coefficients

        Returns:
          (:class:`tuple<tuple>` [2] of :class:`ndarray<ndarray>` [N]) --
            recursion coefficients ``a`` and ``b``

        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")

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
        """
        (a,b) = self.RecursionCoeffs(N)
        if self.normalized is None:
            c = polymod.monomials(a, b, norm)
        else:
            c = polymod.monomials(a, b, self.normalized)
        return c

    def GaussQuadrature(self, N, norm=False):
        r""" Generate Gauss quadrature nodes and weights.

        Args:
          N (int): polynomial order
          norm (bool): whether to normalize the weights

        Returns:
          (:class:`tuple<tuple>` of :class:`ndarray<ndarray>`) -- format
            ``(x,w)`` where ``x`` and ``w`` are ``N+1`` nodes and weights

        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")

    def GaussLobattoQuadrature(self, N, norm=False):
        r""" Generate Gauss Lobatto quadrature nodes and weights.

        Args:
          N (int): polynomial order
          norm (bool): whether to normalize the weights
        
        Returns:
          (:class:`tuple<tuple>` of :class:`ndarray<ndarray>`) -- format
            ``(x,w)`` where ``x`` and ``w`` are ``N+1`` nodes and weights

        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")

    def GaussRadauQuadrature(self, N, norm=False):
        r""" Generate Gauss Radau quadrature nodes and weights.

        Args:
          N (int): polynomial order
          norm (bool): whether to normalize the weights
        
        Returns:
          (:class:`tuple<tuple>` of :class:`ndarray<ndarray>`) -- format
            ``(x,w)`` where ``x`` and ``w`` are ``N+1`` nodes and weights

        Raises:
          NotImplementedError: if not overridden.
        """
        raise NotImplementedError("Not implemented or undefined for this class.")

    def Quadrature(self, N, quadType=None, norm=False):
        r""" Generate quadrature rules of the selected type.

        Args:
          N (int): polynomial order
          quadType (string): quadrature type (Guass, Gauss-Lobatto, Gauss-Radau)
          norm (bool): whether to normalize the weights

        Returns:
          (:class:`tuple<tuple>` of :class:`ndarray<ndarray>`) -- format
            ``(x,w)`` where ``x`` and ``w`` are ``N+1`` nodes and weights

        Raises:
          ValueError: if quadrature type is not available
        """
        if quadType in [None, GAUSS]:
            return self.GaussQuadrature(N, norm)
        elif quadType == GAUSSLOBATTO:
            return self.GaussLobattoQuadrature(N-2, norm)
        elif quadType == GAUSSRADAU:
            return self.GaussRadauQuadrature(N-1, norm)
        else:
            raise ValueError("quadType=%s not available" % quadType)
