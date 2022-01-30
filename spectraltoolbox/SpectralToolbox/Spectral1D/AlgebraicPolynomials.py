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
# Copyright (C) 2015-2017 Massachusetts Institute of Technology
# Uncertainty Quantification group
# Department of Aeronautics and Astronautics
#
# Author: Daniele Bigoni
# E-mail: dabi@limitcycle.it
#

import numpy as np
from scipy.special import erf
from scipy.special import factorial2
from math import factorial

from SpectralToolbox.Spectral1D.AbstractClasses import OrthogonalPolynomial
from SpectralToolbox.Spectral1D.NonOrthogonalBasis \
    import ConstantExtendedHermitePhysicistsFunction

__all__ = ['SquaredOrthogonalPolynomial',
           'SquaredConstantExtendedHermitePhysicistsFunction',
           'PositiveDefiniteSquaredConstantExtendedHermitePhysicistsFunction',
           'SquaredConstantExtendedHermiteProbabilistsFunction',
           'PositiveDefiniteSquaredConstantExtendedHermiteProbabilistsFunction']

class SquaredOrthogonalPolynomial(object):
    r""" Given an :class:`OrthogonalPolynomial` object, provide the evaluation of the corresponding integrated squared polynomial

    Given the polynomials :math:`\{\phi_i\}`, and defining :math:`H^{(0)}_{i,j}=\phi_i \phi_j`,
    provide the interface for the evaluation of
    
    .. math::

      H^{(k)}_{i,j}(x) = \begin{cases}
      \frac{d}{dx} H^{(k-1)}_{i,j}(x)  & k > 0
      \int_0^x H^{(k+1)}_{i,j}(t) dt   & k < 0
      \end{cases}

    Args:
      base_poly (:class:`OrthogonalPolynomial`): base polynomial :math:`\{\phi_i\}`
    """
    def __init__(self, base_poly):
        if not isinstance(base_poly, OrthogonalPolynomial):
            raise ValueError("The base polynomial must be an orthogonal polynomial")
        self.base_poly = base_poly

    def GradEvaluate(self, x, n, m, k=0, norm=True, mem=None):
        r""" ``k``-th derivative/antiderivative of the ``n,m`` squared orthogonal polynomial.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          n (int): order of the first polynomial
          m (int): order of the second polynomial
          k (int): integration or derivative order.
            ``k==0`` evaluates :math:`\phi_n \phi_m`
            ``k>0`` evaluates :math:`(\phi_n \phi_m)^{(k)}` (derivatives)
            ``k<0`` evaluates :math:`\int_0^x \cdots \int_0^{x_{-k-1}} \phi_n \phi_m`
            (integration)
          norm (bool): whether the polynomials are normalized
          mem (list): memoization data structure
        """
        if k == 0:
            p1 = self.base_poly.GradEvaluate(x, n, norm=norm)
            p2 = self.base_poly.GradEvaluate(x, m, norm=norm)
            out = p1 * p2
        elif k > 0:
            raise NotImplementedError("!")
        elif k < 0:
            # Recursive implementation with memoization
            if n < m: # Symmetry
                n, m = m, n
            # Initialize memoization data structure if None
            if mem is None:
                mem = [[] for i in range((n+1)*2)]
            memk = mem[0]
            if len(memk) == 0:
                for i in range(n+1):
                    memk.append( [None] * (i+1) )
            # Retrieve recursion coefficients
            (a, b) = self.base_poly.RecursionCoeffs(n)
            if n == 0 and m == 0:
                if memk[0][0] is None:
                    out = x**-k / factorial(-k)
                    if norm:
                        out /= b[0]
                    memk[0][0] = out
                else:
                    out = memk[0][0]
                return out
            if n == 1 and m == 0:  # Define it for n==0 and m>=1 (pack with first condition)
                if memk[n][m] is None:
                    out = x**(-k) / factorial(-k) * ( x/(-k+1) - a[0] )
                    if norm:
                        out /= np.sqrt(b[1]) * b[0]
                    memk[n][m] = out
                else:
                    out = memk[n][m]
                return out
            if n == 1 and m == 1:
                if memk[n][m] is None:
                    out = x**(-k) / factorial(-k) * \
                          ( 2*x**2./((-k)**2.-3*k+2) - 2*x/(-k+1)*a[0] + a[0]**2 )
                    if norm:
                        out /= b[0] * b[1]
                    memk[n][m] = out
                else:
                    out = memk[n][m]
                return out
            if n > 1 and m == 0:
                if memk[n][m] is None:
                    Hk1n1 = self.GradEvaluate(x, n-1, 0, k=k-1, norm=norm, mem=mem[1:])
                    Hkn1 = self.GradEvaluate(x, n-1, 0, k=k, norm=norm, mem=mem)
                    Hkn2 = self.GradEvaluate(x, n-2, 0, k=k, norm=norm, mem=mem)
                    an = a[n-1]
                    if not norm:
                        bnm1 = b[n-1]
                        bn = 1.
                    else:
                        bnm1 = np.sqrt(b[n-1])
                        bn = np.sqrt(b[n])
                    out = 1/bn * ( (x-an) * Hkn1 - bnm1 * Hkn2 - (-k) * Hk1n1 )
                    memk[n][m] = out
                else:
                    out = memk[n][m]
                return out
            if n > 1 and m == 1:
                if memk[n][m] is None:
                    Hk2n1 = self.GradEvaluate(x, n-1, 0, k=k-2, norm=norm, mem=mem[2:])
                    Hk1n1 = self.GradEvaluate(x, n-1, 0, k=k-1, norm=norm, mem=mem[1:])
                    Hk1n2 = self.GradEvaluate(x, n-2, 0, k=k-1, norm=norm, mem=mem[1:])
                    Hkn1 = self.GradEvaluate(x, n-1, 0, k=k, norm=norm, mem=mem)
                    Hkn2 = self.GradEvaluate(x, n-2, 0, k=k, norm=norm, mem=mem)
                    a0 = a[0]
                    an = a[n-1]
                    if not norm:
                        bnm1 = b[n-1]
                        bn = 1.
                        b1 = 1.
                    else:
                        bnm1 = np.sqrt(b[n-1])
                        bn = np.sqrt(b[n])
                        b1 = np.sqrt(b[1])
                    out = (b1*bn)**-1 * ( (x**2-(a0+an)*x+a0*an) * Hkn1 \
                                          - (a0*bnm1+bnm1*x) * Hkn2 \
                                          + (-k) * (a0+an-2*x) * Hk1n1 \
                                          + (-k) * bnm1 * Hk1n2 \
                                          + (-k+1) * (-k) * Hk2n1 )
                    memk[n][m] = out
                else:
                    out = memk[n][m]
                return out
            if memk[n][m] is None:
                Hk2n1m1 = self.GradEvaluate(x, n-1, m-1, k=k-2, norm=norm, mem=mem[2:])
                Hk1n1m1 = self.GradEvaluate(x, n-1, m-1, k=k-1, norm=norm, mem=mem[1:])
                Hk1n1m2 = self.GradEvaluate(x, n-1, m-2, k=k-1, norm=norm, mem=mem[1:])
                Hk1n2m1 = self.GradEvaluate(x, n-2, m-1, k=k-1, norm=norm, mem=mem[1:])
                Hkn1m1 = self.GradEvaluate(x, n-1, m-1, k=k, norm=norm, mem=mem)
                Hkn1m2 = self.GradEvaluate(x, n-1, m-2, k=k, norm=norm, mem=mem)
                Hkn2m1 = self.GradEvaluate(x, n-2, m-1, k=k, norm=norm, mem=mem)
                Hkn2m2 = self.GradEvaluate(x, n-2, m-2, k=k, norm=norm, mem=mem)
                an = a[n-1]
                am = a[m-1]
                if not norm:
                    bnm1 = b[n-1]
                    bmm1 = b[m-1]
                    bn = 1.
                    bm = 1.
                else:
                    bnm1 = np.sqrt(b[n-1])
                    bmm1 = np.sqrt(b[m-1])
                    bn = np.sqrt(b[n])
                    bm = np.sqrt(b[m])
                out = (bn*bm)**-1 * ( (x**2. - (am + an)*x + an*am) * Hkn1m1 + \
                                      (an*bmm1 - bmm1*x) * Hkn1m2 + \
                                      (am*bnm1 - bnm1*x) * Hkn2m1 + \
                                      bnm1 * bmm1 * Hkn2m2 + \
                                      (-k) * (am + an - 2*x) * Hk1n1m1 + \
                                      (-k) * bmm1 * Hk1n1m2 + \
                                      (-k) * bnm1 * Hk1n2m1 + \
                                      (-k+1) * (-k) * Hk2n1m1 )
                memk[n][m] = out
            else:
                out = memk[n][m]
        return out

    def GradVandermonde(self, x, N, k=0, norm=True):
        r""" ``k``-th derivative/antiderivative of the generalized Vandermonde matrix of the ``N``-th squared orthogonal polynomials.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum order
          k (int): integration or derivative order.
            ``k==0`` evaluates :math:`\phi_n \phi_m`
            ``k>0`` evaluates :math:`(\phi_n \phi_m)^{(k)}` (derivatives)
            ``k<0`` evaluates :math:`\int_0^x \cdots \int_0^{x_{-k-1}} \phi_n \phi_m`
            (integration)
          norm (bool): whether the polynomials are normalized

        Returns:
          (:class:`ndarray<ndarray>` [``m,N,N``]): Product Vandermonde matrix
        """
        if k < 0:
            mem = [[] for i in range((N+1)*2)]
        else:
            mem = None
        npoints = x.shape[0]
        out = np.zeros((npoints, N+1, N+1))
        # Fill lower triangular part starting (N,N) -- necessary for memoization
        for i in range(N,-1,-1):
            for j in range(i,-1,-1):
                out[:,i,j] = self.GradEvaluate(x, i, j, k=k, norm=norm, mem=mem)
        # Symmetrize
        for i in range(N):
            for j in range(i+1,N+1):
                out[:,i,j] = out[:,j,i]
        return out

class SquaredConstantExtendedHermitePhysicistsFunction(object):
    r""" Provide the evaluation of the corresponding integrated squared functions for :class:`ConstantExtendedOrthogonalHermitePhysicistsFunction`

    Given the functions :math:`\{\phi_i\}`, and defining :math:`H^{(0)}_{i,j}=\phi_i \phi_j`,
    provide the interface for the evaluation of
    
    .. math::

      H^{(k)}_{i,j}(x) = \begin{cases}
      \frac{d}{dx} H^{(k-1)}_{i,j}(x)  & k > 0
      \int_0^x H^{(k+1)}_{i,j}(t) dt   & k < 0
      \end{cases}
    """
    
    def __init__(self):
        self.base_fun = ConstantExtendedHermitePhysicistsFunction()
        self.pi_erf_c_mat = None
        self.exp_c_mat = None
        self.mask = None

    def get_monomial_coeffs(self, n, m, k=-1, norm=True):
        r""" ``n,m`` monomials coefficiants

        Args:
          n (int): order of the first polynomial
          m (int): order of the second polynomial
          k (int): only k==-1 implemented
          norm (bool): whether the polynomials are normalized
        """
        if k != -1:
            raise NotImplementedError("k must be -1")
        if self.pi_erf_c_mat is None:
            self.const_mat = np.zeros((n+1,m+1))
            self.pi_erf_c_mat = np.zeros((n+1,m+1))
            self.exp_c_mat = np.zeros((n+1,m+1), dtype=object)
            self.mask = np.zeros((n+1,m+1), dtype=bool)
        if self.pi_erf_c_mat.shape[0] < n+1 or self.pi_erf_c_mat.shape[1] < m+1:
            nold = self.pi_erf_c_mat.shape[0]
            mold = self.pi_erf_c_mat.shape[1]
            nnew = max(n, nold)
            mnew = max(m, mold)
            # const
            new_mat = np.zeros((nnew+1,mnew+1))
            new_mat[:nold,:mold] = self.const_mat
            self.const_mat = new_mat
            # pi_erf_c
            new_mat = np.zeros((nnew+1,mnew+1))
            new_mat[:nold,:mold] = self.pi_erf_c_mat
            self.pi_erf_c_mat = new_mat
            # exp_c
            new_mat = np.zeros((nnew+1,mnew+1), dtype=object)
            new_mat[:nold,:mold] = self.exp_c_mat
            self.exp_c_mat = new_mat
            # mask
            new_mat = np.zeros((nnew+1,mnew+1), dtype=bool)
            new_mat[:nold,:mold] = self.mask
            self.mask = new_mat
        if n < self.pi_erf_c_mat.shape[0] and m < self.pi_erf_c_mat.shape[1] \
           and not self.mask[n,m]:
            # Compute
            if n == 0 or m == 0:
                ntot = n+m-1  # -1 for ConstantExtended
                hpoly = self.base_fun.hpf.H
                c = hpoly.MonomialCoeffs(ntot, norm=norm)
                # Compute integral using monomial formula.
                # First compute coefficients for erf and exp parts
                const = 0.
                pi_erf_c = 0.  # * np.sqrt(np.pi/2.) * erf(x/np.sqrt(2.))
                exp_c = np.zeros(ntot) # * exp(-x**2/2.)
                for nn in range(ntot+1):
                    if nn % 2 == 0:
                        pi_erf_c += c[nn] * factorial2(nn-1,True)
                        for k in range(1,nn//2+1):
                            exp_c[nn-2*k+1] += c[nn] * \
                                               (factorial2(nn-1,True) \
                                                // factorial2(nn-2*k+1,True))
                    else:
                        const += c[nn] * factorial2(nn-1,True)
                        for k in range(0,(nn-1)//2+1):
                            exp_c[nn-2*k-1] += c[nn] * \
                                               (factorial2(nn-1,True) \
                                                // factorial2(nn-2*k-1,True))
            else:
                mr = m - 1  # -1 for ConstantExtended
                nr = n - 1  # -1 for ConstantExtended
                ntot = m+n
                hpoly = self.base_fun.hpf.H
                c1 = hpoly.MonomialCoeffs(nr, norm=norm)
                c2 = hpoly.MonomialCoeffs(mr, norm=norm)
                ch2 = np.zeros(ntot+1) # Coefficients of (\phi_n\phi_m)
                for i in range(nr+1):
                    for j in range(mr+1):
                        ch2[i+j] += c1[i] * c2[j]
                # Compute integral using monomial formula.
                # First compute coefficients for the erf and exp part
                const = 0.
                pi_erf_c = 0.
                exp_c = np.zeros(ntot)
                for nn in range(ch2.shape[0]):
                    if nn % 2 == 0 and ch2[nn] != 0.:
                        pi_erf_c += ch2[nn] * factorial2(nn-1,True) / 2**(nn/2+1)
                        for k in range(1,nn//2+1):
                            exp_c[nn-2*k+1] += ch2[nn] * \
                                               (factorial2(nn-1,True) * 2**(nn//2-k) \
                                                / factorial2(nn-2*k+1,True)) \
                                               / 2.**(nn//2)
                    elif nn % 2 == 1 and ch2[nn] != 0:
                        N = (nn-1)//2
                        const += ch2[nn] * factorial(N) / 2.
                        if N == 0:
                            exp_c[0] += ch2[nn] / 2.
                        else:
                            for k in range(N+1):
                                exp_c[2*k] += ch2[nn] * \
                                              (N * factorial(N-1) \
                                               / factorial(k)) / 2.
            self.const_mat[n,m] = const
            self.pi_erf_c_mat[n,m] = pi_erf_c
            self.exp_c_mat[n,m] = exp_c
            self.mask[n,m] = True
        return (self.const_mat[n,m], self.pi_erf_c_mat[n,m], self.exp_c_mat[n,m])

    def GradEvaluate(self, x, n, m, k=0, norm=True):
        r""" ``k``-th derivative/antiderivative of the ``n,m`` squared constant extended Hermite functions.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          n (int): order of the first polynomial
          m (int): order of the second polynomial
          k (int): integration or derivative order.
            ``k==0`` evaluates :math:`\phi_n \phi_m`
            ``k>0`` evaluates :math:`(\phi_n \phi_m)^{(k)}` (derivatives)
            ``k<0`` evaluates :math:`\int_0^x \cdots \int_0^{x_{-k-1}} \phi_n \phi_m`
            (integration)
          norm (bool): whether the polynomials are normalized
        """
        if k == 0:
            p1 = self.base_fun.GradEvaluate(x, n, norm=norm)
            p2 = self.base_fun.GradEvaluate(x, m, norm=norm)
            return p1 * p2
        elif k > 0:
            raise NotImplementedError("!")
        elif k == -1:
            if n == m == 0:
                return x.copy()
            elif n == 0 or m == 0:
                # Evaluate erf and exp
                ev_pi_erf = np.sqrt(np.pi/2.) * erf(x/np.sqrt(2.))
                ev_exp = np.exp(-x**2./2.)
            else:
                ev_pi_erf = np.sqrt(np.pi) * erf(x)
                ev_exp = np.exp(-x**2.)
            const, pi_erf_c, exp_c = self.get_monomial_coeffs(n, m, k=k, norm=norm)
            ntot = exp_c.shape[0]
            # Build the matrix of x^n
            X = np.ones((x.shape[0],ntot))
            for i in range(1,ntot):
                X[:,i] = X[:,i-1] * x
            # Evaluate integrals
            out = const + pi_erf_c * ev_pi_erf - np.dot(X, exp_c) * ev_exp
            return out
        else:
            raise NotImplementedError("!")

    def GradVandermonde(self, x, N, k=0, norm=True):
        r""" ``k``-th derivative/antiderivative of the generalized Vandermonde matrix of the ``N``-th squared functions.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum order
          k (int): integration or derivative order.
            ``k==0`` evaluates :math:`\phi_n \phi_m`
            ``k>0`` evaluates :math:`(\phi_n \phi_m)^{(k)}` (derivatives)
            ``k<0`` evaluates :math:`\int_0^x \cdots \int_0^{x_{-k-1}} \phi_n \phi_m`
            (integration)
          norm (bool): whether the polynomials are normalized

        Returns:
          (:class:`ndarray<ndarray>` [``m,N,N``]): Product Vandermonde matrix
        """
        npoints = x.shape[0]
        out = np.zeros((npoints, N+1, N+1))
        # Fill lower triangular part starting (N,N)
        for i in range(N,-1,-1):
            for j in range(i,-1,-1):
                out[:,i,j] = self.GradEvaluate(x, i, j, k=k, norm=norm)
        # Symmetrize
        for i in range(N):
            for j in range(i+1,N+1):
                out[:,i,j] = out[:,j,i]
        return out

class PositiveDefiniteSquaredConstantExtendedHermitePhysicistsFunction(
        SquaredConstantExtendedHermitePhysicistsFunction):
    r""" Provide the evaluation of the corresponding integrated squared functions for :class:`ConstantExtendedOrthogonalHermitePhysicistsFunction` and impose positive definitness.

    Impose posite definitness by adding ``eps`` to the diagonal elements
    :math:`\phi_n\phi_n`.

    Args:
      eps (float): positive definitness constant

    .. seealso:: :class:`SquaredConstantExtendedHermitePhysicistsFunction`
    """
    def __init__(self, eps=1e-10):
        super(PositiveDefiniteSquaredConstantExtendedHermitePhysicistsFunction,
              self).__init__()
        self.eps = eps

    def GradEvaluate(self, x, n, m, k=0, norm=True):
        r""" ``k``-th derivative/antiderivative of the ``n,m`` positive definite squared constant extended Hermite functions.

        .. seealso:: :func:`SquaredConstantExtendedHermitePhysicistsFunction.GradEvaluate`
        """
        out = super(PositiveDefiniteSquaredConstantExtendedHermitePhysicistsFunction,
                    self).GradEvaluate(x, n, m, k=k, norm=norm)
        if n == m:
            if k == 0:
                out += self.eps
            elif k > 0:
                raise NotImplementedError("!")
            elif k == -1:
                out += self.eps * x
        return out

class SquaredConstantExtendedHermiteProbabilistsFunction(object):
    r""" Provide the evaluation of the corresponding integrated squared functions for :class:`ConstantExtendedOrthogonalHermiteProbabilistsFunction`

    Given the functions :math:`\{\phi_i\}`, and defining :math:`H^{(0)}_{i,j}=\phi_i \phi_j`,
    provide the interface for the evaluation of
    
    .. math::

      H^{(k)}_{i,j}(x) = \begin{cases}
      \frac{d}{dx} H^{(k-1)}_{i,j}(x)  & k > 0
      \int_0^x H^{(k+1)}_{i,j}(t) dt   & k < 0
      \end{cases}
    """
    def __init__(self):
        self.base_sq_fun = SquaredConstantExtendedHermitePhysicistsFunction()

    def GradEvaluate(self, x, n, m, k=0, norm=True):
        r""" ``k``-th derivative/antiderivative of the ``n,m`` squared constant extended Hermite functions.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          n (int): order of the first polynomial
          m (int): order of the second polynomial
          k (int): integration or derivative order.
            ``k==0`` evaluates :math:`\phi_n \phi_m`
            ``k>0`` evaluates :math:`(\phi_n \phi_m)^{(k)}` (derivatives)
            ``k<0`` evaluates :math:`\int_0^x \cdots \int_0^{x_{-k-1}} \phi_n \phi_m`
            (integration)
          norm (bool): whether the polynomials are normalized
        """
        out = self.base_sq_fun.GradEvaluate(x/np.sqrt(2.), n, m, k=k, norm=norm)
        if k == 0 and norm:
            if (m==0 and n>=1) or (n==0 and m>=1):
                out /= np.sqrt(np.sqrt(2.))
            elif m >= 1 and n >= 1:
                out /= np.sqrt(2.)
        if k != 0:
            if norm:
                if m == 0 and n == 0: # Remove /sqrt(2.) zero-th function
                    out *= np.sqrt(2.)
                elif (m == 0 or n == 0):
                    out *= np.sqrt(np.sqrt(2.))
            else:
                out *= np.sqrt(2.)
        return out

    def GradVandermonde(self, x, N, k=0, norm=True):
        r""" ``k``-th derivative/antiderivative of the generalized Vandermonde matrix of the ``N``-th squared functions.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum order
          k (int): integration or derivative order.
            ``k==0`` evaluates :math:`\phi_n \phi_m`
            ``k>0`` evaluates :math:`(\phi_n \phi_m)^{(k)}` (derivatives)
            ``k<0`` evaluates :math:`\int_0^x \cdots \int_0^{x_{-k-1}} \phi_n \phi_m`
            (integration)
          norm (bool): whether the polynomials are normalized

        Returns:
          (:class:`ndarray<ndarray>` [``m,N,N``]): Product Vandermonde matrix
        """
        npoints = x.shape[0]
        out = np.zeros((npoints, N+1, N+1))
        # Fill lower triangular part starting (N,N)
        for i in range(N,-1,-1):
            for j in range(i,-1,-1):
                out[:,i,j] = self.GradEvaluate(x, i, j, k=k, norm=norm)
        # Symmetrize
        for i in range(N):
            for j in range(i+1,N+1):
                out[:,i,j] = out[:,j,i]
        return out

class PositiveDefiniteSquaredConstantExtendedHermiteProbabilistsFunction(
        SquaredConstantExtendedHermiteProbabilistsFunction):
    r""" Provide the evaluation of the corresponding integrated squared functions for :class:`ConstantExtendedOrthogonalHermiteProbabilistsFunction` and impose positive definitness.

    Impose posite definitness by adding ``eps`` to the diagonal elements
    :math:`\phi_n\phi_n`.

    Args:
      eps (float): positive definitness constant

    .. seealso:: :class:`SquaredConstantExtendedHermiteProbabilistsFunction`
    """
    def __init__(self, eps=1e-10):
        super(PositiveDefiniteSquaredConstantExtendedHermiteProbabilistsFunction,
              self).__init__()
        self.eps = eps

    def GradEvaluate(self, x, n, m, k=0, norm=True):
        r""" ``k``-th derivative/antiderivative of the ``n,m`` positive definite squared constant extended Hermite functions.

        .. seealso:: :func:`SquaredConstantExtendedHermiteProbabilistsFunction.GradEvaluate`
        """
        out = super(PositiveDefiniteSquaredConstantExtendedHermiteProbabilistsFunction,
                    self).GradEvaluate(x, n, m, k=k, norm=norm)
        if n == m:
            if k == 0:
                out += self.eps
            elif k > 0:
                raise NotImplementedError("!")
            elif k == -1:
                out += self.eps * x
        return out

    def GradVandermonde(self, x, N, k=0, norm=True):
        r""" ``k``-th derivative/antiderivative of the generalized Vandermonde matrix of the ``N``-th squared functions.

        Args:
          x (:class:`ndarray<ndarray>` [``m``]): set of ``m`` points where to
            evaluate the polynomials
          N (int): maximum order
          k (int): integration or derivative order.
            ``k==0`` evaluates :math:`\phi_n \phi_m`
            ``k>0`` evaluates :math:`(\phi_n \phi_m)^{(k)}` (derivatives)
            ``k<0`` evaluates :math:`\int_0^x \cdots \int_0^{x_{-k-1}} \phi_n \phi_m`
            (integration)
          norm (bool): whether the polynomials are normalized

        Returns:
          (:class:`ndarray<ndarray>` [``m,N,N``]): Product Vandermonde matrix
        """
        npoints = x.shape[0]
        out = np.zeros((npoints, N+1, N+1))
        # Fill lower triangular part starting (N,N)
        for i in range(N,-1,-1):
            for j in range(i,-1,-1):
                out[:,i,j] = self.GradEvaluate(x, i, j, k=k, norm=norm)
        # Symmetrize
        for i in range(N):
            for j in range(i+1,N+1):
                out[:,i,j] = out[:,j,i]
        return out