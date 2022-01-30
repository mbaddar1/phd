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

import unittest
import math
import numpy as np
import numpy.linalg as npla
import scipy.stats as stats
import scipy.special as special

import SpectralToolbox.Spectral1D as S1D

def plot_vandermonde(x,V,N,title=""):
    import matplotlib.pyplot as plt 
    plt.figure()
    for i in range(N+1):
        plt.plot(x,V[:,i],label='ord=%d'%i)
    plt.legend()
    plt.title(title)
    plt.show(False)

def plot_interpolate(x, F, Fapp, title=""):
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.plot(x,F,label='exact')
    plt.plot(x,Fapp,label='approx')
    plt.legend()
    plt.title(title)
    plt.show(False)

def plot_function(x, F, title=""):
    import matplotlib.pyplot as plt 
    plt.figure()
    plt.plot(x,F)
    plt.show(False)

class TestBasis(unittest.TestCase):
    def setUp(self):
        self.eps = 1e-10
        self.eps_monomials = 1e-8
        self.eps_interpolate = 1e-10
        self.min_ord = 2
        self.max_ord = 41
        self.max_ord_low = 10
        self.max_der = np.inf
        self.N_discr = 1000
        self.N_samp = 100000
        # Defined in sub-classes
        self.integ = self.get_analytic_integral()
        self.P = self.get_basis()
        self.sampler = self.get_sampler()

    def get_analytic_integral(self):
        raise NotImplementedError("Define the analytic integral with respect " +
                                  "to the polynomial sub-class")

    def get_basis(self):
        raise NotImplementedError("Define the basis from the sub-class")

    def get_sampler(self):
        raise NotImplementedError("Define the sampler from the sub-class")

    def test_quadrature(self):
        success = False
        i = -1
        while (not success and i <= self.max_ord):
            i += 1
            (x,w) = self.P.Quadrature(i, norm=False)
            integ = np.dot(w, self.f(x))
            success =  np.abs(integ - self.integ) < self.eps
        self.assertTrue( success )

    def test_interpolate(self):
        success = False
        i = self.min_ord
        samples = self.sampler.rvs(self.N_samp)
        samples = np.sort(samples)
        fexa = self.f(samples)
        while (not success and i <= self.max_ord):
            (x,w) = self.P.Quadrature(i)
            f = self.f(x)
            fi = self.P.interpolate(x, f, samples, i)
            success = np.sqrt( np.sum( (fexa - fi)**2. ) / self.N_samp ) < self.eps_interpolate
            i += 1
        if not success:
            plot_interpolate(samples,fexa,fi,title=self.__class__.__name__)
        self.assertTrue( success )

    def test_derivatives(self):
        success = True
        (x,w) = self.P.Quadrature(self.max_ord_low)
        for i in range(min(5,self.max_der)):
            dV = self.P.GradVandermonde(x, self.max_ord_low, k=i+1, norm=False)
            # Finite difference
            eps = 1e-5
            Vp = self.P.GradVandermonde(x+eps/2., self.max_ord_low, k=i, norm=False)
            Vm = self.P.GradVandermonde(x-eps/2., self.max_ord_low, k=i, norm=False)
            dVfd = (Vp-Vm)/eps
            # Check
            max_err = np.max( np.abs(dV-dVfd)/(np.abs(dVfd)+1) )
            success = np.isclose( max_err, 0., atol=10.*eps )
            if not success:
                plot_vandermonde(x,dV,5,'%s - exact' % self.__class__.__name__)
                plot_vandermonde(x,dVfd,5,'%s - FD' % self.__class__.__name__)
                break
        self.assertTrue( success )
        
    def test_normalized_derivatives(self):
        success = True
        (x,w) = self.P.Quadrature(self.max_ord_low)
        for i in range(min(5,self.max_der)):
            dV = self.P.GradVandermonde(x, self.max_ord_low, k=i+1)
            # Finite difference
            eps = 1e-5
            Vp = self.P.GradVandermonde(x+eps/2., self.max_ord_low, k=i)
            Vm = self.P.GradVandermonde(x-eps/2., self.max_ord_low, k=i)
            dVfd = (Vp-Vm)/eps
            # Check
            max_err = np.max( np.abs(dV-dVfd)/(np.abs(dVfd)+1) )
            success = np.isclose( max_err, 0., atol=10.*eps )
            if not success:
                plot_vandermonde(x,dV,5,'%s - exact' % self.__class__.__name__)
                plot_vandermonde(x,dVfd,5,'%s - FD' % self.__class__.__name__)
                break
        self.assertTrue( success )

class TestOrthogonalBasis(TestBasis):
    
    def setUp(self):
        super(TestOrthogonalBasis,self).setUp()        

    def test_gamma(self):
        fail = False
        i = -1
        while (not fail and i <= self.max_ord):
            i += 1
            gamma = self.P.Gamma(i)
            (x,w) = self.P.Quadrature(i)
            p = self.P.GradEvaluate(x, i, norm=False)
            gapp = np.dot(p, w * p)
            fail = (np.abs( gamma - gapp ) >= self.eps * np.abs(gamma))
        self.assertFalse( fail )

    def test_grad_vandermonde(self):
        (x,w) = self.P.Quadrature(self.max_ord)
        V = self.P.GradVandermonde(x, self.max_ord)
        Iapp = np.dot(V.T, w[:,np.newaxis] * V)
        self.assertTrue( np.allclose( Iapp , np.eye(self.max_ord+1),
                                      atol=1e-12 ) )

    def test_mc_orthogonality(self):
        samples = self.sampler.rvs(self.N_samp)
        V = self.P.GradVandermonde(samples, self.max_ord_low)
        Iapp = np.dot(V.T, self.P.Gamma(0) * V / float(self.N_samp))
        success = np.allclose( Iapp , np.eye(self.max_ord_low+1), atol=1e-1 )
        self.assertTrue( success )

class TestPolynomial(TestOrthogonalBasis):

    def test_gauss_lobatto_quadrature(self):
        success = False
        i = 0
        while (not success and i <= self.max_ord):
            i += 1
            (x,w) = self.P.GaussLobattoQuadrature(i, norm=False)
            integ = np.dot(w, self.f(x))
            success =  np.abs(integ - self.integ) < self.eps
        self.assertTrue( success )

    def test_gauss_radau_quadrature(self):
        success = False
        i = 0
        while (not success and i <= self.max_ord):
            i += 1
            (x,w) = self.P.GaussRadauQuadrature(i, norm=False)
            integ = np.dot(w, self.f(x))
            success =  np.abs(integ - self.integ) < self.eps
        self.assertTrue( success )

    def test_monomials(self):
        success = True
        x = self.get_sampler().rvs(10)
        i = 0
        while i < self.max_ord_low and success:
            # Monomial
            c = self.P.MonomialCoeffs(i)
            X = np.zeros((10,i+1))
            for k in range(i+1):
                X[:,k] = x**k
            mon = np.dot(X, c)
            # Recursion
            rec = self.P.GradEvaluate(x, i)
            # Check result
            success = np.max(np.abs(mon - rec)) < self.eps_monomials
            i += 1
        self.assertTrue( success )
    
class TestLegendrePolynomial(TestPolynomial):

    def setUp(self):
        super(TestLegendrePolynomial,self).setUp()
        self.f = lambda x: np.exp(x)

    def get_analytic_integral(self):
        return np.exp(1.) - np.exp(-1.)

    def get_basis(self):
        return S1D.JacobiPolynomial(0.,0.)

    def get_sampler(self):
        return stats.uniform(-1.,2.)

class TestJacobiPolynomial(TestPolynomial):
    r""" Test for :math:`\text{Beta}(2,5)`

    Note that :math:`\rho_\beta(x,2,5) = 2 w(2x-1,4,1)`
    """

    def setUp(self):
        super(TestJacobiPolynomial,self).setUp()
        self.f = lambda x: np.exp(x)
    
    def get_analytic_integral(self):
        return (536./np.e - 72. * np.e)

    def get_basis(self):
        return S1D.JacobiPolynomial(4.,1.)

    def get_sampler(self):
        return stats.beta(2,5,loc=-1,scale=2)

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestJacobiPolynomial,self).test_mc_orthogonality()

class TestChebyshevFirstKindPolynomial(TestPolynomial):

    def setUp(self):
        super(TestChebyshevFirstKindPolynomial,self).setUp()
        self.f = lambda x: np.exp(x)
    
    def get_analytic_integral(self):
        return np.pi * special.i0(1)

    def get_basis(self):
        return S1D.JacobiPolynomial(-0.5,-0.5)

    def get_sampler(self):
        return stats.beta(0.5,0.5,loc=-1,scale=2) 

class TestChebyshevSecondKindPolynomial(TestPolynomial):

    def setUp(self):
        super(TestChebyshevSecondKindPolynomial,self).setUp()
        self.f = lambda x: np.exp(x)
    
    def get_analytic_integral(self):
        return np.pi * special.i1(1)

    def get_basis(self):
        return S1D.JacobiPolynomial(0.5,0.5)

    def get_sampler(self):
        return stats.beta(1.5,1.5,loc=-1,scale=2)

class TestLaguerreAlpha0Polynomial(TestPolynomial):

    def setUp(self):
        super(TestLaguerreAlpha0Polynomial,self).setUp()
        self.f = lambda x: np.sin(x/5.)

    def get_analytic_integral(self):
        return 5./26.

    def get_basis(self):
        return S1D.LaguerrePolynomial(0.)

    def get_sampler(self):
        return stats.gamma(1.)

    @unittest.skip("NotDef.")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestLaguerreAlpha0Polynomial,self).test_mc_orthogonality()

class TestLaguerreAlpha1Polynomial(TestPolynomial):

    def setUp(self):
        super(TestLaguerreAlpha1Polynomial,self).setUp()
        self.f = lambda x: np.sin(x/5.)

    def get_analytic_integral(self):
        return 125./338.

    def get_basis(self):
        return S1D.LaguerrePolynomial(1.)

    def get_sampler(self):
        return stats.gamma(2.)

    @unittest.skip("NotDef.")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestLaguerreAlpha0Polynomial,self).test_mc_orthogonality()

class TestHermitePhysicistsPolynomial(TestPolynomial):

    def setUp(self):
        super(TestHermitePhysicistsPolynomial,self).setUp()
        self.f = lambda x: np.exp(x)

    def get_analytic_integral(self):
        return np.exp(1./4.) * np.sqrt(np.pi)

    def get_basis(self):
        return S1D.HermitePhysicistsPolynomial()

    def get_sampler(self):
        return stats.norm(loc=0, scale=1./np.sqrt(2.))

    @unittest.skip("NotDef.")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.skip("NotDef.")
    def test_gauss_radau_quadrature(self):
        pass

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestLaguerreAlpha0Polynomial,self).test_mc_orthogonality()

class TestHermiteProbabilistsPolynomial(TestPolynomial):

    def setUp(self):
        super(TestHermiteProbabilistsPolynomial,self).setUp()
        self.f = lambda x: np.exp(x)

    def get_analytic_integral(self):
        return np.exp(1./2.) * np.sqrt(2.*np.pi)

    def get_basis(self):
        return S1D.HermiteProbabilistsPolynomial()

    def get_sampler(self):
        return stats.norm()

    @unittest.skip("NotDef.")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.skip("NotDef.")
    def test_gauss_radau_quadrature(self):
        pass

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestLaguerreAlpha0Polynomial,self).test_mc_orthogonality()

class TestGenericOrthogonalPolynomial(TestPolynomial):

    def setUp(self):
        self.d = 10
        super(TestGenericOrthogonalPolynomial,self).setUp()
        self.f = lambda x: np.exp(x)

    def get_analytic_integral(self):
        return 5./192. * ( 131. * np.exp(1./2.) * np.sqrt(2*np.pi) *
                          (math.erf(1./np.sqrt(2.)) + 1.) + 174. )

    def get_basis(self):
        def muPDF(x,i):
            d = self.d
            return (x**(d-1)/(2**(d/2.-1.) * math.gamma(d/2.))) * np.exp(-x**2./2.)
        endl = np.array([0.])
        endr = np.array([np.inf])
        return S1D.GenericOrthogonalPolynomial(muPDF, endl, endr)

    def get_sampler(self):
        class spherical_sampler:
            def __init__(self, d):
                self.d = d
                self.dist = stats.norm()
            def rvs(self,N):
                dsamp = self.dist.rvs(self.d*N).reshape((N,self.d))
                samp = npla.norm(dsamp,ord=2,axis=1)
                return samp
        return spherical_sampler(self.d)

    @unittest.skip("NotDef")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestGenericOrthogonalPolynomial,self).test_mc_orthogonality()

    @unittest.skip("NotImplemented")
    def test_normalized_derivatives(self):
        pass
        
    @unittest.skip("NotImplemented")
    def test_derivatives(self):
        pass

class TestFourierBasis(TestOrthogonalBasis):

    def setUp(self):
        super(TestFourierBasis,self).setUp()
        self.f = lambda x: np.exp(-(x-np.pi)**2./10.)
        self.max_ord = 1000
        self.eps = 1e-2
        self.eps_interpolate = 1e-2

    def get_analytic_integral(self):
        return np.sqrt(10.*np.pi) * math.erf(np.pi/np.sqrt(10.))

    def get_basis(self):
        return S1D.Fourier()

    def get_sampler(self):
        return stats.uniform(loc=0,scale=2.*np.pi)

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestFourierBasis,self).test_mc_orthogonality()

class TestHermitePhysicistsFunction(TestPolynomial):

    def setUp(self):
        super(TestHermitePhysicistsFunction,self).setUp()
        self.f = lambda x: np.exp(-x**2)

    def get_analytic_integral(self):
        return np.sqrt(np.pi)

    def get_basis(self):
        return S1D.HermitePhysicistsFunction()

    def get_sampler(self):
        return stats.norm(loc=0, scale=1./np.sqrt(2.))

    @unittest.skip("NotDef.")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.skip("NotDef.")
    def test_gauss_radau_quadrature(self):
        pass

    @unittest.skip("NotDef.")
    def test_mc_orthogonality(self):
        pass

    @unittest.skip("NotDef.")
    def test_monomials(self):
        pass

class TestHermiteProbabilistsFunction(TestPolynomial):

    def setUp(self):
        super(TestHermiteProbabilistsFunction,self).setUp()
        self.f = lambda x: np.exp(-x**2/2)

    def get_analytic_integral(self):
        return np.sqrt(2. * np.pi)

    def get_basis(self):
        return S1D.HermiteProbabilistsFunction()

    def get_sampler(self):
        return stats.norm(loc=0, scale=1./np.sqrt(2.))

    @unittest.skip("NotDef.")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.skip("NotDef.")
    def test_gauss_radau_quadrature(self):
        pass

    @unittest.skip("NotDef.")
    def test_mc_orthogonality(self):
        pass

    @unittest.skip("NotDef.")
    def test_monomials(self):
        pass

class TestLaguerreAlpha0Function(TestPolynomial):

    def setUp(self):
        super(TestLaguerreAlpha0Function,self).setUp()
        self.f = lambda x: np.sin(x/5.) * np.exp(-x)

    def get_analytic_integral(self):
        return 5./26.

    def get_basis(self):
        return S1D.LaguerreFunction(0.)

    def get_sampler(self):
        return stats.gamma(1.)

    @unittest.skip("NotDef.")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestLaguerreAlpha0Function,self).test_mc_orthogonality()

    @unittest.skip("NotDef.")
    def test_monomials(self):
        pass

class TestLaguerreAlpha1Function(TestPolynomial):

    def setUp(self):
        super(TestLaguerreAlpha1Function,self).setUp()
        self.f = lambda x: np.sin(x/5.) * np.exp(-x)

    def get_analytic_integral(self):
        return 125./338.

    def get_basis(self):
        return S1D.LaguerreFunction(1.)

    def get_sampler(self):
        return stats.gamma(2.)

    @unittest.skip("NotDef.")
    def test_gauss_lobatto_quadrature(self):
        pass

    @unittest.expectedFailure
    def test_mc_orthogonality(self):
        super(TestLaguerreAlpha0Function,self).test_mc_orthogonality()

    @unittest.skip("NotDef.")
    def test_monomials(self):
        pass

class TestConstantExtendedHermiteProbabilistsFunction(TestBasis):

    def setUp(self):
        super(TestConstantExtendedHermiteProbabilistsFunction,self).setUp()
        self.f = lambda x: np.exp(-x**2/2)

    def get_analytic_integral(self):
        return np.sqrt(2. * np.pi)

    def get_basis(self):
        return S1D.ConstantExtendedHermiteProbabilistsFunction()

    def get_sampler(self):
        return stats.norm(loc=0, scale=1./np.sqrt(2.))

class TestHermiteProbabilistsRadialBasisFunction(TestBasis):

    def setUp(self):
        super(TestHermiteProbabilistsRadialBasisFunction,self).setUp()
        self.min_ord = 41
        self.max_der = 3
        self.eps_interpolate = 1e-4
        self.f = lambda x: np.exp(-x**2/2)

    def get_analytic_integral(self):
        return np.sqrt(2. * np.pi)

    def get_basis(self):
        return S1D.HermiteProbabilistsRadialBasisFunction(42)

    def get_sampler(self):
        return stats.norm(loc=0, scale=1./np.sqrt(2.))

class TestConstantExtendedHermiteProbabilistsRadialBasisFunction(TestBasis):

    def setUp(self):
        super(TestConstantExtendedHermiteProbabilistsRadialBasisFunction,self).setUp()
        self.min_ord = 41
        self.max_der = 3
        self.eps_interpolate = 1e-4
        self.f = lambda x: np.exp(-x**2/2)

    def get_analytic_integral(self):
        return np.sqrt(2. * np.pi)

    def get_basis(self):
        return S1D.ConstantExtendedHermiteProbabilistsRadialBasisFunction(42)

    def get_sampler(self):
        return stats.norm(loc=0, scale=1./np.sqrt(2.))
        

class TestLinearExtendedHermiteProbabilistsRadialBasisFunction(TestBasis):

    def setUp(self):
        super(TestLinearExtendedHermiteProbabilistsRadialBasisFunction,self).setUp()
        self.min_ord = 41
        self.max_der = 3
        self.eps_interpolate = 1e-4
        self.f = lambda x: np.exp(-x**2/2)

    def get_analytic_integral(self):
        return np.sqrt(2. * np.pi)

    def get_basis(self):
        return S1D.LinearExtendedHermiteProbabilistsRadialBasisFunction(42)

    def get_sampler(self):
        return stats.norm(loc=0, scale=1./np.sqrt(2.))
        
def build_suite():
    suite_legendre = unittest.TestLoader().loadTestsFromTestCase( TestLegendrePolynomial )
    suite_chebyshev_first = unittest.TestLoader().loadTestsFromTestCase( TestChebyshevFirstKindPolynomial )
    suite_chebyshev_second = unittest.TestLoader().loadTestsFromTestCase( TestChebyshevSecondKindPolynomial )
    suite_jacobi = unittest.TestLoader().loadTestsFromTestCase( TestJacobiPolynomial )
    suite_laguerre0 = unittest.TestLoader().loadTestsFromTestCase( TestLaguerreAlpha0Polynomial )
    suite_laguerre1 = unittest.TestLoader().loadTestsFromTestCase( TestLaguerreAlpha1Polynomial )
    suite_hermite_phys = unittest.TestLoader().loadTestsFromTestCase( TestHermitePhysicistsPolynomial )
    suite_hermite_prob = unittest.TestLoader().loadTestsFromTestCase( TestHermiteProbabilistsPolynomial )
    suite_generic = unittest.TestLoader().loadTestsFromTestCase( TestGenericOrthogonalPolynomial )
    suite_fourier = unittest.TestLoader().loadTestsFromTestCase( TestFourierBasis )
    suite_hermite_phys_fun = unittest.TestLoader().loadTestsFromTestCase( TestHermitePhysicistsFunction )
    suite_hermite_prob_fun = unittest.TestLoader().loadTestsFromTestCase( TestHermiteProbabilistsFunction )
    suite_laguerre0_fun = unittest.TestLoader().loadTestsFromTestCase( TestLaguerreAlpha0Function )
    suite_laguerre1_fun = unittest.TestLoader().loadTestsFromTestCase( TestLaguerreAlpha1Function )
    suite_constext_hermite_prob_fun = unittest.TestLoader().loadTestsFromTestCase( TestConstantExtendedHermiteProbabilistsFunction )
    suite_hermite_prob_rbf = unittest.TestLoader().loadTestsFromTestCase( TestHermiteProbabilistsRadialBasisFunction )
    suite_constext_hermite_prob_rbf = unittest.TestLoader().loadTestsFromTestCase( TestConstantExtendedHermiteProbabilistsRadialBasisFunction )
    suite_linext_hermite_prob_rbf = unittest.TestLoader().loadTestsFromTestCase( TestLinearExtendedHermiteProbabilistsRadialBasisFunction )
    # GROUP SUITES
    suites_list = [suite_legendre, suite_chebyshev_first,
                   suite_chebyshev_second, suite_jacobi,
                   suite_laguerre0, suite_laguerre1,
                   suite_hermite_phys, suite_hermite_prob,
                   suite_generic, suite_fourier,
                   suite_hermite_phys_fun, suite_hermite_prob_fun,
                   suite_laguerre0_fun, suite_laguerre1_fun,
                   suite_constext_hermite_prob_fun,
                   suite_hermite_prob_rbf, suite_constext_hermite_prob_rbf,
                   suite_linext_hermite_prob_rbf]
    all_suites = unittest.TestSuite( suites_list )
    return all_suites

def run_tests():
    all_suites = build_suite()
    # RUN
    unittest.TextTestRunner(verbosity=2).run(all_suites)

if __name__ == '__main__':
    run_tests()
