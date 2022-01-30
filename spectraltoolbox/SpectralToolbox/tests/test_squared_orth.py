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
import numpy as np
import scipy.stats as stats

import SpectralToolbox.Spectral1D as S1D

class TestSquared(unittest.TestCase):
    def setUp(self):
        self.rtol = 1e-10
        self.atol = 1e-12
        self.max_ord = 10
        qnum = 40
        JAC = S1D.JacobiPolynomial(0, 0)
        xq, wq = JAC.Quadrature(qnum)
        self.xq = (xq+1.)/2.
        self.wq = wq/2.

    def get_basis(self):
        raise NotImplementedError("!")
    def get_sampler(self):
        raise NotImplementedError("!")

    def test_squared(self):
        P = self.get_basis()
        SP = self.get_squared_basis()
        x = self.get_sampler().rvs(1)

        for n in range(self.max_ord+1):
            for m in range(n, self.max_ord+1):
                # Approximate \int \phi_n \phi_m dt with quadrature
                p1 = P.GradEvaluate(x, n, k=0, norm=self.norm)[0]
                p2 = P.GradEvaluate(x, m, k=0, norm=self.norm)[0]
                app_val = p1 * p2
                # Exact value
                exa_val = SP.GradEvaluate(x, n, m, k=0, norm=self.norm)[0]
                # Check
                self.assertTrue( np.allclose(app_val, exa_val, rtol=self.rtol, atol=self.atol) )

    def test_integrated(self):
        SP = self.get_squared_basis()
        x = self.get_sampler().rvs(1)
        xq = self.xq * x[0]
        wq = self.wq * x[0]

        for n in range(self.max_ord+1):
            for m in range(n, self.max_ord+1):
                # Approximate \int \phi_n \phi_m dt with quadrature
                p1p2 = SP.GradEvaluate(xq, n, m, k=0, norm=self.norm)
                app_val = np.dot( p1p2, wq )
                # Exact value
                exa_val = SP.GradEvaluate(x, n, m, k=-1, norm=self.norm)[0]
                # Check
                self.assertTrue( np.allclose(app_val, exa_val, rtol=self.rtol, atol=self.atol) )

    def test_vandermonde(self):
        SP = self.get_squared_basis()
        x = self.get_sampler().rvs(20)

        # Generate Vandermonde
        vand = SP.GradVandermonde(x, self.max_ord, k=-1, norm=self.norm)
        # Check correctness
        for i in range(self.max_ord+1):
            for j in range(self.max_ord+1):
                val = SP.GradEvaluate(x, i, j, k=-1, norm=self.norm)
                self.assertTrue( np.allclose( val, vand[:,i,j], rtol=self.rtol, atol=self.atol ))

class TestNormed(object):
    norm = True

class TestNotNormed(object):
    norm = False
                
class TestHermiteProbabilistsPolynomial(TestSquared):
    def setUp(self):
        super(TestHermiteProbabilistsPolynomial,self).setUp()
    def get_basis(self):
        return S1D.HermiteProbabilistsPolynomial()
    def get_squared_basis(self):
        P = self.get_basis()
        return S1D.SquaredOrthogonalPolynomial( P )
    def get_sampler(self):
        return stats.norm()

class TestConstantExtendedHermitePhysicistsFunction(TestSquared):
    def setUp(self):
        super(TestConstantExtendedHermitePhysicistsFunction,self).setUp()
    def get_basis(self):
        return S1D.ConstantExtendedHermitePhysicistsFunction()
    def get_squared_basis(self):
        return S1D.SquaredConstantExtendedHermitePhysicistsFunction()
    def get_sampler(self):
        return stats.norm()

class TestPositiveDefiniteConstantExtendedHermitePhysicistsFunction(TestSquared):
    def setUp(self):
        super(TestPositiveDefiniteConstantExtendedHermitePhysicistsFunction,
              self).setUp()
    def get_basis(self):
        return S1D.ConstantExtendedHermitePhysicistsFunction()
    def get_squared_basis(self):
        return S1D.PositiveDefiniteSquaredConstantExtendedHermitePhysicistsFunction()
    def get_sampler(self):
        return stats.norm()

class TestConstantExtendedHermiteProbabilistsFunction(TestSquared):
    def setUp(self):
        super(TestConstantExtendedHermiteProbabilistsFunction,self).setUp()
    def get_basis(self):
        return S1D.ConstantExtendedHermiteProbabilistsFunction()
    def get_squared_basis(self):
        return S1D.SquaredConstantExtendedHermiteProbabilistsFunction()
    def get_sampler(self):
        return stats.norm()

class TestPositiveDefiniteConstantExtendedHermiteProbabilistsFunction(TestSquared):
    def setUp(self):
        super(TestPositiveDefiniteConstantExtendedHermiteProbabilistsFunction,
              self).setUp()
    def get_basis(self):
        return S1D.ConstantExtendedHermiteProbabilistsFunction()
    def get_squared_basis(self):
        return S1D.PositiveDefiniteSquaredConstantExtendedHermiteProbabilistsFunction()
    def get_sampler(self):
        return stats.norm()

        
# class TestJacobiPolynomial(TestSquared):
#     r""" Test for :math:`\text{Beta}(2,5)`
#     Note that :math:`\rho_\beta(x,2,5) = 2 w(2x-1,4,1)`
#     """
#     def setUp(self):
#         super(TestJacobiPolynomial,self).setUp()
#     def get_basis(self):
#         return S1D.JacobiPolynomial(4.,1.)
#     def get_sampler(self):
#         return stats.beta(2,5,loc=-1,scale=2)

# class TestLaguerreAlpha1Polynomial(TestSquared):
#     def setUp(self):
#         super(TestLaguerreAlpha1Polynomial,self).setUp()
#     def get_basis(self):
#         return S1D.LaguerrePolynomial(1.)
#     def get_sampler(self):
#         return stats.gamma(2.)

# ALL TESTS
class TestNormedHermiteProbabilistsPolynomial(TestHermiteProbabilistsPolynomial,
                                              TestNormed): pass
class TestNotNormedHermiteProbabilistsPolynomial(TestHermiteProbabilistsPolynomial,
                                                 TestNotNormed): pass
class TestNormedConstantExtendedHermitePhysicistsFunction(
        TestConstantExtendedHermitePhysicistsFunction, TestNormed): pass
class TestNotNormedConstantExtendedHermitePhysicistsFunction(
        TestConstantExtendedHermitePhysicistsFunction, TestNotNormed): pass
class TestNormedPositiveDefiniteConstantExtendedHermitePhysicistsFunction(
        TestPositiveDefiniteConstantExtendedHermitePhysicistsFunction,
        TestNormed): pass
class TestNotNormedPositiveDefiniteConstantExtendedHermitePhysicistsFunction(
        TestPositiveDefiniteConstantExtendedHermitePhysicistsFunction,
        TestNotNormed): pass
class TestNormedConstantExtendedHermiteProbabilistsFunction(
        TestConstantExtendedHermiteProbabilistsFunction, TestNormed): pass
class TestNotNormedConstantExtendedHermiteProbabilistsFunction(
        TestConstantExtendedHermiteProbabilistsFunction, TestNotNormed): pass
class TestNormedPositiveDefiniteConstantExtendedHermiteProbabilistsFunction(
        TestPositiveDefiniteConstantExtendedHermiteProbabilistsFunction,
        TestNormed): pass
class TestNotNormedPositiveDefiniteConstantExtendedHermiteProbabilistsFunction(
        TestPositiveDefiniteConstantExtendedHermiteProbabilistsFunction,
        TestNotNormed): pass

# class TestNormedJacobiPolynomial(TestJacobiPolynomial, TestNormed): pass
# class TestNotNormedJacobiPolynomial(TestJacobiPolynomial, TestNotNormed): pass
# class TestNormedLaguerreAlpha1Polynomial(TestLaguerreAlpha1Polynomial, TestNormed): pass
# class TestNotNormedLaguerreAlpha1Polynomial(TestLaguerreAlpha1Polynomial, TestNotNormed): pass


def build_suite():
    suite_nhermite_prob = unittest.TestLoader().loadTestsFromTestCase(
        TestNormedHermiteProbabilistsPolynomial )
    suite_nnhermite_prob = unittest.TestLoader().loadTestsFromTestCase(
        TestNotNormedHermiteProbabilistsPolynomial )
    suite_n_const_ext_h_phy_fun = unittest.TestLoader().loadTestsFromTestCase(
        TestNormedConstantExtendedHermitePhysicistsFunction)
    suite_nn_const_ext_h_phy_fun = unittest.TestLoader().loadTestsFromTestCase(
        TestNotNormedConstantExtendedHermitePhysicistsFunction)
    suite_n_posdef_const_ext_h_phy_fun = unittest.TestLoader().loadTestsFromTestCase(
        TestNormedPositiveDefiniteConstantExtendedHermitePhysicistsFunction)
    suite_nn_posdef_const_ext_h_phy_fun = unittest.TestLoader().loadTestsFromTestCase(
        TestNotNormedPositiveDefiniteConstantExtendedHermitePhysicistsFunction)
    suite_n_const_ext_h_pro_fun = unittest.TestLoader().loadTestsFromTestCase(
        TestNormedConstantExtendedHermiteProbabilistsFunction)
    suite_nn_const_ext_h_pro_fun = unittest.TestLoader().loadTestsFromTestCase(
        TestNotNormedConstantExtendedHermiteProbabilistsFunction)
    suite_n_posdef_const_ext_h_pro_fun = unittest.TestLoader().loadTestsFromTestCase(
        TestNormedPositiveDefiniteConstantExtendedHermiteProbabilistsFunction)
    suite_nn_posdef_const_ext_h_pro_fun = unittest.TestLoader().loadTestsFromTestCase(
        TestNotNormedPositiveDefiniteConstantExtendedHermiteProbabilistsFunction)

    # suite_njacobi = unittest.TestLoader().loadTestsFromTestCase(
    #     TestNormedJacobiPolynomial )
    # suite_nnjacobi = unittest.TestLoader().loadTestsFromTestCase(
    #     TestNotNormedJacobiPolynomial )
    # suite_nlaguerre = unittest.TestLoader().loadTestsFromTestCase(
    #     TestNormedLaguerreAlpha1Polynomial )
    # suite_nnlaguerre = unittest.TestLoader().loadTestsFromTestCase(
    #     TestNotNormedLaguerreAlpha1Polynomial )
    
    # Group suites
    suites_list = [
        suite_nhermite_prob,
        suite_nnhermite_prob,
        suite_n_const_ext_h_phy_fun,
        suite_nn_const_ext_h_phy_fun,
        # suite_n_posdef_const_ext_h_phy_fun,
        # suite_nn_posdef_const_ext_h_phy_fun,
        suite_n_const_ext_h_pro_fun,
        suite_nn_const_ext_h_pro_fun,
        # suite_n_posdef_const_ext_h_pro_fun,
        # suite_nn_posdef_const_ext_h_pro_fun
    ]
    all_suites = unittest.TestSuite( suites_list )
    return all_suites

def run_tests():
    all_suites = build_suite()
    # RUN
    unittest.TextTestRunner(verbosity=2).run(all_suites)

if __name__ == '__main__':
    run_tests()
