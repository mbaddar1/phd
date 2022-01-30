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
import unittest
from SpectralToolbox.tests import test_spectral1d
from SpectralToolbox.tests import test_squared_orth

def run_all():
    suites_list = [ test_spectral1d.build_suite(),
                    test_squared_orth.build_suite()]
    all_suites = unittest.TestSuite( suites_list )
    # RUN
    tr = unittest.TextTestRunner(verbosity=2).run(all_suites)
    # Raise error if some tests failed or exited with error state
    nerr = len(tr.errors)
    nfail = len(tr.failures)
    if nerr + nfail > 0:
        print("Errors: %d, Failures: %d" % (nerr, nfail))
        sys.exit(1)

if __name__ == '__main__':
    run_all()