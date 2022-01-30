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

from SpectralToolbox._version import __version__

import SpectralToolbox.Spectral1D.Constants
from SpectralToolbox.Spectral1D.Constants import *
import SpectralToolbox.Spectral1D.AbstractClasses
from SpectralToolbox.Spectral1D.AbstractClasses import *
import SpectralToolbox.Spectral1D.OrthogonalPolynomials
from SpectralToolbox.Spectral1D.OrthogonalPolynomials import *
import SpectralToolbox.Spectral1D.OrthogonalFunctions
from SpectralToolbox.Spectral1D.OrthogonalFunctions import *
import SpectralToolbox.Spectral1D.LagrangePolynomials
from SpectralToolbox.Spectral1D.LagrangePolynomials import *
import SpectralToolbox.Spectral1D.LinearInterpolation
from SpectralToolbox.Spectral1D.LinearInterpolation import *
import SpectralToolbox.Spectral1D.NonOrthogonalBasis
from SpectralToolbox.Spectral1D.NonOrthogonalBasis import *
import SpectralToolbox.Spectral1D.AlgebraicPolynomials
from SpectralToolbox.Spectral1D.AlgebraicPolynomials import *
import SpectralToolbox.Spectral1D.OldInterface
from SpectralToolbox.Spectral1D.OldInterface import *
import SpectralToolbox.Spectral1D.auxiliary
from SpectralToolbox.Spectral1D.auxiliary import *


__all__ = []
__all__ += Constants.__all__
__all__ += AbstractClasses.__all__
__all__ += OrthogonalPolynomials.__all__
__all__ += OrthogonalFunctions.__all__
__all__ += LagrangePolynomials.__all__
__all__ += LinearInterpolation.__all__
__all__ += NonOrthogonalBasis.__all__
__all__ += AlgebraicPolynomials.__all__
__all__ += OldInterface.__all__
__all__ += auxiliary.__all__

XML_NAMESPACE = '{SpectralToolbox}'

__author__ = "Daniele Bigoni"
__copyright__ = """LGPLv3, Copyright (C) 2012-2015, The Technical University of Denmark"""
__credits__ = ["Daniele Bigoni"]
__maintainer__ = "Daniele Bigoni"
__email__ = "dabi@imm.dtu.dk"
__status__ = "Production"
