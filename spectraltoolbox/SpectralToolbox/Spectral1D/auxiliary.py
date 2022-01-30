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

from SpectralToolbox.Spectral1D import Basis
from SpectralToolbox.Spectral1D.Constants import *
from SpectralToolbox.Spectral1D.OrthogonalPolynomials import *
from SpectralToolbox.Spectral1D.OrthogonalFunctions import *

__all__ = ['generate', 'from_xml_element']

def generate(ptype, params):
    r""" Generate orthogonal basis objects from ``Spectral1D.AVAIL_POLY``.

    Args:
      ptype (string): one of the available polynomial types as listed
        in ``Spectral1D.AVAIL_POLY``
      params (list): list of parameters need.

    Returns:
      (:class:`OrthogonalBasis`) -- the orthogonal basis required
    """    
    if ptype == JACOBI:
        return JacobiPolynomial(*params)
    if ptype == HERMITEP:
        return HermitePhysicistsPolynomial()
    if ptype == HERMITEF:
        return HermitePhysicistsFunction()
    if ptype == HERMITEP_PROB:
        return HermiteProbabilistsPolynomial()
    if ptype == HERMITEF_PROB:
        return HermiteProbabilistsFunction()
    if ptype == LAGUERREP:
        return LaguerrePolynomial()
    if ptype == LAGUERREF:
        return LaguerreFunction()
    if ptype == ORTHPOL:
        return GenericOrthogonalPolynomial(*params)
    if ptype == FOURIER:
        return Fourier()

def from_xml_element(node):
    return Basis.from_xml_element(node)