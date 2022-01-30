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

__all__ = ['FOURIER', 'JACOBI', 'HERMITEP','HERMITEF',
           'HERMITEP_PROB', 'HERMITEF_PROB',
           'LAGUERREP', 'LAGUERREF', 'ORTHPOL', 'AVAIL_POLY',
           'GAUSS', 'GAUSSRADAU', 'GAUSSLOBATTO', 'GQU',
           'GQN', 'KPU', 'KPN', 'CC',
           'FEJ', 'NESTEDLOBATTO', 'NESTEDGAUSS', 'AVAIL_QUADPOINTS']

FOURIER = 'Fourier'
JACOBI = 'Jacobi'
HERMITEP = 'HermiteP'
HERMITEF = 'HermiteF'
HERMITEP_PROB = 'HermitePprob'
HERMITEF_PROB = 'HermiteFprob'
LAGUERREP = 'LaguerreP'
LAGUERREF = 'LaguerreF'
ORTHPOL = 'ORTHPOL'
AVAIL_POLY = [FOURIER, JACOBI, HERMITEP, HERMITEF, HERMITEP_PROB, HERMITEF_PROB, LAGUERREP, LAGUERREF, ORTHPOL]

GAUSS = 'Gauss'
GAUSSRADAU = 'GaussRadau'
GAUSSLOBATTO = 'GaussLobatto'
GQU = 'Gauss Uniform'
GQN = 'Gauss Normal'
KPU = 'Kronrod-Patterson Uniform'
KPN = 'Kronrod-Patterson Normal'
CC  = 'Clenshaw-Curtis Uniform'
FEJ = 'Fejer Uniform'
NESTEDLOBATTO = 'Nested Lobatto'
NESTEDGAUSS = 'Nested Gauss'
AVAIL_QUADPOINTS = [GAUSS, GAUSSRADAU, GAUSSLOBATTO, GQU, GQN, KPU, KPN, CC, FEJ, NESTEDGAUSS, NESTEDLOBATTO]
