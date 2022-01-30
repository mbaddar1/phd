#
# This file is part of SpectralToolbox.
#
# TensorToolbox is free software: you can redistribute it and/or modify
# it under the terms of the LGNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TensorToolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LGNU Lesser General Public License for more details.
#
# You should have received a copy of the LGNU Lesser General Public License
# along with TensorToolbox.  If not, see <http://www.gnu.org/licenses/>.
#
# DTU UQ Library
# Copyright (C) 2012-2015 The Technical University of Denmark
# Scientific Computing Section
# Department of Applied Mathematics and Computer Science
#
# Author: Daniele Bigoni
#

import numpy as np
import matplotlib.pyplot as plt
import SpectralToolbox.Spectral1D as S1D

nlev = 6

# Nested Gauss
plt.figure(figsize=(6,5))
for l in range(1,nlev+1):
    (x,w) = S1D.nestedgauss(l,norm=False)
    plt.plot( x, l * np.ones(len(x)), 'ok' )
plt.xlim([-1.1,1.1])
plt.ylim([-1,nlev+1])
plt.show(False)

# Nested Gauss
plt.figure(figsize=(6,5))
for l in range(1,nlev+1):
    (x,w) = S1D.nestedlobatto(l,norm=False)
    plt.plot( x, l * np.ones(len(x)), 'ok' )
plt.xlim([-1.1,1.1])
plt.ylim([-1,nlev+1])
plt.show(False)
