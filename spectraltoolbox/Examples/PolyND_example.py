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
from mpl_toolkits.mplot3d import Axes3D
from SpectralToolbox import Spectral1D as S1D
from SpectralToolbox import SpectralND as SND

N = 3

# Legendre x Hermite
x = [np.linspace(-1,1,20), np.linspace(-3,3,20)]
(XX, YY) = np.meshgrid(*x)
alpha = 0
beta = 0
polys = [ S1D.Poly1D(S1D.JACOBI,[alpha,beta]), S1D.Poly1D(S1D.HERMITEP_PROB,None) ]
P = SND.PolyND(polys)
V = P.GradVandermondePascalSimplex( x, N, [0]*2 )
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface( XX, YY, V[:,7].reshape((20,20)), rstride=1, cstride=1, cmap=plt.cm.coolwarm,
                  linewidth=0, antialiased=False)
plt.show(False)
