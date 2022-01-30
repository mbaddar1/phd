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
import scipy.stats as stats
import matplotlib.pyplot as plt
from SpectralToolbox import Spectral1D as S1D
Ns = range(20)
# Uniform distribution
alpha = 0
beta = 0
P = S1D.Poly1D(S1D.JACOBI,[alpha,beta])
plt.figure(figsize=(6,5))
for i in Ns:
    (x,w) = P.Quadrature(i,quadType=S1D.GAUSS)
    x = (x+1.)/2.
    w /= np.sum(w)
    plt.plot(x,i*np.ones(x.shape),'ko')
plt.xlabel('x')
plt.ylim([-0.5,Ns[-1]+0.5])
plt.show(False)
