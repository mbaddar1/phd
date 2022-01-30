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

__revision__ = filter(str.isdigit, "$Revision: 84 $")

__author__ = "Daniele Bigoni"
__copyright__ = """Copyright 2012, Daniele Bigoni"""
__credits__ = ["Daniele Bigoni"]
__maintainer__ = "Daniele Bigoni"
__email__ = "dabi@imm.dtu.dk"
__status__ = "Production"

import numpy as np

GAUSSIAN = 'Gaussian'
BETA = 'Beta'
GAMMA = 'Gamma'
AVAIL_WEIGHTS = [GAUSSIAN, BETA, GAMMA]

class Stroud:
    """ Initialization of the Stroud integral points generator.

    This method generates an instance of the Stroud class, to be used in order to generate
    N-dimensional cubature rules with respect to the selected weight. Available weight types can be
    selected using their string name or by predefined attributes

        * 'Gaussian' or ``Stroud.GAUSSIAN``
        * 'Beta' or ``Stroud.BETA``
        * 'Gamma' or ``Stroud.GAMMA``
    
    Additional parameters are required for some weights.

    +--------------+--------------+
    | Weight       | Parameters   |
    +==============+==============+
    | Gaussian     | None         |
    +--------------+--------------+
    | Beta         | (alpha,beta) |
    +--------------+--------------+
    | Gamma        | alpha        |
    +--------------+--------------+

    See :cite:`Stroud1957` and :cite:`Xiu2008` for details.

    Args:
        weight (Stroud.AVAIL_WEIGHTS): type of the multidimensional weight for the formula.
        params (object): The parameters needed by the selected weight

    """
    weight = []
    params = []
    
    def __init__(self, weight, params):
        
        if weight in AVAIL_WEIGHTS:
            if (weight == BETA):
                if len(params) != 2:
                    print("The number of parameters inserted for the weight of type '%s' is not correct" % weight)
                    return
            if (weight == GAMMA):
                if len(params) != 1:
                    print("The number of parameters inserted for the weight of type '%s' is not correct" % weight)
                    return
        else:
            print("The selected type of weight is not included in the toolbox")
            return
        
        self.weight = weight;
        self.params = params;
    
    def stroud2(self,n):
        """
        stroud2(): Generates n+1 equally weighted points for the degree 2 Stroud formula.
        
        Syntax:
            ``x = stroud2(n)``
        
        Input:
            * ``n`` = (int) number of dimensions of the integration formula.
            
        Output:
            * ``x`` = (float, nd-array (n+1,n)) ``n+1`` quadrature nodes.
            * ``w0`` = (float) weight (equal for all the points)
        """
        
        # Compute the Gaussian nodes
        I0 = 1.
        if (n % 2 == 0):
            k = np.tile(np.arange(0.,n+1,1.),(n,1)).T
            r = np.tile( np.tile(np.arange(1.,np.floor(n/2)+1,1.),(2,1)).T.ravel(), (n+1,1))
            x = 2*r*k*np.pi/(n+1.)
            x[:,0:n+1:2] = np.sqrt(2) * np.cos(x[:,0:n+1:2])
            x[:,1:n+1:2] = np.sqrt(2) * np.sin(x[:,1:n+1:2])
        else:
            k = np.tile(np.arange(0.,n+1,1.),(n-1,1)).T
            r = np.tile( np.tile(np.arange(1.,np.floor(n/2)+1,1.),(2,1)).T.ravel(), (n+1,1))
            x = 2*r*k*np.pi/(n+1.)
            x[:,0:n+1:2] = np.sqrt(2) * np.cos(x[:,0:n+1:2])
            x[:,1:n+1:2] = np.sqrt(2) * np.sin(x[:,1:n+1:2])
            x = np.hstack( (x, 
                            np.reshape((-1)**np.arange(1.,n+2), (n+1,1)) ) )
        
        if (self.weight == BETA):
            I0 = 2. * n
            alpha = self.params[0]
            beta = self.params[1]
            x = 1./(alpha + beta + 2.) * (2.* np.sqrt( ((alpha + 1.)*(beta+1.))/(alpha+beta+3.) ) * x - (alpha - beta) )
        
        if (self.weight == GAMMA):
            I0 = 1.
            alpha = self.params[0]
            x = -np.sqrt(alpha + 1.) * x + (alpha + 1.)
        
        w0 = I0/(n+1)
        
        return (x,w0)
    
    def stroud3(self,n):
        """
        stroud2(): Generates 2n equally weighted points for the degree 3 Stroud formula.
        
        Syntax:
            ``x = stroud3(n)``
        
        Input:
            * ``n`` = (int) number of dimensions of the integration formula.
            
        Output:
            * ``x`` = (float, nd-array (n+1,n)) ``n+1`` quadrature nodes.
            * ``w0`` = (float) weight (equal for all the points)
        
        Description:
            The method is available only for distributions in symmetric regions, thus only for integrals with Gaussian weights and symmetric Beta weights (alpha=beta) in R^n
        """
        
        if ((self.weight == GAMMA) or ((self.weight == BETA) and (self.params[0] != self.params[1]))):
            print("There is no degree 3 Stroud formula for the selected type of weight")
            return
        
        # Compute the gaussian nodes
        I0 = 1.
        if (n % 2 == 0):
            k = np.tile(np.arange(0.,2*n,1.),(n,1)).T
            r = np.tile( np.tile(np.arange(1.,np.floor(n/2)+1,1.),(2,1)).T.ravel(), (2*n,1))
            x = (2.*r-1.)*k*np.pi/n
            x[:,0:n+1:2] = np.sqrt(2) * np.cos(x[:,0:n+1:2])
            x[:,1:n+1:2] = np.sqrt(2) * np.sin(x[:,1:n+1:2])
        else:
            k = np.tile(np.arange(0.,2*n,1.),(n-1,1)).T
            r = np.tile( np.tile(np.arange(1.,np.floor(n/2)+1,1.),(2,1)).T.ravel(), (2*n,1))
            x = (2.*r-1.)*k*np.pi/n
            x[:,0:n+1:2] = np.sqrt(2) * np.cos(x[:,0:n+1:2])
            x[:,1:n+1:2] = np.sqrt(2) * np.sin(x[:,1:n+1:2])
            x = np.hstack( (x, 
                            np.reshape((-1)**np.arange(1.,2*n+1), (2*n,1)) ) )
        
        if (self.weight == BETA):
            I0 = 2. * n
            alpha = self.params[0]
            x = 1. / np.sqrt( 2. * alpha + 3. ) * x
        
        w0 = I0/(2*n)
        
        return (x,w0)
