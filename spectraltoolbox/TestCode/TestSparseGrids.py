# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:08:24 2012

@author: dabi

Sparse grid tester

"""

from numpy import *

from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
from time import clock

import SparseGrids
import Spectral1D

pyplot.close('all')

DIM = 3
k = 3

QUADRULE = SparseGrids.GQN

sg = SparseGrids.SparseGrid(QUADRULE,DIM,k,True)

startTime = clock()
(n,w) = sg.sparseGrid()
stopTime = clock()
print "Elapsed Time: %f s" % (stopTime-startTime)

if DIM == 2:
    # Plot on the vertical direction the weight
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(n[:,0], n[:,1], w)
    pyplot.show()
    
    # Standard plot 2D
    pylab.figure()
    pyplot.scatter( n[:,0], n[:,1], c=amax(abs(w))/abs(w))
    pyplot.gray()

if DIM == 3:
    # Plot in 3D
    fig = pylab.figure()
    ax = Axes3D(fig)
    
    ax.scatter(n[:,0], n[:,1], n[:,2], c=amax(abs(w))/abs(w))
    pyplot.gray()
    pyplot.show()
    
    # Make cross dimension subplots
    pylab.figure()
    for i in range(0,DIM):
        for j in range(0,i+1):
            subplot(DIM,DIM,(i*DIM)+j+1)
            pyplot.scatter(n[:,i],n[:,j],c=amax(abs(w))/abs(w))
            pyplot.gray()
    
if DIM > 3 and DIM < 10:
    # Make cross dimension subplots
    pylab.figure()
    for i in range(0,DIM):
        for j in range(0,i+1):
            subplot(DIM,DIM,(i*DIM)+j+1)
            pyplot.scatter(n[:,i],n[:,j],c=amax(abs(w))/abs(w))
            pyplot.gray()

savetxt('nodes.txt', n, fmt="%1.7e")
savetxt('weights.txt', w, fmt="%1.7e")