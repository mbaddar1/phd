# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:43:39 2012

@author: dabi
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SpectralToolbox import Stroud as ST

plt.close('all')

n = 2

st = ST.Stroud(ST.BETA,[0.0,0.0]);

(x,w0) = st.stroud2(n)
(x3,w03) = st.stroud3(n)

#def f(x,y): return x**2. + y**2.
def f(x,y): return x**3. + y**3.
#def f(x,y): return x * y;
#def f(x,y): return (x+y)*0. + 1.

xI = np.linspace(-1.,1.,30)

(XI,YI) = np.meshgrid(xI,xI)

ZI = f(XI,YI)


y2 = f(x[:,0],x[:,1])
I2 = np.sum(y2) * w0

y3 = f(x3[:,0],x3[:,1])
I3 = np.sum(y3) * w03

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(XI,YI,ZI)
ax.scatter(x[:,0],x[:,1],y2,c='r',s=25)
ax.scatter(x3[:,0],x3[:,1],y3,c='k',s=25)

print "Integral 2nd: %f" % I2
print "Integral 3rd: %f" % I3

# 3D functions
n = 3

def f3(x,y,z): return x**3.+y**3.+z**3.;

(x,w0) = st.stroud2(n)
(x3,w03) = st.stroud3(n)

y2 = f3(x[:,0],x[:,1],x[:,2])
I2 = np.sum(y2) * w0

y3 = f3(x3[:,0],x3[:,1],x3[:,2])
I3 = np.sum(y3) * w03

print "Integral 2nd: %f" % I2
print "Integral 3rd: %f" % I3
