import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

from SpectralToolbox import Spectral1D as S1D
from SpectralToolbox import SpectralND as SND
from SpectralToolbox import Stroud as STR

BW = True
TITLE_ONOFF = False
FORMATS = ['pdf','png','eps']

d = 3
# Full tensor product
P = SND.PolyND( [S1D.Poly1D( S1D.JACOBI,[0.,0.] )] * d )
(x2,w2) = P.GaussLobattoQuadrature( [2]*d ) # Order 2*2-1 = 2
(x3,w3) = P.Quadrature( [1]*d ) # Order 2*1+1 = 3

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x2[:,0],x2[:,1],x2[:,2], c='k',marker='o')
plt.tight_layout()
plt.show(block=False)
for f in FORMATS:
    plt.savefig("figs/FullTensor-Ord2.%s" % f,format=f)

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x3[:,0],x3[:,1],x3[:,2], c='k',marker='^')
plt.tight_layout()
plt.show(block=False)
for f in FORMATS:
    plt.savefig("figs/FullTensor-Ord3.%s" % f,format=f)


# Stroud rule
ST = STR.Stroud( STR.BETA, [0,0] )
(x2,w2) = ST.stroud2(3)
(x3,w3) = ST.stroud3(3)
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x2[:,0],x2[:,1],x2[:,2], c='k',marker='o')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.tight_layout()
plt.show(block=False)
for f in FORMATS:
    plt.savefig("figs/Stroud-Ord2.%s" % f,format=f)

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x3[:,0],x3[:,1],x3[:,2], c='k',marker='^')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
plt.tight_layout()
plt.show(block=False)
for f in FORMATS:
    plt.savefig("figs/Stroud-Ord3.%s" % f,format=f)
