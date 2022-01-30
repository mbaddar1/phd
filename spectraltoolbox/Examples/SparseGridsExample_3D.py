import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

from SpectralToolbox import Spectral1D as S1D
from SpectralToolbox import SpectralND as SND
from SpectralToolbox import HeterogeneousSparseGrids as HSG
from SpectralToolbox import SparseGrids as SG

BW = True
TITLE_ONOFF = False
FORMATS = ['pdf','png','eps']

# POLYS = [S1D.Poly1D(S1D.JACOBI,[0,0]), S1D.Poly1D(S1D.HERMITEP,None)]
# POLYS = [S1D.Poly1D(S1D.JACOBI,[0.5,0.5]), S1D.Poly1D(S1D.JACOBI,[0.5,0.5])]
# PND = SND.PolyND(POLYS)
# tensNs = [5,5]
# sgNs = [4,4]
#sg = HSG.HSparseGrid(POLYS,sgNs)

#############################################
# Krornd-Patterson
Ls = range(1,6)
plt.figure(figsize=(5,4))
for l in Ls:
    (x,w) = SG.KPU(l)
    x = np.asarray(x)
    x = np.hstack([1-x[1:][::-1],x])
    plt.plot(x,l*np.ones(len(x)),'o')

plt.ylim([0,6])
plt.xlim([0.,1.])
plt.xlabel('x')
plt.ylabel('level')
plt.title('Kronrod-Patterson')
plt.show(block=False)

plt.savefig("figs/Kronrod-Patterson-1Drule.pdf",format='pdf')
plt.savefig("figs/Kronrod-Patterson-1Drule.png",format='png')

# sg = SG.SparseGrid(SG.KPU,2,5,sym=1)

# (sgX,sgW) = sg.sparseGrid()

# plt.figure(figsize=(5,4))
# plt.plot(sgX[:,0], sgX[:,1], '.')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Kronrod-Patterson')

# plt.savefig("figs/Kronrod-Patterson-SG.pdf",format='pdf')
# plt.savefig("figs/Kronrod-Patterson-SG.png",format='png')

# sg = SG.SparseGrid(SG.KPU,1,5,sym=1)
# (x,w) = sg.sparseGrid()
# xs = [x.flatten(),x.flatten()]
# x = np.asarray(list(itertools.product(*xs)))

# plt.figure(figsize=(5,4))
# plt.plot(x[:,0], x[:,1], '.')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Krornd-Patterson')

# plt.savefig("figs/Kronrod-Patterson-Tensor.pdf",format='pdf')
# plt.savefig("figs/Kronrod-Patterson-Tensor.png",format='png')

# # scaling with d
# LEVEL = 5
# sg = SG.SparseGrid(SG.KPU,1,LEVEL,sym=1)
# (x,w) = sg.sparseGrid()
# N1D = x.shape[0]
# ds = range(1,21)
# nx = []
# for i,d in enumerate(ds):
#     sg = SG.SparseGrid(SG.KPU,d,LEVEL,sym=1)
#     (x,w) = sg.sparseGrid()
#     nx.append(x.shape[0])

# plt.figure(figsize=(5,4))
# plt.semilogy(ds,N1D**np.asarray(ds),'o-',label='Tensor')
# plt.semilogy(ds,nx,'o-',label='SG')
# plt.xlabel('d')
# plt.ylabel('n')
# plt.grid()
# plt.legend(loc='upper left')
# plt.title('Krornd-Patterson')

# plt.show(block=False)

# plt.savefig("figs/Kronrod-Patterson-scaling.pdf",format='pdf')
# plt.savefig("figs/Kronrod-Patterson-scaling.png",format='png')

########################################
# Clenshaw-Curtis
Ls = range(1,6)
plt.figure(figsize=(5,4))
for l in Ls:
    (x,w) = SG.CC(l)
    x = np.hstack([-x[1:][::-1],x])
    plt.plot(x,l*np.ones(len(x)),'o')

plt.ylim([0,6])
plt.xlim([-1.,1.])
plt.xlabel('x')
plt.ylabel('level')
plt.title('Clenshaw-Curtis')
plt.show(block=False)

plt.savefig("figs/Clenshaw-Curtis-1Drule.pdf",format='pdf')
plt.savefig("figs/Clenshaw-Curtis-1Drule.png",format='png')

# # Tensor grid
# (x,w) = SG.CC(5)
# x = np.hstack([-x[1:][::-1],x])
# xs = [x.flatten(),x.flatten()]
# x = np.asarray(list(itertools.product(*xs)))

# plt.figure(figsize=(5,4))
# plt.plot(x[:,0], x[:,1], '.')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Tensor Grid Clenshaw-Curtis')

# plt.savefig("figs/Clenshaw-Curtis-Tensor.pdf",format='pdf')
# plt.savefig("figs/Clenshaw-Curtis-Tensor.png",format='png')

# # Sparse grid
# sg = SG.SparseGrid(SG.CC,2,5,sym=1)
# (sgX,sgW) = sg.sparseGrid()

# plt.figure(figsize=(5,4))
# plt.plot(sgX[:,0], sgX[:,1], '.')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Sparse Grid Clenshaw-Curtis')

# plt.show(block=False)

# plt.savefig("figs/Clenshaw-Curtis-SG.pdf",format='pdf')
# plt.savefig("figs/Clenshaw-Curtis-SG.png",format='png')

# # scaling with d
# LEVEL = 5
# sg = SG.SparseGrid(SG.CC,1,LEVEL,sym=1)
# (x,w) = sg.sparseGrid()
# N1D = x.shape[0]
# ds = range(1,6)
# nx = []
# for i,d in enumerate(ds):
#     sg = SG.SparseGrid(SG.CC,d,LEVEL,sym=1)
#     (x,w) = sg.sparseGrid()
#     nx.append(x.shape[0])

# plt.figure(figsize=(5,4))
# plt.semilogy(ds,N1D**np.asarray(ds),'o-',label='Tensor')
# plt.semilogy(ds,nx,'o-',label='SG')
# plt.xlabel('d')
# plt.ylabel('n')
# plt.grid()
# plt.legend(loc='upper left')
# plt.title('Clenshaw-Curtis')

# plt.show(block=False)

# plt.savefig("figs/Clenshaw-Curtis-scaling.pdf",format='pdf')
# plt.savefig("figs/Clenshaw-Curtis-scaling.png",format='png')

#########################################
# Fejer's
Ls = range(1,6)
plt.figure(figsize=(5,4))
for l in Ls:
    (x,w) = SG.FEJ(l)
    x = np.hstack([-x[1:][::-1],x])
    plt.plot(x,l*np.ones(len(x)),'o')

plt.ylim([0,6])
plt.xlim([-1.,1.])
plt.xlabel('x')
plt.ylabel('level')
plt.title('Fejer\'s')
plt.tight_layout()
plt.show(block=False)

for f in FORMATS:
    plt.savefig("figs/Fejer-1Drule.%s" % f,format=f)

# Tensor grid
(x,w) = SG.FEJ(5)
x = np.hstack([-x[1:][::-1],x])
xs = [x.flatten(),x.flatten(),x.flatten()]
x = np.asarray(list(itertools.product(*xs)))

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], x[:,2], '.k')
if TITLE_ONOFF:
    plt.title('Tensor Grid Fejer\'s')
plt.tight_layout()
plt.show(block=False)
for f in FORMATS:
    plt.savefig("figs/Fejer-Tensor.%s" % f,format=f)

# Sparse grid
sg = SG.SparseGrid(SG.FEJ,3,5,sym=1)
(sgX,sgW) = sg.sparseGrid()

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sgX[:,0], sgX[:,1], sgX[:,2], '.k')
if TITLE_ONOFF:
    plt.title('Sparse Grid Fejer\'s')
plt.tight_layout()
plt.show(block=False)
for f in FORMATS:
    plt.savefig("figs/Fejer-SG.%s" % f,format=f)

# scaling with d
LEVEL = 5
sg = SG.SparseGrid(SG.FEJ,1,LEVEL,sym=1)
(x,w) = sg.sparseGrid()
N1D = x.shape[0]
ds = range(1,21)
nx = []
for i,d in enumerate(ds):
    sg = SG.SparseGrid(SG.FEJ,d,LEVEL,sym=1)
    (x,w) = sg.sparseGrid()
    nx.append(x.shape[0])

plt.figure(figsize=(5,4))
plt.semilogy(ds,N1D**np.asarray(ds),'o-k',label='Tensor')
plt.semilogy(ds,nx,'v-k',label='SG')
plt.xlabel('d')
plt.ylabel('n')
plt.grid()
plt.legend(loc='upper left')
if TITLE_ONOFF:
    plt.title('Fejer\'s')
plt.tight_layout()

plt.show(block=False)

for f in FORMATS:
    plt.savefig("figs/Fejer-scaling.%s" % f,format=f)

plt.show(block=False)
