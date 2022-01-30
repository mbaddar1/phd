import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import SpectralToolbox.Spectral1D as S1D

N = 4
x = np.linspace(-6,6,100)
P = S1D.PositiveDefiniteSquaredConstantExtendedHermiteProbabilistsFunction()
V = P.GradVandermonde(x, N, k=-1, norm=True)

plt.figure()
for i in range(N+1):
    for j in range(N+1):
        ax = plt.subplot(N+1,N+1,i*(N+1)+j+1)
        ax.plot(x, V[:,i,j])
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_xticklines(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_yticklines(), visible=False)
plt.show(False)

# Draw random coefficients
a = npr.randn(N+1)
fval = np.dot( np.tensordot( V, a, axes=(1,0)), a )
plt.figure()
plt.plot(x, fval)
plt.show(False)