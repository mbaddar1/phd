
#%%

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.image import NonUniformImage
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import null_space
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize
from scipy.optimize.nonlin import nonlin_solve
from scipy.sparse.construct import rand
from sklearn.neighbors import KernelDensity

from colorama import Fore, Style

from tt import TensorTrain
from tictoc import TicToc
import time

from scipy import stats
import math

import torch
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss

from scipy.optimize import minimize, basinhopping

global calls

#sys.path.append("/home/rob/libs/alea/src")
from alea.math_utils.multiindex_set import MultiindexSet
from alea.math_utils.polynomials.polynomials import (LegendrePolynomials, StochasticHermitePolynomials, ChebyshevT)

import pandas as pd
from pandas.compat.pickle_compat import _class_locations_map

_class_locations_map.update({
    ('pandas.core.internals.managers', 'BlockManager'): ('pandas.core.internals', 'BlockManager')
})



from linearbackend import lb

def histogram3d(samplesx, samplesy, bins=(20,20)):
    #make histogram stuff - set bins - I choose 20x20 because I have a lot of data
    hist, xedges, yedges = np.histogram2d(samplesx, samplesy, bins=bins, density = True)
    #xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:]) - abs(xedges[1]-xedges[0])

    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)

    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()

    cmap = cm.get_cmap('jet') # Get desired colormap - you can change this!
    max_height = np.max(dz)   # get range of colorbars so we can normalize
    min_height = np.min(dz)
    # scale each z to [0,1], and get their rgb values
    rgba = [cmap((k-min_height)/max_height) for k in dz] 

    ax = plt.gca()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort='average')
    plt.title("Histogram 2d")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.savefig("target")

def getRandomSamples(N = 1000 *2, verbose = False):
    
    X1 = np.random.uniform(-1.,1, N)
    X2 = np.random.uniform(-0.3,1, N)
    X3 = np.random.uniform(-0.5,1,N)
    X4 = np.random.uniform(0.5,1,N)

    one = lambda x : x**0

    RVsamples = np.array([ np.cos(X4**X1+X2+X3), 
                           X3 + X2*X1 **2  + 0.5*X4])

    if verbose:
        plt.subplot(1,2,1,projection = '3d')
        histogram3d(RVsamples[0,:], RVsamples[1,:], bins=(40,40))
        #plt.subplot(1,2,1)
        #c = plt.hist2d(RVsamples[0,:], RVsamples[1,:], bins = 100, normed = True)
        #plt.colorbar(c[3])

    return RVsamples.transpose()

class TT_PCE_RV(object):
    """
        Class representing a random variable given in polynomial chaos expansion (PCE) for a given family of tensorized stochastic orthogonal polynomials.
        The coefficient tensor for the PCE is represented in low rank tensor train format. 

    
    """

    def __init__(self, dim, family, modes, degrees, ranks ):
        """
            family =  LegendrePolynomials / StochasticHermitePolynomials / JacobiPolynomials

            It is assumed that the orthonormal families are defined on their standard domains, i.e. :
                    - LegendrePolynomials / JacobiPolynomials are defined on [-1,1]
                    - StchasticHermitePolynomials             are defined on IR

        """
            
        self.p = family(normalised = True)
        self.modes = modes
        assert(len(degrees) == modes)
        self.degrees = degrees

        if family ==  LegendrePolynomials:
            self.sampleGenerator = lambda N : lb.random.rand(self.modes, N) 
        elif family == StochasticHermitePolynomials : 
            self.sampleGenerator = lambda N : lb.random.randn(self.modes, N) 
        else: 
            raise NotImplemented("Other Polynomial classes not supported atm, u only need to add a reference coord sampler to make them working.")

        # init the data randomly 
        self.tt = TensorTrain( [dim] + [deg +1 for deg in self.degrees])
        self.tt.fill_random(ranks)


        self.psamples = None



    def samples(self, N, hold_ref_samples = True):
        """
            Generates N samples of the random variable.

            Optional the basis stochastic samples are cached, s.t. only the polynomial application on these samples
            is computed in a sample call. 

            If the number N is different to a precall of samples or hold_ref_samples == False, then new reference samples are computed.

        """

        if self.psamples == None or hold_ref_samples == False or  len(self.psamples) != N:

            samples = self.sampleGenerator(N)
            #  number of mode x polynomials evaluated
            pvals = [self.p.eval(self.degrees[i], x= samples[i,:], all_degrees=True) for i in range(self.modes)]

            # TODO update this to have it cached
            self.psamples = [ lb.stack(pv, axis = 1) for pv in pvals]


        
        return self.tt.contract_2nd_to_end_rank_one(self.psamples)

def kde_based_coord_change(eta_samples, verbose = False):
    modes = eta_samples.shape[0]

    # Kernel Density estimate of the KLE stochastic coordinates
    h = 0.5*1 * (4./(eta_samples.shape[1] * (dim +2)))**(1./(4+dim))

    print("bandwith for KLE variables = ", h)
    kdetypes = ['gaussian','epanechnikov']
    kde_KLE = [KernelDensity(kernel=kdetypes[0], bandwidth=h).fit(eta_samples[i,:].reshape(len(eta_samples[i,:]), 1)) for i in range(modes)]

    if verbose:
        plt.subplot(1,2,1)

        kdenames = ['kde 1', 'kde 2']
        for k in range(dim):
            X_plot = np.linspace(np.min(eta_samples[k])-3,
                                 np.max(eta_samples[k])+3, 1000)[:, np.newaxis]
            log_dens = kde_KLE[k].score_samples(X_plot)
            plt.plot(X_plot[:, 0], np.exp(log_dens), label= kdenames[k])

        plt.hist(eta_samples[0], bins = 200, label = r'$\eta_1$', density=True, alpha = 0.5)
        plt.hist(eta_samples[1], bins = 200, label = r'$\eta_2$', density = True, alpha = 0.5)
        plt.legend()
    
def stochastic_coord_change():
    
    # obtain samples of a full modell
    RVsamples = getRandomSamples()

    # size of the random sample (might representing a random field)
    dim = RVsamples.shape[0]

    # obtain a compressed representation via covariance based orthogonal decomposition
    C = np.cov(RVsamples)
    mu = np.mean(RVsamples)
    k, v = np.linalg.eig(C)

    modes = len(k)

    # with the orthgonal basis {v} in hand we can compute samples of the stochastic coordinates in the KLE expansions
    eta_samples = [None]*dim
    for i in range(dim):
        eta_samples[i] = 1./np.sqrt(k[i]) * (RVsamples*mu).transpose().dot(v[:,i])
    eta_samples = np.array([eta_samples[0], eta_samples[1]])
    print("Obtained KLE stoch coord samples with shape ", eta_samples.shape)



    p = StochasticHermitePolynomials(normalised=False)
    max_degree = 4
    stoch_modes= 2
    Ndofs = (max_degree+1) ** stoch_modes
    
    def generate_random_y(verbose2 = False):

        z1 = 2*np.random.rand(Ndofs)-1
        z1 = z1 / np.linalg.norm(z1)       # z gets length 1
        z_ortho = null_space(np.asmatrix(z1))
        pos = np.random.randint(0,Ndofs-1)
        z2 = z_ortho[:,pos]
        if verbose2:
            print("norm z1 = ", np.linalg.norm(z1))
            print("norm z2 = ", np.linalg.norm(z2))
            print("z1^T z2 = ", np.inner(z1,z2))

        # tensor train / svd at some point
        y = np.array([z1,z2])
        if verbose2:
            print("y shape : ", y.shape) 
            print("constraint : \n", sum( np.outer(y[:,i],y[:,i]) for i in range(Ndofs)))

        return y


    # for fixed y compute the marginal kdes 
    indices = MultiindexSet.createFullTensorSet(m = stoch_modes,
                                                p = max_degree)
    def build_PCE_KDE(y, plot_hist = False, verbose = False):

        ###### possible outside of method ######

        NPCE = 10000  
        normalsamples = np.random.randn(NPCE,stoch_modes)

         # implicit sample evaluated representation of tensor polynomials
        # pvals[alpha 1d ][ mode position] 
        pvals = p.eval(max_degree, x= normalsamples, all_degrees=True)


        # dim - dimensional samples of the PCE distribution for a fixed y
        eta_PCE_samples = sum(y[:,i].reshape(dim,1) * (pvals[idx[0]][:,0] * pvals[idx[1]][:,1]).reshape(1,NPCE) for i, idx in enumerate(indices))
        if verbose:
            print("PCE sample shape: ", eta_PCE_samples.shape)

        #TODO Centralize the samples to obtain the KDEs
        # then choose appropiate h
        h = 1 * (4./(eta_PCE_samples.shape[1] * (dim +2)))**(1./(4+dim))

        if verbose:
            print("bandwith for KLE variables = ", h)
        kde_PCE = [KernelDensity(kernel=kdetypes[0], bandwidth=h).fit(eta_PCE_samples[i,:].reshape(len(eta_PCE_samples[i,:]),1)) for i in range(KLELength)]
        if verbose:            
            print("PCE KDEs computed")
        if plot_hist:
            
            plt.subplot(1,2,1)
            kdenames = ['kde PCE 1', 'kde PCE 2']
            samplenames = [r'$\eta_1$', r'$\eta_2$']
            for k in range(KLELength):
                
                X_plot = np.linspace(np.min(eta_PCE_samples[k])-3,
                                    np.max(eta_PCE_samples[k])+3, 1000)[:, np.newaxis]
                log_dens = kde_PCE[k].score_samples(X_plot)
                plt.plot(X_plot[:, 0], np.exp(log_dens), label= kdenames[k])
                plt.hist(eta_PCE_samples[k], bins = 200, label = samplenames[k], density=True, alpha = 0.5)
        
            plt.legend()
            plt.show()
        return kde_PCE


    def eval_tensorised_log_liklihood(y):
        kde_PCE = build_PCE_KDE(y, plot_hist=False)
        res = np.sum(np.sum(kde_PCE[k].score_samples(eta_samples[k].reshape(-1,1))) for k in range(modes))
        return res

    if False: 
        ySamples = 10 
        best = -np.inf
        ybest = None

        for _ in range(ySamples):
            y = generate_random_y()
            logliklihood = eval_tensorised_log_liklihood(y)

            if logliklihood > best:
                best = logliklihood
                ybest = y
            print("logliklihood value ", logliklihood)


    NPCE = 100
    normalsamples = np.random.randn(NPCE,stoch_modes)

    # implicit sample evaluated representation of tensor polynomials
    # pvals[alpha 1d ][ mode position] 
    pvals = p.eval(max_degree, x= normalsamples, all_degrees=True)

    def L(z,jk, verbose = False, which = ''):

        # dim - dimensional samples of the PCE distribution for a fixed y
        eta_PCE_jk_samples = sum(z[i] * pvals[idx[0]][:,0] * pvals[idx[1]][:,1] for i, idx in enumerate(indices))

        #TODO Centralize the samples to obtain the KDEs
        # then choose appropiate h
        d = 1
        h = 1 * (4./(len(eta_PCE_jk_samples) * (d +2)))**(1./(4+d))
        kde_PCE_jk = KernelDensity(kernel=kdetypes[0], bandwidth=h).fit(eta_PCE_jk_samples.reshape(-1,1))
        res = sum(kde_PCE_jk.score_samples(eta_samples[jk].reshape(-1,1)))
        if verbose: print(which, "  L (z) = ", res)
        return res
    
    # constraint that z has norm 1
    nonlinear_constraint = NonlinearConstraint(lambda z: np.inner(z,z) ,1., 1)

    # constraint that z is orthgonal to mat
    def get_ortho_constraint(mat):
        return LinearConstraint(mat, [0.]*mat.shape[0], [0.]*mat.shape[0])
    
    Loptbest = -np.inf
    ybest = np.zeros((dim, Ndofs))

    # recurrent optimization problem
    for idx in [[0,1], [1,0]]:

        # MINIMIZE the j1 problem 
        j1 = idx[0]
        # z1 in IR^Ndofs with ||z_j1|| = 1 as start value
        z_j1 = 2*np.random.rand(Ndofs)-1
        z_j1 = z_j1 / np.linalg.norm(z_j1)
        print("test L evaluation : ", L(z_j1,j1))
        f_j1 = lambda z: -L(z,j1, verbose= True, which = str(idx) + ' with j1')
        res = minimize(f_j1, z_j1, method='trust-constr', constraints = [nonlinear_constraint], options = {'verbose':1, 'maxiter':100})
        #print(res)
        time.sleep(2)
        z_j1_opt = res.x


        # Minimize the j2 problem 
        j2 = idx[1]
        mat = np.zeros((1,Ndofs))
        mat[0,:] = z_j1_opt

        # generate a random start vector fullfilling orthgonality and spherical constraint
        mat_ortho = null_space(np.asmatrix(mat))
        pos = np.random.randint(0,Ndofs-1)
        z_j2  = mat_ortho[:,pos] # z j2 is normalized already
        z_j2 = z_j2 / np.linalg.norm(z_j2)


        linconstr = get_ortho_constraint(mat)
        f_j2 = lambda z: -L(z,j2, verbose = True, which = str(idx) + ' with j2')
        res = minimize(f_j2, z_j2, method='trust-constr', constraints = [linconstr, nonlinear_constraint])
        z_j2_opt = res.x
        
        Lopt = - f_j1(z_j1_opt) - f_j2(z_j2_opt)

        if Lopt > Loptbest : 
            Loptbest = Lopt
            ybest[j1,:] = z_j1_opt
            ybest[j2,:] = z_j2_opt
            

    build_PCE_KDE(ybest, plot_hist = True, verbose = True)


      



    exit()


    build_PCE_KDE(ybest, plot_hist=True)

    
    
    
    exit()

    

    # Gaussian KDE
    kdetypes = ['gaussian','epanechnikov']
    kde = KernelDensity(kernel=kdetypes[0], bandwidth=0.75).fit(samples)

    X_plot = np.linspace(np.min(samples)-3, np.max(samples)+3, 1000)[:, np.newaxis]
    log_dens = kde.score_samples(X_plot)
    plt.fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
    plt.plot(X_plot[:, 0], np.exp(log_dens), 'r-')
    plt.text(-3.5, 0.31, "Gaussian Kernel Density")

    plt.hist(samples, bins = 100, density = True, alpha = 0.5)
    plt.show()

def family_coord_samples(family, N):

    if isinstance(family, LegendrePolynomials):
        return 2*lb.random.rand(N)-1
    elif isinstance(family, StochasticHermitePolynomials):
        return lb.random.randn(N)
    else:
        raise NotImplementedError("Other polynomial base samplesare not implemented yet")




# TODO: 
#  Debug the stack computation 
#  Introduce debiased symmetric multiscale sinkhorn algorithm
#  KDE based fitting
#
#



class Vectorial_Extended_TensorTrain(object):

    def __init__(self, vdim, families, degrees, init_ranks):
        """
        @param families: either constant polynomial family or a list of polynomial families of len(degrees)
        """
        assert(len(families) == len(degrees))
        self.tt = TensorTrain([vdim] + [d+1 for d in degrees])


        self.tt.fill_random(init_ranks)
        self.ncoords = len(degrees)
        self.fams = families
        self.degrees = degrees
        self.vdim = vdim

    def __call__(self, x):
        assert(x.shape[1] == len(self.fams))
        # basis family evaluations of each basis function for each coordinate
        b_vals = [lb.stack(fam.eval(self.degrees[i], x= x[:,i], all_degrees=True), axis = 1) for i, fam in enumerate(self.fams)]
        return self.tt.contract_2nd_to_end_rank_one(b_vals)

    def grad(self, x):
        raise NotImplementedError("The Gradient for Vectorial Extendd TensorTrain is not implemented yet.")

    def fit(self, x, y, iterations= 2, reg_param_order = 1e-6, reg_decay_rate = 0.75, reg_decay_rhythm = 10,  verbose = True):
        """
            Fits the Extended Tensortrain to the given data (x,y) of some target function 
                     f : K\subset IR^n to IR^m
                                     x -> f(x) = y.

            for a given metric_options:

                metric_options = { 'type' :  metric :
                                   'l2_param'    : #value, 
                                   'l1_param'    : #value,
                                   'salsa_param' : #value}


                The inserted value of metric is :

                    'regularised-L2' 
                    =================

                    Represents fitting in an empirical fashion of 
                                
                                || f - tt ||_L^2(w)  +  l2_param * || tt ||_1

                    Where w is a (possible unknown) underlying probability measure and x is drawn from it.
                        
                    Here ||.||_1 represents an implicit skp induce norm ||.||_1 = <.,.>_1 s.t. 
                                
                                || tt ||_1  = || tt.coeffs ||_F, 

                    this norm is given by the norm s.t. the underlying basis is orthonormal w.r.t. <.,.>_1.

            @param x : input parameter of the training data set : x with shape (b,d)   b \in \mathbb{N}
            @param y : output data with shape (b,m)
        """

        ycpu = y.cpu()
        b = y.shape[0]

        val = self.__call__(x)
        r0 = (lb.linalg.norm(val - y)**2).item() #start residuum to define the regularization order parameter
        

        decaying_reg_param = reg_param_order #* r0

          # rank 1 tensor which will be contracted with  2nd to dth position
        u = [lb.stack(fam.eval(self.degrees[i], x= x[:,i], all_degrees=True), axis = 1) for i, fam in enumerate(self.fams)]

        # 0 - orthogonalize, s.t. sweeping starts on first component
        self.tt.set_core(mu = 0)

        # initialize lists for left and right contractions
        R_stack = [lb.ones((b, 1))]
        L_stack = [lb.ones((b, 1))]

        d = self.tt.n_comps

        def add_contraction(mu, list, side='left'):

            assert ((side == 'left' or side == 'right') or (side == +1 or side == -1))
            with TicToc(key=" o left/right contractions ", do_print=False, accumulate=True, sec_key="ALS: "):  
                if mu > 0 :   
                    core_tensor = self.tt.comps[mu]
                    data_tensor = u[mu-1]
                    contracted_core = lb.einsum('idr, bd -> bir', core_tensor, data_tensor)
                    if (side == 'left' or side == -1):
                        list.append(lb.einsum('bir, bvi -> bvr', contracted_core, list[-1]))
                    elif (side == 'right' or side == 1):
                        list.append(lb.einsum('bir, br -> bi', contracted_core, list[-1]))
                    else:
                        raise ValueError("side = left (-1)  / right ( +1) only.")
                elif mu == 0:
                    # 1 x vdim x r1
                    contracted_core = self.tt.comps[mu]  # the first component is not aligned with a basis function

                    if (side == 'left' or side == -1):
                        s = self.tt.comps[0].shape
                        # i = j = 1 include a reshape to bvr    b = batchsize,  v = vector value size, r = free rank 
                        list.append(lb.einsum('ivr, bj -> ibvrj', contracted_core, list[-1]).reshape(list[-1].shape[0], s[1], s[2])) 
                    else: 
                        raise ValueError('mu == 0 and right should not be called in a sweep')

        def solve_local(mu,L,R, method = 'lqsrt'):

            s = self.tt.comps[mu].shape
            if method ==  'lqsrt': 
                if mu == 0 :
                    # best approximation based on vectorized formulation
                    vdim = s[1]
                    A = lb.kron(lb.eye(vdim), R)
                    rhs = y.T.flatten()
                    ATA, ATy = A.T@A, A.T@rhs
                    if decaying_reg_param is not None:
                        assert isinstance(decaying_reg_param,float)
                        ATA += decaying_reg_param * lb.eye(ATA.shape[0])

                    #print("cond = ", lb.linalg.cond(ATA))
                    c = lb.linalg.solve(ATA,ATy) 
                    self.tt.comps[mu]  =  c.reshape(s[0], s[1], s[2]) 
                   
                else : 
                    A = lb.einsum('bvi,bj,br->bvijr', L, u[mu-1], R)
                    A = A.reshape(A.shape[0]*A.shape[1], A.shape[2]*A.shape[3]*A.shape[4])
                    rhs = y.flatten()

                    ATA, ATy = A.T@A, A.T@rhs
                    if decaying_reg_param is not None:
                        assert isinstance(decaying_reg_param,float)
                        ATA += decaying_reg_param * lb.eye(ATA.shape[0])
                    
                    #print("cond = ", lb.linalg.cond(ATA))
                    c = lb.linalg.solve(ATA,ATy) 
                    self.tt.comps[mu] = c.reshape(s[0],s[1],s[2])

            else : 
                if mu > 0 : 
                    A = lb.einsum('bvi,bj,br->bvijr', L, u[mu-1], R)
                    A = A.reshape(A.shape[0], A.shape[1], A.shape[2]*A.shape[3]*A.shape[4])
                else: 
                    A = R 

                def F(c) :                 
                    lbc = lb.asarray(c)
                    if mu == 0: 
                        lbc = lbc.reshape(s[1],s[2])
                        Ac = A @ lbc.T
                    else: 
                        Ac = lb.einsum('bvi,i->bv', A, lbc)

                    val, reg = 0.5*lb.linalg.norm(Ac - ycpu)**2, decaying_reg_param * lb.linalg.norm(lbc)  # l^2 regularization. L^
                    #print("\t\t\t val = ", val.item(), "   reg = ", reg.item()) 
                    return (val+reg).item()
                x0 = self.tt.comps[mu].reshape(s[0]*s[1]*s[2])

                res  = minimize(F, x0, options = {"disp" : True, "maxiter" : 20, 'eps': 1.4901161193847656e-08})#, method = 'Nelder-Mead')#, )
                res = lb.asarray(res.x)

                self.tt.comps[mu] = res.reshape(s[0],s[1],s[2])

        # before the first forward sweep we need to build the list of right contractions
        for mu in range(d-1,0,-1):
            add_contraction(mu, R_stack, side='right')

        # forward and backward sweep iteration
        for niter in range(iterations):

            if (niter % reg_decay_rhythm) == 0 : 
                #val = self.__call__(x)
                #rk = (lb.linalg.norm(val - y)**2).item() 

                # decaying_reg_param = max(decaying_reg_param * reg_decay_rate, rk * reg_param_order * reg_decay_rate )

                decaying_reg_param *= reg_decay_rate


            #print("{c}Update{r} : rank r{p} = {c1}{rank}{r} -> {c}{rankn}{r}".format(p = pos+1, rank=self.rank[pos+1], rankn = new_rank,c1=Fore.RED,c=Fore.GREEN, r=Style.RESET_ALL) )
            print("{c}".format(c=Fore.GREEN) + 3 * d * "=" + " sweep iteration = {n} ".format(n=niter+1) + 3 * d * "=" + "{r}   reg param = {v}".format(r=Style.RESET_ALL, v = decaying_reg_param))
            # forward half-sweep
            for mu in range(d-1):
                self.tt.set_core(mu)
                if mu > 0:
                    add_contraction(mu-1, L_stack, side='left')
                    del R_stack[-1]
                #print("SOLVE ON MU = ", mu, " UPWARD")
                solve_local(mu,L_stack[-1],R_stack[-1])

                if verbose : 
                    val = self.__call__(x)
                    space = " "*(mu+1) + "{c}\\{r}".format(c=Fore.GREEN, r=Style.RESET_ALL) + " " * (d-mu)
                    print(space, (lb.linalg.norm(val - y)**2).item())

            # before back sweep
            self.tt.set_core(d-1)    # patched
            add_contraction(d-2, L_stack, side='left')
            del R_stack[-1]

            # backward half sweep
            for mu in range(d-1,0,-1):
                self.tt.set_core(mu)
                if mu < d-1:
                    add_contraction(mu+1, R_stack, side='right')
                    del L_stack[-1]
                solve_local(mu,L_stack[-1],R_stack[-1])
                
                if verbose : 
                    val = self.__call__(x)
                    space = " "*(mu) + "{c}/{r}".format(c=Fore.GREEN, r=Style.RESET_ALL) + " " * (d-mu+1)
                    print(space, (lb.linalg.norm(val - y)**2).item())

            # before forward sweep
            self.tt.set_core(0)  # patched
            add_contraction(1, R_stack, side='right')
            del L_stack[-1]


    def sinkhornfit(self, nxsamples, y, iterations = 2, reg_param_l2 = 1e-3, reg_param_l1 = 0., moments_info = None,  verbose = False, verbose_min_surfaces = False):



    
        """
        Let the vectorial extended tensor train xTT be  vdim-valued with given stochastic polynomial family. 

        The parameter y is interpreted as random realisations of some unknown random vector f(\omega). 

        Given nxsamples we draw samples from the underlying stochastic coordinate systems induced by the stochastic polynomial families
        denoted as X. Then for fixed coefficient components in the underlying tensor train, xTT(X) are samples too.

        Then this fitting method aims for reducing the optimal transport costs of moving samples 

                y  ->  xTT(X)

        in terms of Wasserstein metric, based on geomless computation. 

        """


        vdim = y.shape[1] 
        
        if vdim == 1:
            s1, i1= torch.sort(y, dim=0)
        else:
            s1 = torch.as_tensor(y, device = "cuda").contiguous()

        # size of the data batch
        b = y.shape[0]
        x = lb.stack([family_coord_samples(fam, nxsamples) for fam in self.fams], axis = 1)
        if verbose and y.shape[1] == 1:
            rows = max(2,math.ceil((iterations +1) / 4))

            print("rows = ", rows)

            fig, ax = plt.subplots(rows, 4,figsize=(15,15))

            yfit = self.__call__(x)

            ax[0,0]
            #plt.subplot(1,iterations+1, 1)
            ax[0,0].hist(y.cpu().reshape(y.shape[0]), bins = 25, alpha = 0.5, label = 'target', density = True)
            ax[0,0].hist(yfit.cpu().reshape(yfit.shape[0]), bins = 25, alpha = 0.5,  label = 'initiol', density = True)
            ykde = stats.gaussian_kde(y.cpu().reshape(y.shape[0]))
            yfitkde = stats.gaussian_kde(yfit.cpu().reshape(yfit.shape[0]))

            


            X = lb.linspace(min(y.tolist())[0], max(y.tolist())[0], 1000)
            ax[0,0].plot(X.cpu(), ykde(X.cpu()), label = 'tar kde', linewidth = 4)         

            X = lb.linspace(min(yfit.tolist())[0], max(yfit.tolist())[0], 1000)
            ax[0,0].plot(X.cpu(), yfitkde(X.cpu()), label = 'initial guess fit kde', linewidth = 4)  

            plt.legend()

        # rank 1 tensor which will be contracted with  2nd to dth position
        u = [lb.stack(fam.eval(self.degrees[i], x= x[:,i], all_degrees=True), axis = 1) for i, fam in enumerate(self.fams)]

        # 0 - orthogonalize, s.t. sweeping starts on first component
        self.tt.set_core(mu = 0)

        # initialize lists for left and right contractions
        R_stack = [lb.ones((b, 1))]
        L_stack = [lb.ones((b, 1))]

        d = self.tt.n_comps

        def add_contraction(mu, list, side='left'):

            #print("Add contraction with mu = {mu}. ".format(mu=mu))

            assert ((side == 'left' or side == 'right') or (side == +1 or side == -1))
            with TicToc(key=" o left/right contractions ", do_print=False, accumulate=True, sec_key="ALS: "):  
                if mu > 0 :   
                    core_tensor = self.tt.comps[mu]
                    data_tensor = u[mu-1]
                    contracted_core = lb.einsum('idr, bd -> bir', core_tensor, data_tensor)
                    if (side == 'left' or side == -1):
                        list.append(lb.einsum('bir, bvi -> bvr', contracted_core, list[-1]))
                    elif (side == 'right' or side == 1):
                        list.append(lb.einsum('bir, br -> bi', contracted_core, list[-1]))
                    else:
                        raise ValueError("side = left (-1)  / right ( +1) only.")
                elif mu == 0:
                    # 1 x vdim x r1
                    contracted_core = self.tt.comps[mu]  # the first component is not aligned with a basis function

                    if (side == 'left' or side == -1):
                        s = self.tt.comps[0].shape
                        # i = j = 1 include a reshape to bvr    b = batchsize,  v = vector value size, r = free rank 
                        list.append(lb.einsum('ivr, bj -> ibvrj', contracted_core, list[-1]).reshape(list[-1].shape[0], s[1], s[2])) 
                    else: 
                        raise ValueError('mu == 0 and right should not be called in a sweep')

        def getPCEsamples(x, mu, L, R):
            #x = lb.asarray(x)
            G = x.reshape(self.tt.comps[mu].shape) # ri d ri+1
            if mu == 0 : 
                # G has shape 1 x v x r1
                samples = lb.einsum('ivr, br->ibv', G, R).reshape(b, self.vdim)
            else:
                # L = b  v  r_1
                # G = r_1  d  r_2
                # R = b r_2   
                core_tensor = G
                data_tensor = u[mu-1]
                contracted_core = lb.einsum('rds, bd -> brs', core_tensor, data_tensor) # b    r1   r2 
                CR = lb.einsum('brs,bs-> br', contracted_core, R)
                samples = lb.einsum('bvr,br->bv', L, CR)

            return samples
          

            # DEBUG CODE  #TODO DELETE:
            # ignore L and R :

            def debugCode(x, mu):

                # compute L :
                if mu > 0 : 
                    # 1 x v x r
                    myL = self.tt.comps[0]
                    data = [lb.einsum('rds, bd-> brs', self.tt.comps[k], u[k-1]) for k in range(1, mu)]
                    for ii, dat in enumerate(data) :
                        if ii == 0 : 
                            myL = lb.einsum('ivr, brs -> ibvs', myL, dat).reshape(b,self.vdim, dat.shape[2]) # bvs
                        else: 
                            myL = lb.einsum('bvr, brs->bvs', myL, dat)
                # compute R : 
                data = [lb.einsum('rds, bd -> brs', self.tt.comps[k], u[k-1]) for k in range(mu+1, d)]
                if mu < d-1:
                    myR = data[-1]
                    for k in range(1, len(data)):
                        myR = lb.einsum('brs, bsi-> bri', data[-1-k], myR)

                if mu > 0 : 
                    core_tensor = x.reshape(self.tt.comps[mu].shape)
                    data_tensor = u[mu-1]
                    myG = lb.einsum('rds, bd -> brs', core_tensor, data_tensor)
                else:
                    myG = x.reshape(self.tt.comps[mu].shape)

                # CONTRACT  L G R to obtain samples
                if mu == 0:
                    # G  1 x v x r    *  R 
                    samples = lb.einsum('ivr, brj -> ibvj', myG, myR).reshape(b, self.vdim)


                    # debug area 
                    G = x.reshape(self.tt.comps[mu].shape) 
                    tsamples = lb.einsum('ivr, br->ibv', G, R).reshape(b, self.vdim)
                    errL = 0
                   # print("myR .shape = ", myR.shape)
                    sr = R.shape
                    
                    Rs =  R.reshape(sr[0],sr[1], 1)
                    #print("Rs .shape = ", Rs.shape)
                    errR = lb.linalg.norm(myR - Rs)
                    #print("mu = {mu} :  L : {L} , R : {R} ".format(mu=mu,L=errL, R = errR))
                    print("mu = {mu}".format(mu=mu))
                    print("sample err = ", lb.linalg.norm(samples - tsamples))
                    ######

                elif mu < d-1:
                    GR = lb.einsum('brs, bsi->bri', myG, myR)
                    samples = lb.einsum('bvr,bri->bvi', myL, GR).reshape(b,self.vdim)


                    # debug area
                    G = x.reshape(self.tt.comps[mu].shape) 
                    core_tensor = G
                    data_tensor = u[mu-1]
                    contracted_core = lb.einsum('rds, bd -> brs', core_tensor, data_tensor) # b    r1   r2 
                    CR = lb.einsum('brs,bs-> br', contracted_core, R)
                    tsamples = lb.einsum('bvr,br->bv', L, CR)
                   
                    #errR = lb.linalg.norm(myR - R)
                    
                    print("mu = {mu}".format(mu=mu))
                    print("sample err = ", lb.linalg.norm(samples - tsamples))
                    ######
                else: 
                    # mu == d-1:
                    samples = lb.einsum('bvr,bri->bvi', myL, myG).reshape(b, self.vdim)



                    #print("my L shape =", myL.shape)
                    #print(" L shape ", L.shape)
                    #print("myL :\n", myL)
                    #print("=======================")
                    #print("L : \n ", L)


                    # debug area
                    G = x.reshape(self.tt.comps[mu].shape) 
                    core_tensor = G
                    data_tensor = u[mu-1]
                    contracted_core = lb.einsum('rds, bd -> brs', core_tensor, data_tensor) # b    r1   r2 
                    CR = lb.einsum('brs,bs-> br', contracted_core, R)
                    tsamples = lb.einsum('bvr,br->bv', L, CR)
                    errL = lb.linalg.norm(myL - L)
                    errR = 0
                    #print("mu = {mu} :  L : {L} , R : {R} ".format(mu=mu,L=errL, R = errR))
                    #print("mu = {mu} = d-1 :".format(mu=mu))
                    print("sample err = ", lb.linalg.norm(samples - tsamples))
                    ######

                return samples

            correctsamples = debugCode(x, mu)

            #return correctsamples
            
        loss = SamplesLoss(loss="sinkhorn", p=2, blur = 0.05, scaling = 0.9)#, blur=.05, scaling=0.8)   #blurring = 0.05

        def solve_local(mu,L,R, iterationnumber):
            s = self.tt.comps[mu].shape
            x0 = self.tt.comps[mu].reshape(s[0]*s[1]*s[2])

            def F(x):

                xx = lb.asarray(x)
                
                #print("x = ", xx)

                samples = getPCEsamples(xx, mu,L,R)
                #samples = lb.asarray(samples)
                #samples = torch.as_tensor(samples, device = "cuda").contiguous()

                l = 0.

                if vdim == 1 and False:
                    s2, i2 = torch.sort(samples, dim = 0)
                    d = s1.squeeze() - s2.squeeze()
                    l = 1./(2*len(d)) * lb.linalg.norm(d)**2   # 
  
                else : 
                    l = loss(y, samples.contiguous())
                

                    s2, i2 = torch.sort(samples, dim = 0)
                    d = s1.squeeze() - s2.squeeze()
                    print("                               l = {l}  ( {k} ) ".format(l=l, k = 1./(2*len(d)) * lb.linalg.norm(d)**2))
                    
                
                    #else:
                    #ll = loss(y, samples.contiguous())
                    #print("     \t\t          strange transport cost  : ", ll.item())


                if reg_param_l2 > 0 : 
                    l += reg_param_l2 * lb.linalg.norm(xx)
                if reg_param_l1 > 0 : 
                    l +=  reg_param_l1* lb.linalg.norm(xx, ord=1)


                if moments_info is not None:
                    # TODO : GET THIS TO STACK LEVEL same as L and R

                    d = self.tt.comps[0].shape[1]

                    if moments_info["2nd_moment_param"] > 0 :
                        # Compute the 2nd moment  = moments(order = 2)
                        if mu == 0 : 
                            comp = xx.reshape(s)
                        if mu > 0 :
                            comp = self.tt.comps[0]
                        M = lb.kron(comp[0,:,:], comp[0,:,:])  # d*d x r*r   object

                        for k in range(1, self.ncoords+1):
                            dk = self.tt.dims[k]

                            if k != mu : 
                                comp = self.tt.comps[k]
                            else:
                                comp = xx.reshape(s)

                            M = sum( lb.dot(M,lb.kron(comp[:,i,:], comp[:,i,:])) for i in range(dk))
                        M = M.reshape(d,d)

                        l += moments_info["2nd_moment_param"] * lb.linalg.norm(M - moments_info["tar_2nd_moment"])**2

                    if moments_info["mean_param"] > 0 :
                        # Compute the first moment :  = moments(order = 1)
                        if mu == 0:
                            comps = [xx.reshape(s)] + self.tt.comps[1:]
                        else:
                            comps = self.tt.comps[:mu] + [xx.reshape(s)] + self.tt.comps[mu+1:]

                        u = [lb.zeros((1,deg+1)) for deg in self.degrees]
                        for ei in u:
                            ei[0,0] = 1.
                        G = [ lb.einsum('ijk, bj->ibk', c, v)  for  c,v in zip(comps[1:], u) ]  
                        mean = G[-1]
                        for pos in range(self.tt.n_comps-3, -1,-1):
                            mean = lb.einsum('ibj, jbk -> ibk', G[pos], mean) # k = 1 only

                        mean = lb.einsum('idj,jbk->ibdk', comps[0], mean)  # i==k==1 

                        mean = mean.reshape(mean.shape[1], mean.shape[2])


                        l += moments_info["mean_param"] * lb.linalg.norm(mean - moments_info["tar_mean"])**2
                    

                    
        
                

                add_info = ""
                #if moments_info is not None:
                #    add_info = " mean = {m} (target mean = {tm} \n  2nd moment =\n {m2} \n target 2nd moment = \n{tm2}".format(m = mean, tm = moments_info["tar_mean"], m2 = M, tm2=  moments_info["tar_2nd_moment"])
                
                print("\t\t\t  Wasserstein distance: {:.3f}".format(l.item()))
                #print("\t\t\t sorting distance : ", l.item(), add_info)
                return l.item()


         

            if verbose_min_surfaces : 
                if mu == 0 : 
                    X = lb.linspace(-2,2,1000)
                    Y = lb.asarray([F(xx.cpu()) for xx in X])

                    plt.figure()
                    plt.semilogy(X.cpu(), Y.cpu())
                if mu == 1:
                    X = lb.linspace(-2,1,40)
                    Y = lb.linspace(-2,2,40)
                    Z =  [F ( [xx.cpu(),yy.cpu()] )  for xx in X for yy in Y]
                    Z = lb.asarray(Z)
                    X,Y = lb.meshgrid(X,Y)
                    Z = Z.reshape(X.shape)
                    plt.figure()
                    ax = plt.axes(projection='3d')
                    ax.plot_surface(X.cpu(), Y.cpu(), Z.cpu(),cmap='viridis', edgecolor='none')


                    
                    ax.azim = -60
                    ax.dist = 10
                    ax.elev = 60
                

            #res  = minimize(F, x0, options = {"maxiter" : 5})
            res  = minimize(F, x0.cpu())#, method = 'Nelder-Mead', options = {"maxiter" : 50})
            res = lb.asarray(res.x)


            if verbose_min_surfaces : 
                name = "LocProblem_mu{mu}_{n}D_iter{i}".format(mu=mu,n=res.shape[0], i=iterationnumber)
                ax = plt.gca()
                if res.shape[0] == 1 : 

                    print("x0 = ", x0 , " res = ", res)
                    
                    ax.scatter(x0[0].cpu(), F(x0.cpu()), c='red', label = "x0 = " + str(x0.item()))
                    ax.scatter(res[0].cpu(), F(res.cpu()), c='black', label = str(res.item()))
                    plt.legend()
                    plt.savefig(name + ".png")

                elif res.shape[0] == 2 : #and iterationnumber == 0 : 

                    print("x0 = ", x0 , " res = ", res)

                    xx0 = x0.cpu()
                    ax.scatter(xx0[0], xx0[1], F(xx0)*1.1, c='black', label = "x0 = " + str(xx0))
                    ax.scatter(res[0].cpu(), res[1].cpu(), F(res.cpu())*1.1, c='red', label = str(res))
                    plt.legend()

                    plt.savefig(name + ".png")


            self.tt.comps[mu] = res.reshape(s[0],s[1],s[2])

        # before the first forward sweep we need to build the list of right contractions
        for mu in range(d-1,0,-1):
            add_contraction(mu, R_stack, side='right')

        # forward and backward sweep iteration
        for niter in range(iterations):
            # forward half-sweep
            for mu in range(d-1):
                self.tt.set_core(mu)
                if mu > 0:
                    add_contraction(mu-1, L_stack, side='left')
                    del R_stack[-1]
                solve_local(mu,L_stack[-1],R_stack[-1], iterationnumber = niter)

            # before back sweep
            self.tt.set_core(d-1)    # patched
            add_contraction(d-2, L_stack, side='left')
            del R_stack[-1]

            # backward half sweep
            for mu in range(d-1,0,-1):
                self.tt.set_core(mu)
                if mu < d-1:
                    add_contraction(mu+1, R_stack, side='right')
                    del L_stack[-1]
                solve_local(mu,L_stack[-1],R_stack[-1], iterationnumber = niter)

            # before forward sweep
            self.tt.set_core(0)  # patched
            add_contraction(1, R_stack, side='right')
            del L_stack[-1]


            print("finished one iteration")
            print("======================================")
            # B = 50
            # plt.subplot(1, iterations, niter+1)
            # plt.hist(y, bins = B, alpha = 0.5, label = 'target')
            # #plt.scatter(y[:,0], y[:,1], color = 'blue', s = 1)
            # currSamples = self.__call__(x)
            # #plt.scatter(currSamples[:,0], currSamples[:,1], color = 'red', s = 1)
            # plt.hist(currSamples, bins = B, alpha = 0.5,  label = 'fit')
            # #plt.legend()
            # #plt.savefig("plot.png")"

            if verbose and y.shape[1] == 1:
                yfit = self.__call__(x)
                print(" Try Plotting now" )

                row =  int((niter+1) / 4)
                col =  (niter+1) % 4

                print("TRY ACCESS (row, col ) = ", row, col)


                #plt.subplot(1,iterations+1, niter + 2)
                ax[row,col].hist(y.cpu().reshape(y.shape[0]), bins = 50, alpha = 0.5, label = 'target', density = True)
                ax[row,col].hist(yfit.cpu().reshape(yfit.shape[0]), bins = 50, alpha = 0.5,  label = 'fit', density = True)


                ykde = stats.gaussian_kde(y.cpu().reshape(y.shape[0]))
                yfitkde = stats.gaussian_kde(yfit.cpu().reshape(yfit.shape[0]))

                X = lb.linspace(min(y.tolist())[0], max(y.tolist())[0], 1000)
                ax[row,col].plot(X.cpu(), ykde(X.cpu()), label = 'tar kde', linewidth = 2)         

                X = lb.linspace(min(yfit.tolist())[0], max(yfit.tolist())[0], 1000)
                ax[row,col].plot(X.cpu(), yfitkde(X.cpu()), label = 'fit kde', linewidth = 2)           

                ax[row,col].legend()


            TENSOR = self.tt.full()
            print("CURRENT TENSOR FIT : \n", TENSOR)

        
        print("surr empirical mean : ", lb.mean(self.__call__(x)))
        print("surr empirical cov : ", lb.cov(self.__call__(x)))

        if verbose and y.shape[1] == 1:
            plt.setp(ax, ylim=(0,1.))
            plt.savefig("plot.png")


    def moments(self, order):
        """
            The class representing a random variable  X = (X_i)_{i=1}^d with values in IR^d.

            order = 1 : Computes the mean vector IE[X] \in IR^d


            order = 2 : 
            Computes the matrix of the the expectation of the inner products

                                    IE [ X_i X_j ]  \in IR^{d,d}

        """
        # TODO: THE FAMILY MUST BE ASSERTED TO BE ORTHONORMAL

        if order == 1 : 
            u = [lb.zeros((1,deg+1)) for deg in self.degrees]
            for ei in u:
                ei[0,0] = 1.
            return self.tt.contract_2nd_to_end_rank_one(u)
        
        elif order == 2 : 

            d = self.tt.comps[0].shape[1]
            M = lb.kron(self.tt.comps[0][0,:,:], self.tt.comps[0][0,:,:])  # d*d x r*r   object
            for k in range(1, self.ncoords+1):
                dk = self.tt.dims[k]
                M = sum( lb.dot(M,lb.kron(self.tt.comps[k][:,i,:], self.tt.comps[k][:,i,:])) for i in range(dk))
            M = M.reshape(d,d)
            return M
        
        else: 
            raise NotImplementedError("Other moments than 1st (Mean) and 2nd are not implemented yet.")

def randomSamples1DPolynomial(N):

    A = 2*lb.random.rand(N) -1
    B = 2*lb.random.rand(N) -1

    
    A = LegendrePolynomials().eval(2, A, all_degrees=True)
    B = LegendrePolynomials().eval(2, B, all_degrees=True)


    S = A[2] + B[2] + 2* A[1]*B[1] + A[1] + B[1] + A[0]*B[0]


    #S =  A ** 2 + B** 2 + A*B + A + B
    plt.hist(S, bins = 100, density = True)
    plt.savefig("hist")

    return S.reshape(len(S),1)

def randomSamples1DRandomPolynomial(N, deg = 2):

    P = 2*lb.random.rand(N) -1
    Q = 2*lb.random.rand(N) -1
    R = 2*lb.random.rand(N) -1


    P = LegendrePolynomials().eval(deg, P, all_degrees=True)
    Q = LegendrePolynomials().eval(deg, Q, all_degrees=True)
    R = LegendrePolynomials().eval(deg, R, all_degrees=True)


    C = 4*lb.random.randn((deg+1)**3)-2
    C = C.reshape(deg+1,deg+1,deg+1)

    S = sum( C[i,j,k]* P[i]*Q[j] * R[k] for i in range(deg+1) for j in range(deg+1) for k in range(deg+1))

      #S =  A ** 2 + B** 2 + A*B + A + B
    plt.hist(S, bins = 100, density = True)
    plt.savefig("hist")

    return S.reshape(len(S),1), C

def testSinkhorn():



    vdim = 1
    N = 1000

    x = lb.random.rand(N, vdim)
    y =2* lb.random.randn(N, vdim) +1

    # diameter : rough estimation of the maximum distance of points
    #
    # scaling (float, default=.5) – If loss is "sinkhorn",
    #  specifies the ratio between successive values of σ=ε1/p in the ε-scaling descent.
    #  This parameter allows you to specify the trade-off between speed (scaling < .4) and accuracy (scaling > .9).
    sinkhorn = lambda blur, scaling = 0.5, debias = True : SamplesLoss(loss="sinkhorn", p=2, blur=blur, scaling=scaling, debias = debias, diameter = None)

    
    s1, _ = torch.sort(x, dim = 0)
    s2, _ = torch.sort(y, dim = 0)
    d = s1.squeeze() - s2.squeeze()
    OT = (1./(2*len(d)) * lb.linalg.norm(d)**2).item()

    blurs = [0.001,0.01,0.02,0.05,0.125]
    las, lmas, ls, lls, r_accs,  r_mas, rs, r_lows = [], [], [], [], [], [], [], []

    for blur in blurs : 
    
        scale = 0.5
        scale_medium = 0.9
        scale_acc = 0.99
        scale_low = 0.2

        # default loss value
        l = sinkhorn(blur, scale, True)(x,y)
        lm =  sinkhorn(blur, scale_medium, True)(x,y)
        la = sinkhorn(blur, scale_acc, debias = True)(x,y)
        ll = sinkhorn(blur, scale_low, debias = True)(x,y)

 
        l_sqrt     = sinkhorn(math.sqrt(2.)*blur, scale)(x,y)


        lm_sqrt = sinkhorn(math.sqrt(2.)*blur, scale_medium)(x,y)
        la_sqrt = sinkhorn(math.sqrt(2.)*blur, scale_acc)(x,y)
        ll_sqrt = sinkhorn(math.sqrt(2.)*blur, scale_low)(x,y)

        r_low = (2*ll - ll_sqrt).item()
        r     = (2*l - l_sqrt).item()
        r_acc = (2*la - la_sqrt).item()
        r_m   = (2*lm - lm_sqrt).item()

        l = l.item()
        la= la.item()
        ll = ll.item()
        lm = lm.item()

      
 
        las.append(la)
        ls.append(l)
        lls.append(ll)
        lmas.append(lm)
        r_accs.append(r_acc)
        r_lows.append(r_low)
        rs.append(r)
        r_mas.append(r_m)
    
        print("exact Wasserstein distance : ", round(OT,5 ))
        print("loss acc :                   ", round(la,5),    round(abs(la-OT),6))
        print("loss  =                      ", round(l,5),     round(abs(l-OT),6))
        print("loss low :                   ", round(ll,5),    round(abs(ll-OT),6))
        print("richard acc :                ", round(r_acc,5), round(abs(r_acc-OT),6))
        print("richard acc :                ", round(r,5),     round(abs(r-OT),6))
        print("richard low :                ", round(r_low,5), round(abs(r_low-OT),6))


    plt.plot(blurs, [OT]*len(blurs), linestyle = 'solid', c = 'blue',  label = 'OT')
    plt.plot(blurs, las, linestyle = 'solid', c = 'red', label = 'Sink scale = '+str(scale_acc))
    plt.plot(blurs, ls, linestyle = '--', c = 'red',label = 'Sink scale = '+str(scale))
    #plt.plot(blurs, lls, linestyle = 'dotted', c = 'red',label = 'Sink scale = '+str(scale_low))
    plt.plot(blurs, lmas, linestyle = '-.', c = 'red',label = 'Sink scale = '+str(scale_medium))

    plt.plot(blurs, r_accs,linestyle = 'solid', c = 'black', label = 'Rich scale = '+str(scale_acc))
    plt.plot(blurs, rs,linestyle = '--', c = 'black', label =  'Rich scale = '+str(scale))
    #plt.plot(blurs, r_lows,linestyle = 'dotted', c = 'black', label = 'Rich scale = '+str(scale_low))
    plt.plot(blurs, r_mas,linestyle = '-.', c = 'black', label = 'Rich scale = '+str(scale_medium))



    plt.legend()
    plt.savefig("Divergences.png")

def irwin_hall_distribution():
    vdim = 1
    N = 10000

    n = 3

    y =  sum(lb.random.rand(N, vdim) for _ in range(n))

    exact_mean = n/2
    exact_variance = n/12
    exact_2nd_moment = exact_variance + exact_mean**2

    modes = n
    degrees = [1] * modes
    init_ranks = [vdim]+ [n] * (modes-1) # of length modes since 1st component related to the random vector output
    families = [LegendrePolynomials(normalised = True) for i in range(modes)]

    rv = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)

    moments_info = {"mean_param" : 1e7 / N, "2nd_moment_param" : 1e7 / N, "tar_mean" : exact_mean + lb.zeros((1,vdim)),  "tar_2nd_moment": exact_2nd_moment * lb.eye(vdim)}

    rv.sinkhornfit(nxsamples = N, y = y, reg_param_l2 = 1./N**2, reg_param_l1 = 1./N**2, moments_info = moments_info, iterations = 2, verbose = True, verbose_min_surfaces = False)


    print("tar empirical mean : ", lb.mean(y))
    print("tar empirical cov : ", lb.cov(y))

    print("==============================================")
    print(" surr exact mean : ", rv.moments(order = 1))
    print(" surr exact 2nd : ", rv.moments(order = 2))
    print("      exact mean : ", exact_mean)
    print("      exact 2nd : ", exact_2nd_moment)
    print("==============================================")

    print("rv ranks : ", rv.tt.rank)
    rv.tt.round(1e-6)
    print("rv ranks rounded : ", rv.tt.rank)


    #print("tt : ", rv.tt)
    #print("tt full : ", rv.tt.full())

def uncorrelatedTest():
    vdim = 1
    N = 10000
    y =  2*lb.random.rand(N, vdim)-1.

    y = LegendrePolynomials().eval(1, y, all_degrees=False)

     #y = getRandomSamples(N = b, verbose = True) #lb.random.rand(b,vdim)
    modes = 1
    degrees = [1] * modes
    init_ranks = [vdim+1]+ [1] * (modes-1) # of length modes since 1st component related to the random vector output
    families = [LegendrePolynomials(normalised = True) for i in range(modes)]

    rv = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)

    moments_info = {"mean_param" : 1e7, "2nd_moment_param" : 1e7, "tar_mean" : lb.zeros((1,vdim)),  "tar_2nd_moment": lb.eye(vdim)}

    rv.sinkhornfit(nxsamples = N, y = y, reg_param_l2 = 1./N, reg_param_l1 = 0, moments_info = moments_info, iterations = 2, verbose = False, verbose_min_surfaces= True)

    print("tar empirical mean : ", lb.mean(y))
    print("tar empirical cov : ", lb.cov(y))


    print(" surr exact mean : ", rv.moments(order = 1))
    print(" surr exact cov : ", rv.moments(order = 2))

    print("tt : ", rv.tt)
    print("tt full : ", rv.tt.full())


    scales =  [ 2./(2*n+1) * 0.5 for n in range(degrees[0] +1)]
    sinv = [s**(-0.5) for s in scales]
    exact = lb.asarray([0.,sinv[1]**(-1),sinv[1]**(-1),0.])

    #print("exact : ", exact)

def KLEfieldSimulation(modes, vdim = 1):

    N = 10000
    deg = 1
    x =  (2*lb.random.rand(1, vdim)-1.)
    field = lambda y: sum( y[:,m].reshape(N,1) * lb.sin((m+1)*x)  for m in range(modes))   #
    y = 2*lb.random.rand(N, modes) -1.
    
   

    S = field(y)

    if vdim ==1 : 

        SS= S.flatten()

        plt.hist(SS.cpu(), bins = 100, density = True)
        Skde = stats.gaussian_kde(SS.cpu().reshape(SS.shape[0]))
        X = lb.linspace(min(SS.tolist()), max(SS.tolist()), 1000)
        plt.plot(X.cpu(), Skde(X.cpu()), label = 'tar kde', linewidth = 2)
        plt.savefig("hist")

    print("After plot")


    vdim = 1
    #y = getRandomSamples(N = b, verbose = True) #lb.random.rand(b,vdim)
    modes = modes
    degrees = [1] * modes
    init_ranks = [vdim]+ [2] * (modes-1) # of length modes since 1st component related to the random vector output
    families = [LegendrePolynomials(normalised = True) for i in range(modes)]

    rv = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)

    rv.sinkhornfit(nxsamples = N, y = S, reg_param_l2 = 5e-1, reg_param_l1 = 5e-1)

    #print("Init B = ", B)


    print("Reference matrix object : \n", [ 1. for m in range(modes)])   #  lb.sin((m+1)*x).item()
    print("tar mean : ", lb.mean(S).item())
    print("tar cov : ", lb.cov(S).item())

    scales =  [ 2./(2*n+1) * 0.5 for n in range(deg +1)]
    sinv = [s**(-0.5) for s in scales]
    exact = lb.asarray([0.,sinv[1]**(-1),sinv[1]**(-1),0.])

    print("exact : ? ", exact)

def testNon_Centralized_cov():

    vdim = 3
    #y = getRandomSamples(N = b, verbose = True) #lb.random.rand(b,vdim)
    modes = 4
    degrees = [3,4,2,3]
    init_ranks = [2]+ [2,3,4] # of length modes since 1st component related to the random vector output
    families = [LegendrePolynomials(normalised = True) for i in range(modes)]

    rv = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)

    M = rv.moments(2)

    print("M = \n", M)


    mean = rv.moments(1)
    print("mean = \n ", mean)

def randomSampleSumUniform1D(N, modes):

    S = 0
    for k in range(modes):
        S += (2* lb.random.rand(N) -1.)
    
    plt.hist(S, bins = 100, density = True)
    plt.savefig("hist")

    return S.reshape(len(S),1)

def tensorTrainRekonstruktionTest():

    N = 1000
    deg = 1

def lowrankTest():
    N = 10000
    deg = 6
    lr = 3

    A = lb.random.randn(deg+1, deg+1)
    
    u, s, vh = lb.linalg.svd(A)

    B = sum(s[k] * lb.outer(u[:,k], vh[k,:]) for k in range(lr))

    P = 2*lb.random.rand(N) -1
    Q = 2*lb.random.rand(N) -1

    P = LegendrePolynomials().eval(deg, P, all_degrees=True)
    Q = LegendrePolynomials().eval(deg, Q, all_degrees=True)

    S = sum( B[i,j]* P[i]*Q[j] for i in range(deg+1) for j in range(deg+1) )

      #S =  A ** 2 + B** 2 + A*B + A + B
    plt.hist(S.cpu(), bins = 100, density = True)
    Skde = stats.gaussian_kde(S.cpu().reshape(S.shape[0]))
    X = lb.linspace(min(S.tolist()), max(S.tolist()), 1000)
    plt.plot(X.cpu(), Skde(X.cpu()), label = 'tar kde', linewidth = 2)
    plt.savefig("hist")

    S= S.reshape(len(S),1)



    vdim = 1
    #y = getRandomSamples(N = b, verbose = True) #lb.random.rand(b,vdim)
    modes = 2
    degrees = [6] * modes
    init_ranks = [vdim]+ [lr] * (modes-1) # of length modes since 1st component related to the random vector output
    families = [LegendrePolynomials(normalised = True) for i in range(modes)]

    rv = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)

    rv.sinkhornfit(nxsamples = N, y = S, reg_param_l2 = 5e-1, reg_param_l1 = 5e-1)

    #print("Init B = ", B)



    print("tar mean : ", lb.mean(S))
    print("tar cov : ", lb.cov(S))
    

    
   




    exit()

def multimodalTest():
    
    y = randomSamplesMultimodal1D(N = 10000)
    #y = lb.asarray(y)
    #exit()
    b = y.shape[0]    
    #exit()

    vdim = 1
 
    #y = getRandomSamples(N = b, verbose = True) #lb.random.rand(b,vdim)

    modes = 3
    degrees = [6] * modes
    init_ranks = [vdim]+ [2] * (modes-1) # of length modes since 1st component related to the random vector output
    families = [LegendrePolynomials(normalised = True) for i in range(modes)]

    rv = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)

    rv.sinkhornfit(nxsamples = b, y = y, reg_param_l2 = 1e-1, reg_param_l1 = 1e-1)

def globalTestMultimodal(verbose = True):


    y = randomSamplesMultimodal1D(N = 20000)

    deg = 10


    #deg = 10 :
    #     [-0.0027021164  0.6084552415 -0.0084444535 -0.236859112  -0.1639410097  0.0166368822 -0.1144716987  0.0468440044  0.079487244   0.0000409992 -0.0097671376 -1.3420090816 -0.0098301907  0.1279389008 -0.1632767284 -0.1055733987 -0.186021923   0.0989605401  0.0531830705  0.0788033646  0.0826350273  0.1117015644  0.1354204223 -0.5531184795  0.1090725254  0.1517696001 -0.0604981831 -0.2102459997 -0.1116350521  0.1718754894  0.1169955711  0.1915867309  0.0531391373  0.4680473653  0.0568618606 -0.1033273662 -0.1264084257 -0.003062876  -0.1027989605 -0.1774761963  0.0024568521  0.1045313589 -0.0188320324  0.1045535556  0.0229028389  0.387367891   0.0782282032 -0.280293002  -0.0775716838 -0.024706926  -0.0821740978 -0.0453581799  0.0111376579  0.0081625481 -0.1045129267  0.2141659215 -0.0505622216  0.1341404308 -0.0665208682 -0.1136150978  0.0687463886  0.2482939271  0.1558200157 -0.2089765838  0.0127244658 -0.052276976   0.0463378193 -0.3269139189  0.0586709378  0.1321695591  0.1404930191
    #   0.0783362784  0.0638057211  0.1792046729  0.1331991043 -0.177782314   0.0070146815 -0.1069094273 -0.0080765334 -0.3081069923 -0.0392685483  0.2382157155  0.1279367873 -0.1523174647  0.0434348025  0.0037809352 -0.0443994955 -0.0327400129 -0.0456010229 -0.0433560407 -0.0940022381 -0.0913631141  0.108133679  -0.0325624625 -0.1393383904 -0.2024338387 -0.064168798   0.2062999957 -0.0813521179  0.354250158   0.0286746194 -0.0940121965  0.05247893   -0.1349320646 -0.0054800083  0.0716617088 -0.0862044531 -0.1491482833  0.0172841084  0.1937106454  0.0790423484  0.1882268557 -0.036191349   0.1113831071 -0.0866881488  0.0395224266 -0.0826724622  0.1179697227  0.0682349371 -0.1507798017  0.0995787803]

    tar_samples = torch.as_tensor(y, device = "cuda").contiguous()
    s1, i1= torch.sort(tar_samples, dim=0)

    x0 = 2*lb.random.rand((deg+1)**2)-1


    N = len(y)

    P = 2*lb.random.rand(N) -1
    Q = 2*lb.random.rand(N) -1

    P = LegendrePolynomials(normalised= True).eval(deg, P, all_degrees=True)
    Q = LegendrePolynomials(normalised= True).eval(deg, Q, all_degrees=True)

    def samples(x):
        C = x.reshape(deg+1,deg+1)
        S = sum( C[i,j]* P[i]*Q[j] for i in range(deg+1) for j in range(deg+1) )
        S = torch.as_tensor(S, device = "cuda").contiguous()
        return S



    def F(x):
        S = samples(x)
        s2, i2 = torch.sort(S, dim = 0)

        d = s1.squeeze() - s2.squeeze()

        l = lb.linalg.norm(d)**2

        print("\t\t l = ", l.item())
        return l.item()

    
    #res  = minimize(F, x0, method = 'Nelder-Mead', options = {"maxiter" : 20000})
    minimizer_kwargs = {"method": "BFGS"}
    res = basinhopping(F, x0, niter = 50, minimizer_kwargs=minimizer_kwargs)
            #F(x0)
    yfit = samples(res.x)
    print("RES X = ", res.x)


    #yfit = samples(x0)

    ycpu = y.cpu()
    ycpu = ycpu.reshape(ycpu.shape[0])
    yfitcpu = yfit.cpu()
    yfitcpu = yfitcpu.reshape(yfitcpu.shape[0])


    if verbose: 
        #plt.subplot(1,iterations+1, niter + 2)
        plt.hist(ycpu, bins = 50, alpha = 0.5, label = 'target', density = True)
        plt.hist(yfitcpu, bins = 50, alpha = 0.5,  label = 'fit', density = True)

        ykde = stats.gaussian_kde(ycpu)
        yfitkde = stats.gaussian_kde(yfitcpu)

        X = lb.linspace(min(ycpu.tolist()), max(ycpu.tolist()), 1000)
        plt.plot(X, ykde(X), label = 'tar kde', linewidth = 2)         

        X = lb.linspace(min(yfitcpu.tolist()), max(yfitcpu.tolist()), 1000)
        plt.plot(X, yfitkde(X), label = 'fit kde', linewidth = 2)           


        plt.legend()

    plt.savefig("globalTest")

def globalTestSumUniform(y, verbose = True):
    deg = 1


    scales =  [ 2./(2*n+1) * 0.5 for n in range(deg +1)]

    sinv = [s**(-0.5) for s in scales]

    tar_samples = torch.as_tensor(y, device = "cuda").contiguous()
    s1, i1= torch.sort(tar_samples, dim=0)

    x0exact = lb.asarray([0.,sinv[1]**(-1),sinv[1]**(-1),0.])

    #x0 =  lb.asarray([0.,sinv[1]**(-1),sinv[1]**(-1),0.])

    x0 = 2*lb.random.rand(4)-1


    N = len(y)

    P = 2*lb.random.rand(N) -1
    Q = 2*lb.random.rand(N) -1

    deg = 1

    P = LegendrePolynomials(normalised= True).eval(deg, P, all_degrees=True)
    Q = LegendrePolynomials(normalised= True).eval(deg, Q, all_degrees=True)

    def samples(x):
        C = x.reshape(deg+1,deg+1)
        S = sum( C[i,j]* P[i]*Q[j] for i in range(deg+1) for j in range(deg+1) )
        S = torch.as_tensor(S, device = "cuda").contiguous()
        return S



    def F(x):
        S = samples(x)
        s2, i2 = torch.sort(S, dim = 0)

        d = s1.squeeze() - s2.squeeze()

        l = lb.linalg.norm(d)**2

        print("\t\t l = ", l.item())
        return l.item()


    res  = minimize(F, x0, method = 'Nelder-Mead', options = {"maxiter" : 1000})
            #F(x0)
    yfit = samples(res.x)
    print("RES X = ", res.x)

    print("F ( exact ) = ", F(x0exact))

    print("exact = ", x0exact)


    #yfit = samples(x0)

    ycpu = y.cpu()
    ycpu = ycpu.reshape(ycpu.shape[0])
    yfitcpu = yfit.cpu()
    yfitcpu = yfitcpu.reshape(yfitcpu.shape[0])


    if verbose: 
        #plt.subplot(1,iterations+1, niter + 2)
        plt.hist(ycpu, bins = 50, alpha = 0.5, label = 'target', density = True)
        plt.hist(yfitcpu, bins = 50, alpha = 0.5,  label = 'fit', density = True)

        ykde = stats.gaussian_kde(ycpu)
        yfitkde = stats.gaussian_kde(yfitcpu)

        X = lb.linspace(min(ycpu.tolist()), max(ycpu.tolist()), 1000)
        plt.plot(X, ykde(X), label = 'tar kde', linewidth = 2)         

        X = lb.linspace(min(yfitcpu.tolist()), max(yfitcpu.tolist()), 1000)
        plt.plot(X, yfitkde(X), label = 'fit kde', linewidth = 2)           


        plt.legend()

    plt.savefig("globalTest")

def deleteme():
                print("Core shape = : ",  self.tt.comps[mu].shape)
                print("\t\t\t\t\t vdim = ", y.shape[1])
                print("\t\t\t\t\t b    = ", y.shape[0])
                #x0 = self.tt.comps[mu].reshape(s[0]*s[1]*s[2])
                x0 = self.tt.comps[mu].reshape(s[1], s[2]).T


                # R * x0.T    

                print("as matrix : ", x0.shape)

                res1 = R @ x0
                
                print("res1 shape = ", res1.shape)

                res2 = lb.kron(lb.eye(s[1]), R) @ self.tt.comps[mu].flatten()

                res2test = res2.reshape(y.shape[0], y.shape[1])

                print("res 1 = \n", res1)
                print("res 2 = \n",  res2test)




                print("diff 1 : ", 0.5*lb.linalg.norm(res1-y)**2)
                print("diff 2 : ", 0.5*lb.linalg.norm(res2 - y.flatten())**2)
                print("diff 3 : ", 0.5* lb.linalg.norm(res2 - y.T.flatten())**2)

def testVectorialFitting():

    modes = 4

    # sin ( x)  = (e^(ix) - e^{-ix}) / 2i   in the complex field, matrix rank is invariant of field choise, thus f has FTT rank 2 
    f = lambda m : lambda x : lb.sin(sum((m%2 + 1) * x[:,i]**2 for i in range(modes)) )
        
    vdim = 6
    degrees = [10] * modes
    ranks   = [vdim] + [4] * (modes-1) # of length modes since 1st component related to the random vector output
    families = [ChebyshevT(normalised = True) for i in range(modes)]  #ChebyshevT LegendrePolynomials
    surr = Vectorial_Extended_TensorTrain(vdim, families, degrees, ranks)

    # create data points
    N= 3*modes*max(ranks)**2*degrees[0] + 2*vdim
    x = lb.random.rand(N,modes)*2 -1
    y = lb.stack([f(m+1)(x) for m in range(vdim)], axis = 1)
    y = lb.asarray(y)

    surr.fit(x = x, y = y, iterations = 30)

    print("train seterror: ", lb.linalg.norm(surr(x) - y)**2)


    print("x shape = ", x.shape)
    print("y.shape = ", y.shape)

    # validation set 
    x = lb.random.rand(N,modes)*2 -1
    y = lb.stack([f(m+1)(x) for m in range(vdim)], axis = 1)

    print("validation set error: ", lb.linalg.norm(surr(x) - y)**2)

    print("ranks = ", surr.tt.rank)
    surr.tt.round(1e-14)
    print("rounded ranks= ", surr.tt.rank)

    
    print("validation set error: ", lb.linalg.norm(surr(x) - y)**2)

    exit()


    discr = 20
    X = lb.linspace(min(x[:,0]), max(x[:,0]), discr)
    Y = lb.linspace(min(x[:,1]), max(x[:,1]), discr)

    X,Y = lb.meshgrid(X,Y)
    Z = lb.zeros(X.shape)

    for i in range(discr):
        for j in range(discr):
            XX =  lb.tensor([X[i,j],Y[i,j] ]).reshape(1,2)
            Z[i,j] = surr(XX)[0,0]

    plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(X,Y,Z)
    plt.savefig("plot_surface_TT.png")

    exit()

def TUProblem():
    from mpl_toolkits.mplot3d import Axes3D

    import os
    cwd = os.getcwd()

    from scipy.io import loadmat

    # Outs_n2_q5_ann1000_m0.mat   # ellipsen 1 parameter   2. spalte 0nullen 3. nan         [0.05, 20]  
    # Outs_n2_q5_ann1000_m6.mat   # arbitrary case mit 5 parametern       random samples    [0.1, 10]    1024  = 32 x 32 
    # Outs_n2_q5_cheb10_m0.mat


    data = loadmat(cwd+'/data/Outs_n2_q5/Outs_n2_q5_ann1000_m6.mat')["a"]

    print(data.shape)

    x = lb.tensor(data[:,:5])
    Y = lb.tensor(data[:, 6:])

    print(Y.shape)
    errors = {}
   

    y = Y[:,3:4]

    print(y.shape)

    
    #y = y.reshape(len(y),1)

    #exit()

    vdim = y.shape[1]
    modes = x.shape[1]
    degrees = [5] * modes
    init_ranks = [vdim+1] + [5] * (modes-1) # of length modes since 1st component related to the random vector output
    families = [ChebyshevT(a= 0.1, b= 10, normalised = True) for i in range(modes)]

    surr = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)



    surr.fit(x = x, y = y, iterations = 200, reg_param_order = 1e-2, reg_decay_rate = 0.5, reg_decay_rhythm = 20)

    print("data error: ", lb.linalg.norm(surr(x) - y)**2)
    #errors[k] = (lb.linalg.norm(surr(x) - y)**2).item()


    if False:
        for item in errors.items():
            key, val = item
            print("{k} -->  {v}".format(k=key,v=val))


        import collections
        od = collections.OrderedDict(sorted(errors.items()))

        skeys = [k for k, _ in od.items()]
        svals = [v for _, v in od.items()]

        print(skeys)
        print(type(skeys[0]))
        print(svals)
        print(type(svals[0]))
        

        plt.scatter(skeys,svals)
        ax = plt.gca()
        #ax.set_yscale('log')
        plt.savefig("Errors")



    exit()
    
    
    
    data = lb.asarray(data)

    print("data shape = ", data.shape)

    print(data)

    exit()

    mask = ~torch.any(data.isnan(), dim = 1)

    data_filtered = data[mask]

    print("data filtered ", data_filtered.shape)


    #k =  

    x = lb.asarray(data_filtered[:,:2])
    y = lb.asarray(data_filtered[:, 2:])

    y = y[:, 1:4]#.reshape(y.shape[0],1)

    print("x.shape = ", x.shape)
    print("y.shape = ", y.shape)




 
    
    vdim = y.shape[1]
    modes = 2
    degrees = [10] * modes
    init_ranks = [5] * modes # of length modes since 1st component related to the random vector output
    families = [ChebyshevT(normalised = True) for i in range(modes)]

    surr = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)



    surr.fit(x = x, y = y, iterations = 40)

    print("data error: ", lb.linalg.norm(surr(x) - y)**2)

    exit()


    #for idx, k in enumerate([2,1000]):
    #
    ax = plt.gca(projection='3d')
    ax.plot_trisurf(x[:,0],x[:,1],y[:, 0])
    plt.savefig("trisurf.png")


    discr = 20
    X = lb.linspace(min(x[:,0]), max(x[:,0]), discr)
    Y = lb.linspace(min(x[:,1]), max(x[:,1]), discr)

    X,Y = lb.meshgrid(X,Y)

    Z = lb.zeros(X.shape)

    for i in range(discr):
        for j in range(discr):
            XX =  lb.tensor([X[i,j],Y[i,j] ]).reshape(1,2)
            mat = surr(XX)

            Z[i,j] = surr(XX)[0,0]

    plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot_surface(X,Y,Z)
    plt.savefig("plot_surface_TT.png")


    
    exit()  


def fuzzyboundary():

    full_name = 'data/elasticity_hr80_lc0.1_p2_dimdom6.pkl'

    df = pd.read_pickle(full_name) 


    print(df)

   

    #exit()

    #exit()


    x = np.vstack(np.array(df["param"])).astype(np.float)

    y = np.vstack(np.array(df["av_sigy"])).astype(np.float)   # "energy" average_displacement sigma1_max av_sigy thole_area
 
    x = lb.tensor(x)
    y = lb.tensor(y)




    vdim = y.shape[1]
    modes = x.shape[1]
    degrees = [10] * modes
    init_ranks = [vdim] + [10] * (modes-1) # of length modes since 1st component related to the random vector output
    families = [ChebyshevT(a= 0.3, b= 0.7, normalised = True) for i in range(modes)]

    surr = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)



    surr.fit(x = x, y = y, iterations = 400)

    print("data error: ", lb.linalg.norm(surr(x) - y)**2)

    exit()




def randomSamplesMultimodal1D(N):

    A = lb.random.randn(N) - 2
    B = 0.5*lb.random.randn(N) + 1

    S = lb.concatenate([A,B])

    plt.hist(S.cpu(), bins = 100, density = True)
    plt.savefig("hist")

    #S = lb.asarray(S)
    return S.reshape(len(S),1)

def testCuda():

    x = lb.random.rand(1000,2).to(device = "cuda").contiguous()

    vdim = 1
 
    #y = getRandomSamples(N = b, verbose = True) #lb.random.rand(b,vdim)

    modes = 2
    degrees = [1] * modes
    init_ranks = [2] * modes # of length modes since 1st component related to the random vector output
    families = [LegendrePolynomials(normalised = True) for i in range(modes)]

    rv = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)

    print(rv(x))

def multidimensionalTest():

    vdim = 2
    N = 10000

    n = 3

    S =  sum(lb.random.rand(N, vdim) for _ in range(n))

    print("S.shape = ", S.shape)

    xedges = [0, 1, 3, 5]
    yedges = [0, 2, 3, 4, 6]

    x = np.random.normal(2, 1, 100)
    y = np.random.normal(1, 1, 100)

    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))

    # Histogram does not follow Cartesian convention (see Notes),

    # therefore transpose H for visualization purposes.

    H = H.T


    fig = plt.figure(figsize=(7, 3))

    ax = fig.add_subplot(131, title='imshow: square bins')

    plt.imshow(H, interpolation='nearest', origin='lower',

            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    plt.savefig("target")




def main():

    #fuzzyboundary()

    #exit()



    TUProblem()


    exit()

    multidimensionalTest()


    exit()

    #testSinkhorn()

    #exit()

    irwin_hall_distribution()


    exit()


    uncorrelatedTest()

    exit()

    testNon_Centralized_cov()

    exit()

    #KLEfieldSimulation(modes = 5)


    lowrankTest()



    exit()
    multimodalTest()


    exit()

    testVectorialFitting()

    exit()



    #testCuda()


    #exit()

    TUProblem()

    exit()

    #globalTestMultimodal()

    #exit()

    #sshfs gruhlke@gate.wias-berlin.de:/Home/optimier/gruhlke/numerics cluster/
    

    #y, C = randomSamples1DRandomPolynomial(N=10000)

    #y = randomSamples1D(N=10000)

    y = randomSampleSumUniform1D(N=10000, modes = 2)
    deg = 10


    #deg = 10 :
    #     [-0.0027021164  0.6084552415 -0.0084444535 -0.236859112  -0.1639410097  0.0166368822 -0.1144716987  0.0468440044  0.079487244   0.0000409992 -0.0097671376 -1.3420090816 -0.0098301907  0.1279389008 -0.1632767284 -0.1055733987 -0.186021923   0.0989605401  0.0531830705  0.0788033646  0.0826350273  0.1117015644  0.1354204223 -0.5531184795  0.1090725254  0.1517696001 -0.0604981831 -0.2102459997 -0.1116350521  0.1718754894  0.1169955711  0.1915867309  0.0531391373  0.4680473653  0.0568618606 -0.1033273662 -0.1264084257 -0.003062876  -0.1027989605 -0.1774761963  0.0024568521  0.1045313589 -0.0188320324  0.1045535556  0.0229028389  0.387367891   0.0782282032 -0.280293002  -0.0775716838 -0.024706926  -0.0821740978 -0.0453581799  0.0111376579  0.0081625481 -0.1045129267  0.2141659215 -0.0505622216  0.1341404308 -0.0665208682 -0.1136150978  0.0687463886  0.2482939271  0.1558200157 -0.2089765838  0.0127244658 -0.052276976   0.0463378193 -0.3269139189  0.0586709378  0.1321695591  0.1404930191
    #   0.0783362784  0.0638057211  0.1792046729  0.1331991043 -0.177782314   0.0070146815 -0.1069094273 -0.0080765334 -0.3081069923 -0.0392685483  0.2382157155  0.1279367873 -0.1523174647  0.0434348025  0.0037809352 -0.0443994955 -0.0327400129 -0.0456010229 -0.0433560407 -0.0940022381 -0.0913631141  0.108133679  -0.0325624625 -0.1393383904 -0.2024338387 -0.064168798   0.2062999957 -0.0813521179  0.354250158   0.0286746194 -0.0940121965  0.05247893   -0.1349320646 -0.0054800083  0.0716617088 -0.0862044531 -0.1491482833  0.0172841084  0.1937106454  0.0790423484  0.1882268557 -0.036191349   0.1113831071 -0.0866881488  0.0395224266 -0.0826724622  0.1179697227  0.0682349371 -0.1507798017  0.0995787803]

    y = torch.as_tensor(y, device = "cuda").contiguous()


    #globalTest(y)

    #exit()
    

    print("y shape = ", y.shape)

    b = y.shape[0]    
    #exit()

    vdim = 1
 
    #y = getRandomSamples(N = b, verbose = True) #lb.random.rand(b,vdim)

    modes = 2
    degrees = [1] * modes
    init_ranks = [2] * modes # of length modes since 1st component related to the random vector output
    families = [LegendrePolynomials(normalised = True) for i in range(modes)]

    rv = Vectorial_Extended_TensorTrain(vdim, families, degrees, init_ranks)

    rv.sinkhornfit(nxsamples = b, y = y)

    #print("reference Coeffs : \n ", C)


    exit()

    use_cuda =  torch.cuda.is_available()

    if use_cuda :
        with TicToc(key=" o Cuda prep (empty cache)", do_print=False, accumulate=True, sec_key="Torch:"):
            torch.cuda.empty_cache()
        with TicToc(key=" o Cuda prep (sync)", do_print=False, accumulate=True, sec_key="Torch:"):
            torch.cuda.synchronize()



    exit()

    Nsamples = 1000

    family = LegendrePolynomials



    TTRV = TT_PCE_RV(dim = 2, family = family, modes = 2, degrees = [4,3], ranks = [1,2, 3,1])

    for k in range(5):

        if True : 
            with TicToc(key=" o generate Samples", do_print=False, accumulate=True, sec_key="Samples:"):
                pce_samples = TTRV.samples(N = Nsamples)
                #print(pce_samples.shape)
                tar_samples = getRandomSamples(N=Nsamples)
                #print(tar_samples.shape)


            if True and use_cuda : 
                with TicToc(key=" o Move Samples to Cuda", do_print=False, accumulate=True, sec_key="Samples:"):
                    # TODO TICTOC Here to catch this timing
                    tar_samples = torch.as_tensor(tar_samples, device = "cuda").contiguous()#, requires_grad=True)
                    print(tar_samples.shape)
                    pce_samples = torch.as_tensor(pce_samples.to(device = "cuda")).contiguous() #torch.Tensor(pce_samples, device = torch.device("cuda"))
                    print(pce_samples.shape)

            else:
                tar_samples = torch.as_tensor(tar_samples)
        else:

             tar_samples = torch.randn(1000, 3).cuda()
             pce_samples = torch.randn(Nsamples, 3).cuda()

             print(tar_samples.shape)



        # Define a Sinkhorn (~Wasserstein) loss between sampled measures
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
        with TicToc(key=" o Sinkhorn computation", do_print=False, accumulate=True, sec_key="Samples:"):
            L = loss(tar_samples, pce_samples)  # By default, use constant weights = 1/number of samples
        
        #g_x, = torch.autograd.grad(L, [tar_samples])  # GeomLoss fully supports autograd!

        print("Wasserstein distance: {:.3f}".format(L.item()))
    
    TicToc.sortedTimes()


    exit()


    # Create some large point clouds in 3D
    x = torch.randn(1000, 3, requires_grad=True).cuda()
    y = torch.randn(2000, 3).cuda()

   

    







if __name__ == "__main__":
    main()
# %%
