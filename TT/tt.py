from types import LambdaType

import psutil
from colorama import Fore, Style
# from numpy import core
from numpy.core.numeric import full
import copy

from scipy.linalg import null_space
from TT import tensor
import numpy as np
from numpy.polynomial.legendre import Legendre

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pympler.tracker import SummaryTracker
from itertools import product

from TT.tictoc import TicToc

from TT.feature_utils import orthpoly_basis

# TODO: try import mechanic
# import pytorch as lb

# import numpy as lb # linear backend

# from linearbackend import Linear_Backend
# backend_options = {  "backend": "numpy"/"torch",
#                     "device" : "cpu"/"cuda"}

# lb = Linear_Backend(backend_options = {"backend" : "torch", "device" : "cuda"})

from TT.linearbackend import lb

# tensor utilities
import TT.tensor


class Rule(object):
    def __eval__(self, sigma, pos):
        pass


# TODO: dependence on u and v really necessary?
class Threshold(Rule):
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, u, sigma, v, pos):
        return lb.max([lb.sum(sigma > self.delta), 1])


class Truncate(Rule):
    def __init__(self, ranks):
        self.rank = ranks

    def __call__(self, u, sigma, v, pos):
        return self.rank[pos]


class Dörfler_Adaptivity(Rule):
    """
        Adaptivty rank rule :

                - Dörfler condition is fullfilled if there exists L s.t. :

                        delta * (sum k=0^L  sigma[k] )  >=  sum_{k=L+1} sigma[k]

                - ranks have an upper bound
                - if Dörfler condition holds then  cutoff all singularvalues with index > L+1.
                  In particular keep sigma[L+1] as a treshhold sing. val. keeping track of a max rank needed.
                  It can be rounded later.

                - if  the dörfler condition is not fullfilled for any L in [0,...,len(sigma)-1], then
                  the new rank is increased or stays the same, i.e.

                            new rank = min { maxrank, oldrank + rankincr* }

                  here
                      rankincr*  = min (rankincr,  max possible rank increase)

                  with
                        max possible rank increase =  difference of shapes of v

    """

    def __init__(self, delta, maxranks, dims, rankincr=2, verbose=True):
        self.delta = delta
        self.rankincr = rankincr

        self.verbose = verbose

        self.maxranks = [0.] * len(maxranks)

        for k in range(len(maxranks)):
            urank = min(np.prod(dims[:k + 1]), np.prod(dims[k + 1:]))  # upper bound of rank due to the unfolding

            if maxranks[k] > urank:
                print("Warning upper limit of rank exceeded ({a} > {b}). Choose upper limit as max rank.".format(
                    a=maxranks[k], b=urank))

                self.maxranks[k] = urank
            else:
                self.maxranks[k] = maxranks[k]

    def __call__(self, u, sigma, v, pos):

        # vmax = abs(v.shape[0]-v.shape[1])
        umax = abs(max(u.shape[0], u.shape[1]) - len(sigma))

        # if self.verbose:
        #    for k in range(len(sigma)):
        #        l = lb.sum(sigma[:k])
        #        r = lb.sum(sigma[k:])
        #        print("{k} :  l = {l}   {r} = r".format(k=k,l=l,r=r))

        for k in range(len(sigma)):
            l = lb.sum(sigma[:k])
            r = lb.sum(sigma[k:])

            if self.delta * l >= r:
                if self.verbose:
                    print("Dörfer rank recommendation: {c1} old rank = {r1} {r} -> {c2}new rank = {r2}{r}".format(
                        r=Style.RESET_ALL, c1=Fore.RED, c2=Fore.GREEN, r1=len(sigma), r2=k))
                return k  # + min(umax,self.rankincr)

        # else a rank increasing is performed

        newrank = min(self.maxranks[pos], len(sigma) + min(umax, self.rankincr))

        if self.verbose:
            if len(sigma) == newrank:
                print("Dörfer rank recommendation: {c} old rank = new rank = {rank}{r}".format(r=Style.RESET_ALL,
                                                                                               c=Fore.RED,
                                                                                               rank=len(sigma)))
            else:
                print("Dörfer rank recommendation: {c1} old rank = {r1} {r} -> {c2}new rank = {r2}{r}".format(
                    r=Style.RESET_ALL, c1=Fore.RED, c2=Fore.GREEN, r1=len(sigma), r2=newrank))

        return newrank


# TODO:
class Rank_One_Kick(Rule):
    pass


# TODO @ David:
#      Implement a round routine after fill random / add  / or more ?? (can be done by retract with uranks)
#      Which a very small threshold that respects the maximal allowed rank
class TensorTrain(object):
    def __init__(self, dims, comp_list=None):

        self.n_comps = len(dims)
        self.dims = dims
        self.comps = [None] * self.n_comps

        self.rank = None
        self.core_position = None

        # upper bound for ranks
        self.uranks = [1] + [min(np.prod(dims[:k + 1]), np.prod(dims[k + 1:])) for k in range(len(dims) - 1)] + [1]

        if comp_list is not None:
            self.set_components(comp_list)

    def set_components(self, comp_list):
        """
           @param comp_list: List of order 3 tensors representing the component tensors
                            = [C1, ..., Cd] with shape
                            Ci.shape = (ri, self.dims[i], ri+1)

                            with convention r0 = rd = 1

        """
        # the length of the component list has to match
        assert (len(comp_list) == self.n_comps)

        # each component must be a order 3 tensor object
        for pos in range(self.n_comps):
            assert (len(comp_list[pos].shape) == 3)

        # the given components inner dimension must match the predefined fixed dimensions
        for pos in range(self.n_comps):
            assert (comp_list[pos].shape[1] == self.dims[pos])

        # neibourhood communication via rank size must match
        for pos in range(self.n_comps - 1):
            assert (comp_list[pos].shape[2] == comp_list[pos + 1].shape[0])

        # setting the components
        for pos in range(self.n_comps):
            self.comps[pos] = copy.copy(comp_list[pos])

    def fill_random(self, ranks):
        """
            Fills the TensorTrain with random elements for a given structure of ranks.
            If entries in the TensorTrain object have been setted priviously, they are overwritten
            regardless of the existing rank structure.

            @param ranks #type list
        """
        self.__check_ranks(ranks)

        for pos in range(self.n_comps):
            self.comps[pos] = lb.random.rand(self.rank[pos], self.dims[pos], self.rank[pos + 1])

        # # truncate to maximal rank
        # self.retract(self.uranks[1:-1])

    def __shift_to_right(self, pos, variant):
        with TicToc(key=" o right shifts", do_print=False, accumulate=True, sec_key="Core Moves:"):
            c = self.comps[pos]
            s = c.shape
            c = tensor.left_unfolding(c)
            if variant == 'qr':
                q, r = lb.linalg.qr(c)
                self.comps[pos] = q.reshape(s[0], s[1], q.shape[1])
                self.comps[pos + 1] = lb.einsum('ij, jkl->ikl ', r, self.comps[pos + 1])
            else:  # variant == 'svd'
                u, S, vh = lb.linalg.svd(c, full_matrices=False)
                # print("c.shape", c.shape)
                # print("u.shape", u.shape)
                # print("S.shape", S.shape)
                # print("vh.shape", vh.shape)

                u, S, vh = u[:, :len(S)], S[:len(S)], vh[:len(S), :]

                # store orthonormal part at current position
                self.comps[pos] = u.reshape(s[0], s[1], u.shape[1])
                self.comps[pos + 1] = lb.einsum('ij, jkl->ikl ', lb.diag(S) @ vh, self.comps[pos + 1])

            # update the according th rank
            # self.rank[pos + 1] = self.comps[pos].shape[2]

    def __shift_to_left(self, pos, variant):
        with TicToc(key=" o left shifts", do_print=False, accumulate=True, sec_key="Core Moves:"):
            c = self.comps[pos]

            s = c.shape
            c = tensor.right_unfolding(c)
            if variant == 'qr':
                q, r = lb.linalg.qr(lb.transpose(c, 1, 0))
                qT = lb.transpose(q, 1, 0)
                self.comps[pos] = qT.reshape(qT.shape[0], s[1], s[2])  # refolding
                self.comps[pos - 1] = lb.einsum('ijk, kl->ijl ', self.comps[pos - 1], lb.transpose(r, 1, 0))

            else:  # perform svd
                u, S, vh = lb.linalg.svd(c, full_matrices=False)
                # store orthonormal part at current position
                self.comps[pos] = vh.reshape(vh.shape[0], s[1], s[2])
                self.comps[pos - 1] = lb.einsum('ijk, kl->ijl ', self.comps[pos - 1], u @ lb.diag(S))

            # update the according th rank
            # self.rank[pos] = self.comps[pos].shape[0]

    def set_core(self, mu, variant='qr'):
        """
        # TODO
        """

        cc = []  # changes components

        if self.core_position is None:
            assert (variant in ['qr', 'svd'])
            self.core_position = mu
            # from left to right shift of the non-orthogonal component
            for pos in range(0, mu):
                self.__shift_to_right(pos, variant)
            # right to left shift of the non-orthogonal component
            for pos in range(self.n_comps - 1, mu, -1):
                self.__shift_to_left(pos, variant)
            # self.rank[mu+1] = self.comps[mu].shape[2]

            cc = list(range(self.n_comps))

        else:
            while self.core_position > mu:
                cc.append(self.core_position)
                self.shift_core('left')
            while self.core_position < mu:
                cc.append(self.core_position)
                self.shift_core('right')

            cc.append(mu)

        assert (self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)

        self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps)]
        return cc

    def shift_core(self, direction, variant='qr'):
        assert (direction in [-1, 1, 'left', 'right'])
        assert (self.core_position is not None)

        if direction == 'left':
            shift = -1
        elif direction == 'right':
            shift = 1
        else:
            shift = direction
        # current core position
        mu = self.core_position
        if shift == 1:
            self.__shift_to_right(mu, variant)
        else:
            self.__shift_to_left(mu, variant)

        self.core_position += shift

    def dot_rank_one(self, rank1obj):
        """
          Implements the multidimensional contraction of the underlying Tensor Train object
          with a rank 1 object being product of vectors of sizes di
          @param rank1obj: a list of vectors [vi i = 0, ..., modes-1] with len(vi)=di
                           vi is of shape (b,di) with bi > 0
        """
        with TicToc(key=" o dot rank one ", do_print=False, accumulate=True, sec_key="TT application:"):
            # the number of vectors must match the component number
            assert (len(rank1obj) == self.n_comps)
            for pos in range(0, self.n_comps):
                # match of inner dimension with respective vector size
                assert (self.comps[pos].shape[1] == rank1obj[pos].shape[1])
                # vectors must be 2d objects
                assert (len(rank1obj[pos].shape) == 2)

            G = [lb.einsum('ijk, bj->ibk', c, v) for c, v in zip(self.comps, rank1obj)]
            # print(G)
            res = G[-1]
            # contract from right to left # TODO here we assume row-wise memory allocation of matrices in G
            for pos in range(self.n_comps - 2, -1, -1):
                # contraction w.r.t. the 3d coordinate of G[pos]
                # res = lb.dot(G[pos], res)
                res = lb.einsum('ibj, jbk -> ibk', G[pos], res)  # k = 1 only
            # res is of shape b x 1
            return res.reshape(res.shape[1], res.shape[2])

    def contract_2nd_to_end_rank_one(self, rank1obj):
        with TicToc(key=" o contract_2nd_to_end_rank_one ", do_print=False, accumulate=True, sec_key="TT application:"):
            # the number of vectors must match the component number
            assert (len(rank1obj) == self.n_comps - 1)
            for pos in range(1, self.n_comps):
                # match of inner dimension with respective vector size
                print(self.comps[pos].shape[1])
                print(rank1obj[pos - 1].shape[1])
                assert (self.comps[pos].shape[1] == rank1obj[pos - 1].shape[1])
                # vectors must be 2d objects
                assert (len(rank1obj[pos - 1].shape) == 2)

            G = [lb.einsum('ijk, bj->ibk', c, v) for c, v in zip(self.comps[1:], rank1obj)]
            # print(G)
            res = G[-1]
            # contract from right to left # TODO here we assume row-wise memory allocation of matrices in G
            for pos in range(self.n_comps - 3, -1, -1):
                # contraction w.r.t. the 3d coordinate of G[pos]
                # res = lb.dot(G[pos], res)
                res = lb.einsum('ibj, jbk -> ibk', G[pos], res)  # k = 1 only

            res = lb.einsum('idj,jbk->ibdk', self.comps[0], res)  # i==k==1

            return res.reshape(res.shape[1], res.shape[2])

    def full(self):
        """
            Obtain the underlying full tensor.

            WARNING: This can become abitrary slow and may exceed memory.
        """
        res = lb.zeros((self.dims))
        for idx in product(*[list(range(d)) for d in self.dims]):
            val = lb.asarray(1)
            for k in range(self.n_comps):
                c, s = self.comps[k], self.comps[k].shape
                val = lb.dot(val, c[:, idx[k], :].reshape(s[0], s[2]))
            res[idx] = val

        return res

    # TODO: shouldn't this be called __set_r/anks?

    def __check_ranks(self, rank):
        if len(rank) == self.n_comps + 1:
            assert (rank[0] == 1 and rank[-1] == 1)
            self.rank = rank
        elif len(rank) == self.n_comps - 1:
            self.rank = [1] + rank + [1]
        elif isinstance(rank, int):
            self.rank = [1] + [rank] * (self.n_comps - 1) + [1]
        else:
            raise InputError("Parameter rank not in correct form.")

        # for pos, r in enumerate(self.rank):
        #    if r > self.uranks[pos]:
        ##        print("Note: Maximal rank exceeded. Fall back to max rank.")
        #       self.rank[pos] = self.uranks[pos]

    def round(self, epsilon, verbose=False):
        # TODO : delta = eps / sqrt(d - 1) * || self.full() ||
        delta = epsilon
        self.modify_ranks(Threshold(delta), verbose)

    def retract(self, ranks, verbose=False):
        self.__check_ranks(ranks)
        self.modify_ranks(Truncate(ranks), verbose)

    def modify_ranks(self, rule, verbose=False):
        with TicToc(key=" o modify ranks ", do_print=False, accumulate=True, sec_key="TT modification:"):
            # TODO handle the case if core is at last position
            if self.core_position != 0:
                self.set_core(0)

            # Possible modify ranks r2, ..., rM-2
            for pos in range(self.n_comps - 1):
                c = self.comps[pos]
                s = c.shape
                c = c.reshape(s[0] * s[1], s[2])

                u, sigma, v = lb.linalg.svd(c, full_matrices=False)
                # obtain the possible new rank according to truncation/retraction rule
                new_rank = rule(u, sigma, v, pos)
                if verbose:
                    print("{c}Update{r} : rank r{p} = {c1}{rank}{r} -> {c}{rankn}{r}".format(p=pos + 1,
                                                                                             rank=self.rank[pos + 1],
                                                                                             rankn=new_rank,
                                                                                             c1=Fore.RED, c=Fore.GREEN,
                                                                                             r=Style.RESET_ALL))
                    print("sing. vals for r{p} :\n".format(p=pos + 1), sigma)

                if new_rank > len(sigma):
                    # print("C[pos ].shape  = ", self.comps[pos ].shape)
                    # print("C[pos +1].shape  = ",self.comps[pos +1 ].shape)

                    #   u, sigma, v  = svd ( c )  with c = self.comps[pos]
                    k = new_rank - len(sigma)

                    # 1. Add  k new columns to the left unfolding of u :
                    #   - leftunfold(u)  is  M x r matrix
                    #   - add  k orthogonal columns called u_k to u to obtain  upk of shape M x ( r  + k )
                    #   - undo the left unfolding w.r.t. M  and store  self.comps[pos] = upk
                    # "add" random vectors from kernel of u^T as orthogonal projection of a random vectors
                    u_k = lb.random.rand(u.shape[0], k)
                    u_k -= (u @ u.T) @ u_k

                    # enlarged u plus k columns
                    u_pk = lb.concatenate([u, u_k], axis=1)
                    self.comps[pos] = u_pk.reshape(s[0], s[1], u_pk.shape[1])

                    # 2. Enlarge the singular values s with k new very small entries.
                    s_pk = lb.concatenate([sigma, lb.array([1e-16] * k)])

                    # 3.  K = v * self.comps[pos+1]    w.r.t. 3rd  and right unfolding
                    #      yields a   r x N orthogonal matrix. Add k orthgonal rows K_k
                    #     to obtain a (r+k) x N orthogonal matrix K_kp = []
                    K = lb.einsum('ir, rkl->ikl ', v, self.comps[pos + 1])
                    s = K.shape
                    K = K.reshape(s[0], s[1] * s[2])

                    assert (lb.abs(K.shape[0] - K.shape[1]) >= k)
                    # get randomized orthogonal rows
                    K_k = lb.random.rand(k, K.shape[1])
                    K_k -= K_k @ (K.T @ K)
                    K_pk = lb.concatenate([K, K_k])

                    # 4. Then undo the unfolding of   K_pk  and scale it with the enlarged sing. values
                    #    to define the new right component self.comps[pos+1]
                    K_pk = K_pk.reshape(K_pk.shape[0], s[1], s[2])
                    self.comps[pos + 1] = lb.einsum('ij,jkl->ikl', lb.diag(s_pk), K_pk)

                else:
                    # update informations
                    u, sigma, v = u[:, :new_rank], sigma[:new_rank], v[:new_rank, :]

                    new_shape = (s[0], s[1], new_rank)
                    self.comps[pos] = u.reshape(new_shape)

                    self.comps[pos + 1] = lb.einsum('ir, rkl->ikl ', lb.dot(lb.diag(sigma), v), self.comps[pos + 1])

                    # update the rank information
                # self.rank[pos+1] = self.comps[pos].shape[2]

            self.core_position = self.n_comps - 1
            assert (self.comps[0].shape[0] == 1 and self.comps[-1].shape[2] == 1)
            self.rank = [1] + [self.comps[pos].shape[2] for pos in range(self.n_comps)]

    @staticmethod
    def hadamard_product(A, B):
        """ Computes <A,B> = AoB  with o beeing the hadamard product"""

        assert (lb.equal(A.dims, B.dims).all())
        # assert( lb.equal(A.ranks, B.ranks).all() )
        n_comps = len(A.dims)
        d = A.dims[0]
        v = lb.sum(lb.kron(A.comps[0][:, i, :], B.comps[0][:, i, :]) for i in range(d))

        for pos in range(1, n_comps):
            d = A.dims[pos]
            v = lb.sum(v.dot(lb.kron(A.comps[pos][:, i, :], B.comps[pos][:, i, :])) for i in range(d))

        return float(v)

    # TODO: rename
    @staticmethod
    def skp(A, B):
        return TensorTrain.hadamard_product(A, B)

    @staticmethod
    def frob_norm(A):
        return lb.sqrt(TensorTrain.skp(A, A))

    def __add__(self, other):
        """ Adds a nother TT"""
        assert (lb.equal(self.dims, other.dims).all())

        res = TensorTrain(self.dims)  # , self.rank + other.ranks
        comps = []

        # Set first component
        data = lb.concatenate([copy.copy(self.comps[0][0, :, :]), copy.copy(other.comps[0][0, :, :])], axis=1)
        data = data.reshape(1, data.shape[0], data.shape[1])
        comps.append(data)

        # set middle components
        for p in range(1, res.n_comps - 1):
            # r_{i}^1  di  r_{i+1}^1
            c1, c2 = self.comps[p], other.comps[p]
            rp1, _, rpp1 = c1.shape
            # r_{i}^2  di  r_{i+1}^2
            rp2, _, rpp2 = c2.shape
            data = lb.zeros((rp1 + rp2, self.dims[p], rpp1 + rpp2))
            data[:rp1, :, :rpp1] = copy.copy(c1)
            data[rp1:, :, rpp1:] = copy.copy(c2)
            comps.append(data)

        data = lb.concatenate([copy.copy(self.comps[-1][:, :, 0]), copy.copy(other.comps[-1][:, :, 0])], axis=0)
        data = data.reshape(data.shape[0], data.shape[1], 1)
        comps.append(data)

        res.set_components(comps)

        # # retract result to maximal possible ranks
        # res.retract(res.ranks[1:-1])
        return res

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self, other):
        other.comps[0] *= -1.0
        return other

    @staticmethod
    def tt_svd(A):
        # tensor train svd of full tensor A
        pass

    def __repr__(self):
        output = "{c}TensorTrain :{r}".format(c=Fore.GREEN, r=Style.RESET_ALL) + "\nranks = " + str(
            self.rank) + "\n dims = " + str(self.dims)
        output += "\n Components: "
        for p, c in enumerate(self.comps):
            output += "\n\n {color}C[{p}] with shape = {s} :{reset} \n {c}".format(p=p, c=c, s=c.shape,
                                                                                   color=Fore.GREEN,
                                                                                   reset=Style.RESET_ALL)
        return output

    def __str__(self):
        output = "{c}TensorTrain :{r}".format(c=Fore.GREEN, r=Style.RESET_ALL) + "\nranks = " + str(
            self.rank) + "\n dims = " + str(self.dims)
        output += "\n\nComponents: "
        for p, c in enumerate(self.comps):
            output += "\n\n {color}C[{p}] with shape = {s} :{reset} \n {c}".format(p=p, c=c, s=c.shape,
                                                                                   color=Fore.GREEN,
                                                                                   reset=Style.RESET_ALL)
        return output


# # TODO: remove, since superseded by orthpoly_basis class
# class TensorPolynomial(object):

#     def __init__(self, degrees):
#         self.d = len(degrees)
#         self.degs = degrees

#         # maps  n -> polynomials of degree degs[n]
#         self.polynomials = {}

#     def set_polynomial(self, polytype, indices = None):

#         idx = indices if indices is not None else range(self.d)
#         for m in idx:
#             self.polynomials[m] = [ polytype(deg) for deg in range(self.degs[m])]

#     def __call__(self, x):
#         """
#            @param x is an  b x d  array of input data with b > 0
#            @returns returns a rank 1 object u = [u1, ..., ud] with ui of shape b x deg[i],
#                     so query u[i][j,k] is the k-th polynomial (in that dimension) evaluated at the j-th samples i-th component

#         """
#         assert(x.shape[1] == self.d)

#         u = [None] * self.d
#         for m in range(self.d):
#             u[m] = lb.stack([ self.polynomials[m][k](x[:,m]) for k in range(self.degs[m]) ], axis = 1)

#         return u

def conjugate_grad(S, b, x, tol=1e-3):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    # A = np.array(At)
    # b = np.squeeze(np.array(bt))
    # x = np.squeeze(np.array(xt))
    n = len(b)
    r = S(x) - b
    p = - r
    r_k_norm = lb.dot(r, r)
    for i in range(2 * n):
        # Ap = np.dot(A, p)
        Ap = S(p)
        alpha = r_k_norm / lb.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = lb.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < tol:
            # print('Itr:', i)
            break
        p = beta * p - r
    return lb.tensor(x)[:, None]


class ALS_Regression(object):
    """
    TODO basis choice for the fitting for numerical stability :

                waehle basis  B ortho bzgl ||.||_1


                min ||f- TT||_0  + lam ||TT||_1

                berechne gram matrix von B bzgl  0  = G


                G = eigen value representatn  =  U D U^T

                B =  U * B
    """

    def __init__(self, xTT):
        self.xTT = xTT
        # left and right contraction of non-active components
        self.L = None
        self.R = None

        # self.loc_solver_opt = {'modus' : 'normal', }
        self.memory_summary_tracker = SummaryTracker()

    def memory_track_print_diff(self, msg):
        if self.memory_summary_tracker is not None:  # mem-track is active
            print(msg)
            self.memory_summary_tracker.print_diff()

    # TODO enable ALS with only forward half sweeps
    # TODO early stopping when residual grows
    # TODO early stopping when rank updates yield not significant improvement/overfitting
    # TODO early stopping based on overfitting (compute residual on separate validation set)
    # TODO L1 regularisation (Philipp)
    def solve(self, x, y, iterations, tol, verboselevel, rule=None, reg_param=None):

        """
            @param loc_solver : 'normal', 'least_square',
            x shape (batch_size, input_dim)
            y shape (batch_size, 1)
        """

        tol = -1  # FIXME hack to make full iterations for memory leak check
        # size of the data batch
        b = y.shape[0]

        # feature evaluation on input data
        u = self.xTT.tfeatures(x)

        # 0 - orthogonalize, s.t. sweeping starts on first component
        self.xTT.tt.set_core(mu=0)

        # TODO: name stack instead of list
        # initialize lists for left and right contractions
        R_stack = [lb.ones((b, 1))]
        L_stack = [lb.ones((b, 1))]

        d = self.xTT.tt.n_comps

        def add_contraction(mu, list, side='left'):

            assert ((side == 'left' or side == 'right') or (side == +1 or side == -1))

            with TicToc(key=" o left/right contractions ", do_print=False, accumulate=True, sec_key="ALS: "):
                core_tensor = self.xTT.tt.comps[mu]
                data_tensor = u[mu]
                contracted_core = lb.einsum('idr, bd -> bir', core_tensor, data_tensor)
                if (side == 'left' or side == -1):
                    list.append(lb.einsum('bir, bi -> br', contracted_core, list[-1]))
                else:
                    list.append(lb.einsum('bir, br -> bi', contracted_core, list[-1]))

        def solve_local_iterativeCG(mu, L, R):
            """
                solves the local system of equation approximately in a matrix free fashion
                performing   one gradient decent step ( = one Conjugate Gradient step) with
                optimal stepsize.

                min  || A x - b ||^2 +  lam ||x||_F^2

                with stationary point equation:

                     S x  = b*

                with
                    S = (2 A^t*A + lam I)
                    b* = A^t b

                we set :


                r = b* - Sx
                s = r^Tr / r^T(S r)
                x = x + s* r

                here s is the locally optimal stepsize for quadratic function

            """
            with TicToc(key=" o CG step workload ", do_print=False, accumulate=True, sec_key="ALS: "):
                A = lb.einsum('bi,bj,br->bijr', L, u[mu], R)
                A = A.reshape(A.shape[0], A.shape[1] * A.shape[2] * A.shape[3])
                ATA, ATy = A.T @ A, lb.squeeze(A.T @ y)

                S = ATA
                if reg_param is not None:
                    assert isinstance(reg_param, float)
                    S += reg_param * lb.eye(ATA.shape[0])

                c = self.xTT.tt.comps[mu]
                v = c.reshape(c.shape[0] * c.shape[1] * c.shape[2])
                v = conjugate_grad(lambda v: S @ v, ATy, v)

                c = v.reshape(c.shape[0], c.shape[1], c.shape[2])
                self.xTT.tt.comps[mu] = c

        def solve_local_iterativeCG_matrixfree(mu, L, R):
            """
                solves the local system of equation approximately in a matrix free fashion
                performing   one gradient decent step ( = one Conjugate Gradient step) with
                optimal stepsize.

                min  || A x - b ||^2 +  lam ||x||_F^2

                with stationary point equation:

                     S x  = b*

                with
                    S = (2 A^t*A + lam I)
                    b* = A^t b

                we set :


                r = b* - Sx
                s = r^Tr / r^T(S r)
                x = x + s* r

                here s is the locally optimal stepsize for quadratic function

            """
            core_shape = self.xTT.tt.comps[mu].shape
            with TicToc(key=" o CG step workload ", do_print=False, accumulate=True, sec_key="ALS: "):
                # Compute  S*x
                def S(v):
                    with TicToc(key=" o S application ", do_print=False, accumulate=True, sec_key="ALS: "):
                        c = v.reshape(core_shape)
                        Ac = lb.einsum('bi,bj,br,ijr->b', L, u[mu], R, c)
                        AtAc = lb.einsum('bi,bj,br,b->ijr', L, u[mu], R, Ac)
                        Sc = 2 * AtAc
                        if reg_param is not None:
                            assert isinstance(reg_param, float)
                            Sc += reg_param * c
                        return Sc.reshape(Sc.shape[0] * Sc.shape[1] * Sc.shape[2])

                b_ast = 2 * lb.einsum('bi,bj,br,bk->ijr', L, u[mu], R, y)
                b_ast = b_ast.reshape(core_shape[0] * core_shape[1] * core_shape[2])

                c = self.xTT.tt.comps[mu]  # start value is the old core data
                v = c.reshape(c.shape[0] * c.shape[1] * c.shape[2])
                v = conjugate_grad(S, b_ast, v)

                self.xTT.tt.comps[mu] = v.reshape(c.shape)

        def solve_local(mu, L, R):

            with TicToc(key=" o least square matrix allocation ", do_print=False, accumulate=True, sec_key="ALS: "):
                A = lb.einsum('bi,bj,br->bijr', L, u[mu], R)
                A = A.reshape(A.shape[0], A.shape[1] * A.shape[2] * A.shape[3])

                if reg_param is not None:
                    assert isinstance(reg_param, float)

            with TicToc(key=" o local solve ", do_print=False, accumulate=True, sec_key="ALS: "):

                # c, res, rank, sigma = lb.linalg.lstsq(A, y, rcond = None)
                ATA, ATy = A.T @ A, A.T @ y

                if reg_param is not None:
                    assert isinstance(reg_param, float)
                    ATA += reg_param * lb.eye(ATA.shape[0])

                # c = lb.linalg.solve(ATA, ATy)

                # rel_err = lb.linalg.norm(A @ c - y) / lb.linalg.norm(y)
                rel_err = 0.5
                if rel_err > 1e-4:
                    with TicToc(key=" o local solve via lstsq ", do_print=False, accumulate=True, sec_key="ALS: "):
                        if reg_param is not None:
                            Ahat = lb.concatenate([A, lb.sqrt(reg_param) * lb.eye(A.shape[1])], 0)
                            yhat = lb.concatenate([y, lb.zeros((A.shape[1], 1))], 0)
                            c, res, rank, sigma = lb.linalg.lstsq(Ahat, yhat, rcond=None)
                        else:
                            c, res, rank, sigma = lb.linalg.lstsq(A, y, rcond=None)

                s = self.xTT.tt.comps[mu].shape
                self.xTT.tt.comps[mu] = c.reshape(s[0], s[1], s[2])

        # initialize residual
        # TODO rename to rel res
        curr_res = lb.linalg.norm(self.xTT(x) - y) ** 2 / lb.linalg.norm(y) ** 2  # quadratic norm
        if verboselevel > 0: print("START residuum : ", curr_res)

        # initialize stop condition
        niter = 0
        stop_condition = niter > iterations or curr_res < tol

        # loc_solver =  solve_local_iterativeCG
        loc_solver = solve_local

        # before the first forward sweep we need to build the list of right contractions
        for mu in range(d - 1, 0, -1):
            add_contraction(mu, R_stack, side='right')

        history = []

        iter = 0
        gb_const = 1024 * 1024 * 1024
        vmem_gb = np.round(psutil.virtual_memory().total / gb_const, 1)
        while not stop_condition:
            print(
                f'Percentage of consumed vmem at iter # {iter} = '
                f'{np.round(psutil.virtual_memory().used / gb_const, 1)} GB = '
                f'{psutil.virtual_memory().percent}% out of {vmem_gb} GB')
            self.memory_track_print_diff("At loop beginning")
            # forward half-sweep
            for mu in range(d - 1):
                self.xTT.tt.set_core(mu)
                if mu > 0:
                    add_contraction(mu - 1, L_stack, side='left')
                    del R_stack[-1]
                loc_solver(mu, L_stack[-1], R_stack[-1])

            # before back sweep
            self.xTT.tt.set_core(d - 1)
            add_contraction(d - 2, L_stack, side='left')
            del R_stack[-1]

            # backward half sweep
            for mu in range(d - 1, 0, -1):
                self.xTT.tt.set_core(mu)
                if mu < d - 1:
                    add_contraction(mu + 1, R_stack, side='right')
                    del L_stack[-1]
                loc_solver(mu, L_stack[-1], R_stack[-1])

            # before forward sweep
            self.xTT.tt.set_core(0)
            add_contraction(1, R_stack, side='right')
            del L_stack[-1]

            # update stop condition
            niter += 1
            curr_res = lb.linalg.norm(self.xTT(x) - y) ** 2 / lb.linalg.norm(y) ** 2
            # update reg_param
            # reg_param = 1e-6*curr_res.item()
            stop_condition = niter > iterations or curr_res < tol
            if verboselevel > 0:  # and  niter % 10 == 0:
                print("{c}{k:<5}. iteration. {r} Data residuum : {c2}{res}{r}".format(c=Fore.GREEN, c2=Fore.RED,
                                                                                      r=Style.RESET_ALL, k=niter,
                                                                                      res=curr_res))

            history.append(curr_res)
            Hlength = 5
            rateTol = 1e-5

            if len(history) > Hlength:
                latestH = lb.tensor(history[-Hlength:])
                relative_history_rate = lb.cov(latestH) / lb.mean(latestH)

                if relative_history_rate < rateTol:
                    if verboselevel > 0:
                        print("===== Attempting rank update ====")
                    if rule is not None:
                        self.xTT.tt.modify_ranks(rule)
                        # set core to 0 and re-initialize lists
                        self.xTT.tt.set_core(mu=0)
                        R_stack = [lb.ones((b, 1))]
                        L_stack = [lb.ones((b, 1))]
                        for mu in range(d - 1, 0, -1):
                            add_contraction(mu, R_stack, side='right')

                        history = []

            self.memory_track_print_diff("At loop end")
            iter += 1
        return self.xTT.tt

    def tangent_fit(self,
                    x,
                    y,
                    reg=False,
                    reg_coeff=1e-6,
                    reg0=False,
                    reg0_coeff=1e-2,
                    verbose=True):
        """performs a fit sum_k| T(x_k) - y_k |^2 ---> min_T,
        where x_k are the vector valued data points, y_k are the targets and
        the minimum is sought over elements of the tangent space to the TT
        manifold M_r at the current TT, where r is the current TTs rank.
        The current TT is right orthogonalized initially.

        Parameters
        ----------
        data : torch.tensor
            data points x_i of the regression. Shape (batch_size,input_dim).
        targets : torch.tensor
            targets y_i of the regression. Shape (batch_size,output_dim).
        verbose : bool
            determines if error of the fit is printed.

        Returns
        -------
        None.

        """

        self.xTT.tt.set_core(self.xTT.d - 1)

        Xtt = copy.copy(self.xTT)

        d_list = Xtt.tfeatures.degs
        d = Xtt.d

        r = [1] + [Xtt.tt.comps[mu].shape[2] for mu in range(d)]

        if reg0 == True:
            xhat = lb.concatenate([x, lb.zeros((1, d))], 0)
            yhat = lb.concatenate([y, lb.zeros((1, 1))], 0)
        else:
            xhat = x
            yhat = y

        b = yhat.shape[0]

        # get the left orthgonal basis for every core except the last
        Q_hat_list = []
        for _ind in range(d - 1):
            u = tensor.left_unfolding(Xtt.tt.comps[_ind])
            k = u.shape[0] - u.shape[1]
            if k <= 0:
                # raise Exception("r_{i-1}*n_i > r_i must hold true for all cores. Violated here.")
                Q_hat_list.append(None)
            else:
                u_k = lb.random.rand(u.shape[0], k)
                u_k -= (u @ u.T) @ u_k
                Q_hat_list.append(u_k)

        # get indices of 0-dimensional spaces
        nones = [i for i in range(len(Q_hat_list)) if Q_hat_list[i] == None]

        # Now assemble the matrices C_i from left to right
        C_list = []

        # initialize lists for left and right contractions
        data = Xtt.tfeatures(xhat)
        R_stack = [lb.ones((b, 1))]
        L = lb.ones((b, 1))

        def add_contraction(mu, data_tensor, contraction, side='left'):

            assert ((side == 'left' or side == 'right') or (side == +1 or side == -1))

            with TicToc(key=" o left/right contractions ", do_print=False, accumulate=True, sec_key="ALS: "):
                core_tensor = self.xTT.tt.comps[mu]
                data_tensor = data_tensor[mu]
                contracted_core = lb.einsum('idr, bd -> bir', core_tensor, data_tensor)
                if (side == 'left' or side == -1):
                    if isinstance(contraction, list):
                        contraction.append(lb.einsum('bir, bi -> br', contracted_core, contraction[-1]))
                    else:
                        contraction = lb.einsum('bir, bi -> br', contracted_core, contraction)
                        return contraction
                else:
                    if isinstance(contraction, list):
                        contraction.append(lb.einsum('bir, br -> bi', contracted_core, contraction[-1]))
                    else:
                        contraction = lb.einsum('bir, bi -> bi', contracted_core, contraction)
                        return contraction

        # build list of right contractions before sweeping
        for mu in range(d - 1, 0, -1):
            add_contraction(mu, data, R_stack, side='right')

        # sweep from left to right
        for mu in range(0, d):
            # for all but the first core, we need to build left contractions
            if mu > 0:
                L = add_contraction(mu - 1, data, L, side='left')
                del R_stack[-1]

            if mu in nones:
                C_list.append(None)

            else:
                # get contraction from right up to current core
                R = R_stack[-1]
                data_tensor = data[mu]

                # build linear operator A for ||A*core-targets||->min
                C_tensor = lb.einsum('bi, bd, br -> bidr', L, data_tensor, R)

                # flatten all but the batch dimension
                C = C_tensor.reshape(C_tensor.shape[0],
                                     C_tensor.shape[1] *
                                     C_tensor.shape[2],
                                     C_tensor.shape[3])

                # TODO: check for numpy ordering !!!!!
                # Due to the order in which reshaping works in torch, we need to make
                # manual adjustments. the default reshaping will not concatenate
                # the columns of the left unfolding (as intended) but the rows.
                # Hence, we transpose the left unfolding before the last reshaping.
                C = lb.transpose(C, 1, 2)
                C = C.reshape(C.shape[0], C.shape[1] * C.shape[2])

                if reg0:
                    C[-1, :] *= reg0_coeff

                C_list.append(C)

        # Finally, assemble the global matrix A
        A = lb.tensor([])

        for _ind in range(d - 1):
            C = C_list[_ind]
            if C is not None:
                CZ = lb.tensor([])
                Q_hat = Q_hat_list[_ind]
                for j in range(r[_ind + 1]):
                    CZ = lb.concatenate((CZ, C[:, j * r[_ind] * d_list[_ind]:
                                                  (j + 1) * r[_ind] * d_list[_ind]] @ Q_hat), 1)

                A = lb.concatenate((A, CZ), 1)

        # add last C matrix where Z = Id
        A = lb.concatenate((A, C_list[-1]), 1)

        # we arrived at the regression problem || Ax - targets ||^2 ---> min
        targets = copy.deepcopy(yhat)

        # # normal equations
        # # assemble the linear system A^TA*x = A^T*targets
        # ATA = lb.einsum('bi, bj -> ij', A, A)
        # ATb = lb.einsum('bi, bl -> i', A, y)[:, None]

        # # build linear operator for lsq and solve
        # operator = ATA
        # if reg:
        #     operator += reg_coeff * lb.eye(operator.shape[0])
        # x = lb.linalg.solve(operator,ATb)

        # lstsq
        if reg:
            targets = lb.concatenate([targets, lb.zeros((A.shape[1], 1))], 0)
            A = lb.concatenate((A, lb.sqrt(reg_coeff) * lb.eye(A.shape[1])), 0)

        x, res, rank, sigma = lb.linalg.lstsq(A, targets)

        if verbose:
            if (reg == True and reg0 == True):
                err = (lb.linalg.norm(A[:-1 - A.shape[1], :] @ x - y) / lb.linalg.norm(y)).item()
            elif reg == True:
                err = (lb.linalg.norm(A[:-A.shape[1], :] @ x - y) / lb.linalg.norm(y)).item()
            elif reg0 == True:
                err = (lb.linalg.norm(A[:-1, :] @ x - y) / lb.linalg.norm(y)).item()
            else:
                err = (lb.linalg.norm(A @ x - y) / lb.linalg.norm(y)).item()
            print("tangent fit residuum : {c2}{res}{r}".format(c2=Fore.RED, r=Style.RESET_ALL, res=err))

        # recover the element W of the tangent space from coefficients x
        W = []
        for _ind in range(d - 1):
            Q_hat = Q_hat_list[_ind]
            if Q_hat is not None:
                W_ind = lb.tensor([])
                length_x_ind = r[_ind] * d_list[_ind] * r[_ind + 1] \
                               - r[_ind + 1] ** 2
                x_ind = x[:length_x_ind]

                dim = r[_ind] * d_list[_ind] - r[_ind + 1]
                if dim == 0:
                    dim = 1
                for j in range(r[_ind + 1]):
                    W_ind_j = 0.
                    for k in range(dim):
                        W_ind_j = W_ind_j + x_ind[j * dim + k] * Q_hat[:, k][:, None]
                    W_ind = lb.concatenate((W_ind, W_ind_j), 1)
            else:
                W_ind = lb.zeros((r[_ind] * d_list[_ind], r[_ind + 1]))
                length_x_ind = 0

            W.append(W_ind)
            x = x[length_x_ind:]

            # W is in left unfolded shape, we fold back to tensor shape
        for i in range(len(W)):
            W[i] = W[i].reshape(r[i], d_list[i], r[i + 1])

        # last part of W is still missing
        W.append(x.view(r[-2], d_list[-1], r[-1]))

        return W


class Extended_TensorTrain(object):

    def __init__(self, tfeatures, ranks, comps=None):
        """
            tfeatures should be a function returning evaluations of feature functions if given a data batch as argument,
            i.e. tfeatures(x), where x is an lb.array of size (batch_size, n_comps),
            is a list of lb.arrays such that tfeatures(x)[i][j,k] is the k-th feature function (in that dimension)
            evaluated at the j-th samples i-th component
        """

        self.tfeatures = tfeatures
        self.d = self.tfeatures.d

        # TODO also allow ranks len = d + 1  with [1] [...] + [1] shape
        assert (len(ranks) == self.tfeatures.d - 1)
        self.rank = ranks

        self.tt = TensorTrain(tfeatures.degs)
        if comps is None:
            self.tt.fill_random(ranks)
        else:
            # TODO allow ranks len d+1
            for pos in range(self.tfeatures.d - 1):
                assert (comps[pos].shape[2] == ranks[pos])
            self.tt.set_components(comps)

        self.L = None
        self.R = None

        dofs = 0
        for d in range(self.d):
            dofs += lb.numel(self.tt.comps[d])
        self.dofs = dofs

    def __call__(self, x):
        assert (x.shape[1] == self.d)
        u = self.tfeatures(x)
        return self.tt.dot_rank_one(u)

    def set_ranks(self, ranks):
        self.tt.retract(self, ranks, verbose=False)

    def grad(self, x):
        """computes the analytical gradient of the forward pass. Works only
           for polynomial features.

        Parameters
        ----------
        x : lb.tensor
            input of shape (batch_size,input_dim)

        Returns
        -------
        gradient : lb.tensor
            gradient of the forward pass. Shape (batch_size,input_dim)

        """
        assert (x.shape[1] == self.d)
        # initialize gradient
        gradient = lb.zeros((x.shape[0], self.d))

        # lift data to feature space and feature-derivative space
        embedded_data = self.tfeatures(x)
        embedded_data_grad = self.tfeatures.grad(x)

        data = [embedded_data_grad[0]] + embedded_data[1:]
        gradient[:, 0] = lb.squeeze(self.tt.dot_rank_one(data))
        for mu in range(1, self.d - 1):
            data = embedded_data[:mu] + [embedded_data_grad[mu]] + embedded_data[mu + 1:]
            gradient[:, mu] = lb.squeeze(self.tt.dot_rank_one(data))
        data = embedded_data[:self.d - 1] + [embedded_data_grad[self.d - 1]]
        gradient[:, self.d - 1] = lb.squeeze(self.tt.dot_rank_one(data))

        return gradient

    def fit(self, x, y, iterations, rule=None, tol=8e-6, verboselevel=0, reg_param=None):
        """
            Fits the Extended Tensortrain to the given data (x,y) of some target function
                     f : K\subset IR^d to IR^m
                                     x -> f(x) = y.

            @param x : input parameter of the training data set : x with shape (b,d)   b \in \mathbb{N}
            @param y : output data with shape (b,m)
        """

        assert (x.shape[1] == self.d)

        solver = ALS_Regression(self)
        # with TicToc(key=" o ALS total ", do_print=False, accumulate=True, sec_key="ALS: "):
        #     res = solver.solve(x,y,iterations,tol,verboselevel, rule)
        with TicToc(key=" o ALS total ", do_print=False, accumulate=True, sec_key="ALS: "):
            res = solver.solve(x, y, iterations, tol, verboselevel, rule, reg_param)
        self.tt.set_components(res.comps)

    def tangent_fit(self, x, y, reg=False, reg_coeff=1e-6, reg0=False, reg0_coeff=1e-2, verbose=True):

        solver = ALS_Regression(self)
        with TicToc(key=" o tangent fit total ", do_print=False, accumulate=True, sec_key="TF: "):
            res = solver.tangent_fit(x, y, reg, reg_coeff, reg0, reg0_coeff, verbose)
        return res

    # TODO: - move to base class
    #       - enable other rules than retraction
    def tangent_add(self, alpha, W):
        """performs an addition U + alpha*del_U, where U is the current TT and
        del_U is the tangent space element represented by W.

        Parameters
        ----------
        W : list
            list of cores representing del_U.

        Returns
        -------
        added_cores_list : list
            list of new cores after addition and retraction.
        """
        res = Extended_TensorTrain(self.tfeatures, self.rank)
        comps = []

        # compute first part
        data = lb.concatenate([copy.copy(alpha * W[0][0, :, :]), copy.copy(self.tt.comps[0][0, :, :])], axis=1)
        data = data.reshape(1, data.shape[0], data.shape[1])
        comps.append(data)

        # compute intermediate parts
        for p in range(1, res.tt.n_comps - 1):
            # r_{i}^1  di  r_{i+1}^1
            c1, c2 = self.tt.comps[p], alpha * W[p]
            rp1, _, rpp1 = c1.shape
            # r_{i}^2  di  r_{i+1}^2
            rp2, _, rpp2 = c2.shape
            data = lb.zeros((rp1 + rp2, self.tt.dims[p], rpp1 + rpp2))
            data[:rp1, :, :rpp1] = copy.copy(c1)
            data[rp1:, :, rpp1:] = copy.copy(c1)
            data[rp1:, :, :rpp1] = copy.copy(c2)
            comps.append(data)

        # last part
        data = lb.concatenate([copy.copy(self.tt.comps[-1][:, :, 0]),
                               copy.copy(self.tt.comps[-1][:, :, 0]) + alpha * copy.copy(W[-1][:, :, 0])], axis=0)
        data = data.reshape(data.shape[0], data.shape[1], 1)
        comps.append(data)

        res.tt.set_components(comps)
        res.tt.retract(self.rank)

        return res

    def check_dofs(self):
        dofs = 0
        for d in range(self.d):
            dofs += self.tt.comps[d].size()
        self.dofs = dofs


class ALS_Solver(object):
    def __init__(self):
        pass


def testplotxtt():
    dims = [3, 3]
    tfeatures = TensorPolynomial(dims)

    tfeatures.set_polynomial(polytype=Legendre.basis)
    xTT = Extended_TensorTrain(tfeatures, ranks=[3])

    X = lb.linspace(-1, 1, 100)
    Y = lb.linspace(-1, 1, 100)
    X, Y = lb.meshgrid(X, Y)
    X = X.reshape(X.shape[0] * X.shape[1], )
    Y = Y.reshape(Y.shape[0] * Y.shape[1], )
    x = lb.stack([X, Y], axis=1)
    y = xTT(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X.reshape(100, 100), Y.reshape(100, 100), y.reshape(100, 100))
    plt.show()


def testmodifyrank():
    d = 5
    dims = [4] * d
    # ranks = lb.random.randint(2, 50, d-1).tolist()#[3] * (d-1)
    ranks = [7] * 4  # [4,16,16,4]
    print("random Ranks = ", ranks)
    X = TensorTrain(dims=dims)
    X.fill_random(ranks)

    X.set_core(d - 1)
    X.set_core(0)

    print(X.ranks)

    rule = Dörfler_Adaptivity(delta=1e-10, maxranks=[32] * (d - 1), dims=dims)

    X.modify_ranks(rule)

    print("X.ranks = ", X.ranks)

    X.set_core(0)

    exit()

    # X.round(1e-16)

    # print(X.ranks)

    ##X.set_core(4)
    # print(X.ranks)

    # exit()
    # X.set_core(0)

    Y = TensorTrain(dims=dims)
    Y.fill_random(ranks)
    Y.set_core(0)

    print("X.ranks = ", X.ranks)
    print("Y.ranks = ", Y.ranks)
    # for c in X.comps:
    #    print(c.shape)

    print("============")

    Z = X + X

    Z.set_core(4)

    print(Z.ranks)

    Z.set_core(0)

    print(Z.ranks)

    Z.round(1e-16)
    print("Z ranks rounded : ", Z.ranks)

    exit()

    for c in Z.comps:
        print(c.shape)

    exit()

    exit()

    X.modify_ranks(rule, verbose=True)


class alea_basis_wrapper(object):

    def __init__(self, degrees, families):
        assert (len(degrees) == len(families))
        self.fams = families
        self.degrees = degrees

    def __call__(self, x):
        assert (x.shape[1] == len(self.fams))
        # basis family evaluations of each basis function for each coordinate
        b_vals = [lb.stack(fam.eval(self.degrees[i], x=x[:, i], all_degrees=True), axis=1) for i, fam in
                  enumerate(self.fams)]
        return b_vals


def main(verbose):
    # lb.random.seed(1)

    d = 5
    dims = [10] * d
    ranks = [4, 6, 8, 4]  # * (d-1)

    # d = 2
    # dims = [4,4]
    # ranks = [4] * (d-1)

    ranks2 = [1] * (d - 1)
    ranks3 = [10] * (d - 1)
    # tfeatures = TensorPolynomial(dims)
    # tfeatures.set_polynomial(polytype = Legendre.basis)

    tfeatures = orthpoly_basis(degrees=dims, domain=[-1., 1], norm='H1')

    xTT = Extended_TensorTrain(tfeatures, ranks)

    # Create a target solution
    # X = TensorTrain(dims= dims)
    # lb.random.seed(1)
    # X.fill_random(ranks)
    # A = X.full()
    # def tensorleg(x, idx):
    #     res = Legendre.basis(idx[0])(x[:,0])
    #     for k in range(1,len(idx)):
    #         res = Legendre.basis(idx[k])(x[:,k]) * res
    #     return res
    # f = lambda x :  lb.tensor(np.sum(A[i]*tensorleg(x, i) for i in product(*[list(range(d)) for d in dims])))

    # sin ( x)  = (e^(ix) - e^{-ix}) / 2i   in the complex field, matrix rank is invariant of field choise, thus f has FTT rank 2
    f = lambda x: lb.tensor(np.sin(np.sum(x[:, i] ** 2 for i in range(d))))

    from scipy.linalg import expm

    S = np.random.rand(d, d)
    S = 0.5 * (S + S.T)

    S = expm(S)
    # f = lambda x : lb.tensor(np.exp(- 0.5 * np.einsum('bi,ij,bj->b',x,S,x)))# np.exp(np.sum(x[:,i] for i in range(d)))

    # create data points
    N = 3 * d * max(ranks) ** 2 * dims[0]
    xx = lb.random.rand(N, d) * 2 - 1
    yy = f(xx).reshape(N, 1)

    # lb.random.seed(1)
    # Xtt = Extended_TensorTrain(tfeatures,ranks2)

    # print(Xtt.tt.comps[0])

    # xx2 = lb.random.rand(N,d)*2 -1
    # yy2 = Xtt(xx2)

    # W = Xtt.tangent_fit(xx2,yy2)

    # res = Xtt.tangent_add(1.,W)
    # print(lb.linalg.norm(res(xx2)-2*Xtt(xx2))/lb.linalg.norm(2*Xtt(xx2)))

    print("================= start fit ==============")
    # rule = Dörfler_Adaptivity(delta=1e-6, maxranks=[32] * (d - 1), dims=dims, rankincr=1)
    xTT.fit(xx, yy, iterations=2000, verboselevel=1, rule=None, reg_param=1e-6)

    print("non rounded rank: ", xTT.tt.rank)
    xTT.tt.round(1e-6, verbose=True)
    print("rounded rank: ", xTT.tt.rank)
    print("Final residuum: ", lb.linalg.norm(xTT(xx) - yy) ** 2 / lb.linalg.norm(yy) ** 2)

    TicToc.sortedTimes()

    # if d == 2 :
    #     fig = plt.figure()

    #     X = lb.linspace(-1,1,100)
    #     Y = lb.linspace(-1,1,100)
    #     X, Y= lb.meshgrid(X,Y)

    #     xx = lb.stack([X.flatten(),Y.flatten()], axis = 1)

    #     yy = xTT(xx).reshape(100,100)
    #     yyf= f(xx).reshape(100,100)

    #     ax = fig.add_subplot(121, projection='3d')
    #     ax.plot_surface(X,Y,lb.abs(yy-yyf))

    #     ax = fig.add_subplot(122, projection='3d')
    #     ax.plot_surface(X,Y,yyf)

    #     plt.show()


if __name__ == "__main__":
    main(True)
    # checks()
