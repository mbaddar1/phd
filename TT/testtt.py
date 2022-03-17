
from tt import TensorTrain

import numpy as np
import unittest

class TestTT(unittest.TestCase):

    def test_set_core(self):

        dims, ranks  = [3,4,7,4], [3,4,2]

        X = TensorTrain(dims = dims)
        X.fill_random(ranks = ranks)

        # full tensor as reference
        A = X.full()
        Mu = np.random.permutation(len(dims))

        for mu in Mu:
            X.set_core(mu)
            B = X.full()
            #print(" || X - CoreMove[mu={mu}](X) || = ".format(mu=mu), np.linalg.norm(A-B))
            self.assertAlmostEqual(np.linalg.norm(A-B), 0., places=13)

            # assertEqual
            # assertTrue
            # assertFalse

    def test_addition(self):
        dims = [6,4,8,5,4]
        ranksA, ranksB =  [4,3,4,3],  [2,3,4,2]
        A = TensorTrain(dims )
        A.fill_random(ranksA)

        B = TensorTrain(dims )
        B.fill_random(ranksB)
        Af = A.full()
        Bf = B.full()
        Y = A+B
        Yf = Y.full()
        self.assertAlmostEqual(np.linalg.norm(Af+Bf-Yf), 0., places=12)
   
    def test_rounding(self):
        dims = [6,4,5,4]
        ranks = [ [4,3,3],  [2,3,2], [6]*3 ] # third 3 contains ranks > dim which is shortened by default
        for rs in ranks:
            A = TensorTrain(dims)
            A.fill_random(rs)       
            A.set_core(0) 

            B = A + A
            B.round(1e-13, verbose = False) 

            Af = A.full()
            Bf = B.full()


            self.assertAlmostEqual(np.linalg.norm(2*Af-Bf), 0., places=10)

            self.assertTrue(np.equal(A.ranks, B.ranks).all(), msg = "\n A.ranks = {rA} \n B.ranks = {rB}".format(rA=A.ranks, rB=B.ranks))

            print("done")

    def test_skp(self):
        dims = [6,2,3,4]
        ranks = [2,4,3]
        A = TensorTrain(dims)
        A.fill_random(ranks)       
        A.set_core(0) 

        Af = A.full().flatten()
        print("af shape : ", Af.shape)
        res = TensorTrain.skp(A,A)

        self.assertAlmostEqual(res, np.dot(Af,Af) , places=12)

    def test_dot_rank1(self):
        dims, ranks = [6,4], [3]
        X = TensorTrain(dims = dims)
        X.fill_random(ranks)

        u = [np.random.rand(1,d) for d in dims]
        A = X.full()

        res = X.dot_rank_one(u)

        res2 = float(u[0].dot(A.dot(u[1].T)))
        print("test 2d rank1 contraction: ", np.linalg.norm(res-res2))
        self.assertAlmostEqual(float(res), res2 , places=12, msg  =" error in test dot rank1 product.")
       
    

if __name__ == '__main__':
    unittest.main(