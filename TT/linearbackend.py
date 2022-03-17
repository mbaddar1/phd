# set flag which backend is used

# try:
#     import torch                   
# except ImportError:
#     try:
#         import numpy as np    
#     except ImportError:     
#         print("No Linear backend package avaiable")
#     #else:

import numpy as np
import torch
print("pytorch version: ", torch.__version__)  



#from linearbackend import Linear_Backend
#backend_options = {  "backend": "numpy"/"torch", 
#                     "device" : "cpu"/"cuda"}




class Linear_Algebra():
    def __init__(self, backend_options):

        assert "backend" in backend_options 
        
        self.backend = backend_options["backend"]
        assert(self.backend in ["numpy", "torch"])

        if self.backend == "torch":
            if "device" in backend_options:
                self.device = backend_options["device"]
                assert(self.device in ["cpu", "cuda"])
            else:
                self.device = "cpu"

    def svd(self, a, full_matrices=True, compute_uv=True, hermitian=False):
        if self.backend == "numpy" : 
            return np.linalg.svd(a, full_matrices, compute_uv, hermitian)
        elif self.backend == "torch":
            return torch.linalg.svd(a, full_matrices)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")
    
    def qr(self, a, mode='reduced'):
        if self.backend == "numpy" : 
            return np.linalg.qr(a, mode)

        elif self.backend == "torch":
            return torch.linalg.qr(a, mode)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def norm(self, x, ord=None, axis=None, keepdims=False):
        if self.backend == "numpy" : 
            return np.linalg.norm(x,ord,axis,keepdims)

        elif self.backend == "torch":
            return torch.linalg.norm(x,ord,axis,keepdims)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def solve(self, a,b):
        if self.backend == "numpy" : 
            return np.linalg.solve(a,b)

        elif self.backend == "torch":
            return torch.linalg.solve(a,b)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")
    
    def inv(self, a):
        if self.backend == "numpy" : 
            return np.linalg.inv(a)

        elif self.backend == "torch":
            return torch.linalg.inv(a)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def det(self, a):
        if self.backend == "numpy" : 
            return np.linalg.det(a)

        elif self.backend == "torch":
            return torch.linalg.det(a)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def lstsq(self, a, b, rcond=None):
        if self.backend == "numpy" : 
            return np.linalg.lstsq(a,b,rcond)

        elif self.backend == "torch":
            return torch.linalg.lstsq(a,b,rcond)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def cond(self, A,p=None):
        if self.backend == "numpy" : 
            return np.linalg.cond(A,p)

        elif self.backend == "torch":
            return torch.linalg.cond(A,p)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")




class Random(object):

    def __init__(self, backend_options):

        assert "backend" in backend_options 
        
        self.backend = backend_options["backend"]
        assert(self.backend in ["numpy", "torch"])

        if self.backend == "torch":
            if "device" in backend_options:
                self.device = backend_options["device"]
                assert(self.device in ["cpu", "cuda"])
            else:
                self.device = "cpu"

    def rand(self, *d):
        if self.backend == "numpy" : 
            return np.random.rand(*d)

        elif self.backend == "torch":
            return torch.rand(*d,device=self.device)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def randn(self, *d):
        if self.backend == "numpy" : 
            return np.random.randn(*d)

        elif self.backend == "torch":
            return torch.randn(*d,device=self.device)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def seed(self, seed=None):
        if self.backend == "numpy" : 
            return np.random.seed(seed)

        elif self.backend == "torch":
            return torch.manual_seed(seed)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")




class Linear_Backend(object):
    """
    Wrapper class to unified functionality of pytorch and numpy. 
    """
    def __init__(self, backend_options):

        assert "backend" in backend_options 
        
        self.backend = backend_options["backend"]
        assert(self.backend in ["numpy", "torch"])

        if self.backend == "torch":
            if "device" in backend_options:
                self.device = backend_options["device"]
                assert(self.device in ["cpu", "cuda"])
            else:
                self.device = "cpu"

        self.linalg = Linear_Algebra(backend_options)

        self.random = Random(backend_options)

        # define data types
        if self.backend == "numpy":
            self.float = np.float
            self.double = np.double
        else:
            self.float = torch.float
            self.double = torch.float

        # set default backend type
        if self.backend == "torch":
            torch.set_default_dtype(torch.float64)

    def array(self, object, dtype=None):
        if self.backend == "numpy":
            return np.array(object,dtype=dtype)
        if self.backend == "torch":
            return torch.tensor(object,dtype=dtype,device=self.device)

    def tensor(self, object, dtype=None):
        return self.array(object,dtype=dtype)

    def concatenate(self, A, axis=0):    
        if self.backend == "numpy" : 
            return np.concatenate(A,axis)
        elif self.backend == "torch":
            return torch.cat(A,axis)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")
        pass
    
    def dot(self, a,b):
        if self.backend == "numpy" : 
            return np.dot(a,b)
        elif self.backend == "torch":
            if (len(a.shape) == 1 and len(b.shape) == 1):
                return torch.dot(a,b)
            elif (len(a.shape) == 0 or len(b.shape) == 0):
                return torch.multiply(a,b)
            elif (len(a.shape) == 2 and len(b.shape) == 2):
                return torch.matmul(a,b)
            else:
                raise NotImplemented("Input is not supported by lb.dot. Check definition in linear backend.")
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def einsum(self, equation, *operands):
        if self.backend == "numpy" : 
            return np.einsum(equation, *operands)

        elif self.backend == "torch":
            return torch.einsum(equation, *operands)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def zeros(self, shape, dtype=float, order='C'):
        if self.backend == "numpy" : 
            return np.zeros(shape, dtype, order)

        elif self.backend == "torch":
            return torch.zeros(*shape, dtype=dtype, device=self.device)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def eye(self, n,dtype=None):
        if self.backend == "numpy" : 
            return np.eye(n, dtype=dtype)

        elif self.backend == "torch":
            return torch.eye(n,dtype=dtype,device=self.device)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def ones(self, shape, dtype=float, order='C'):
        if self.backend == "numpy" : 
            return np.ones(shape, dtype, order)

        elif self.backend == "torch":
            return torch.ones(*shape, dtype=dtype, device=self.device)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def diag(self, input, diagonal=0):
        if self.backend == "numpy" : 
            return np.diag(input,diagonal)

        elif self.backend == "torch":
            return torch.diag(input,diagonal)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def sum(self, a, axis=None, dtype = None):
  
        if self.backend == "numpy" : 
            return np.sum(a,axis,dtype)

        elif self.backend == "torch":
            if axis is not None:
                return torch.sum(a,axis,dtype=dtype)
            else:
                return torch.sum(a,dtype=dtype)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def max(self, a, axis=None, out=None,):
        if self.backend == "numpy" : 
            return np.max(a, axis=axis, out=out)

        elif self.backend == "torch":
            if axis is not None:
                return torch.max(torch.tensor(a),dim=axis)
            else:
                return torch.max(torch.tensor(a))
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")


    def sin(self, a):
        if self.backend == "numpy":
            return np.sin(a)

        elif self.backend == "torch":
            return torch.sin(a)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")


    def cos(self, a):
        if self.backend == "numpy":
            return np.cos(a)

        elif self.backend == "torch":
            return torch.cos(a)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")
    



    def asarray(self, a,dtype=None, order=None):
        if self.backend == "numpy":
            return np.asarray(a, dtype,order)

        elif self.backend == "torch":
            return torch.tensor(np.asarray(a, dtype,order),device=self.device)

        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def equal(self, x1,x2):
        if self.backend == "numpy" : 
            return np.equal(x1,x2)

        elif self.backend == "torch":
            return torch.equal(x1,x2)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def sqrt(self, x):
        if self.backend == "numpy" : 
            return np.sqrt(x)

        elif self.backend == "torch":
            return torch.sqrt(torch.tensor(x,device=self.device))
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def kron(self,a,b):
        if self.backend == "numpy" : 
            return np.kron(a,b)

        elif self.backend == "torch":
            return torch.kron(a,b)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def prod(self, input, dim=None, keepdim=False, dtype=None):
        if self.backend == "numpy" : 
            return np.prod(input,axis=dim,dtype=dtype,keepdims=keepdim)
        elif self.backend == "torch":
            if dim==None:
                return torch.prod(input,dtype=dtype)
            else:
                return torch.prod(input, dim=dim, keepdim=keepdim, dtype=dtype)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def abs(self,x):
        if self.backend == "numpy" : 
            return np.abs(x)

        elif self.backend == "torch":
            return torch.abs(torch.tensor(x,device=self.device))
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def stack(self, arrays, axis=0):
        if self.backend == "numpy" : 
            return np.stack(arrays, axis)

        elif self.backend == "torch":
            return torch.stack(arrays,axis)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")          

    def meshgrid(self, *tensors):
        if self.backend == "numpy" : 
            return np.meshgrid(*tensors)
        elif self.backend == "torch":
            return torch.meshgrid(*tensors)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def linspace(self,start, end, steps):
        if self.backend == "numpy" : 
            return np.linspace(start,end,steps)
        elif self.backend == "torch":
            return torch.linspace(start,end,steps,device=self.device)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def arange(self, start,stop,step,dtype=None):
        if self.backend == "numpy" : 
            return np.arange(start,stop,step,dtype=dtype)
        elif self.backend == "torch":
            return torch.arange(start,stop,step,dtype=dtype,device=self.device)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def squeeze(self, a, axis=None):
        if self.backend == "numpy" : 
            return np.squeeze(a,axis=axis)
        elif self.backend == "torch":
            if axis is not None:
                return torch.squeeze(a,dim=axis)
            else:
                return torch.squeeze(a)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def transpose(self, a, dim0, dim1):     # to transpose a matrix: dim0=1, dim1=0
        if self.backend == "numpy" : 
            axes = (dim0, dim1)
            return np.transpose(a, axes)
        elif self.backend == "torch":
            return torch.transpose(a, dim0, dim1)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")
        
    def all(self, input):
        if self.backend == "numpy" : 
            return np.all(input)
        elif self.backend == "torch":
            return torch.all(input)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def numel(self, input):
        if self.backend == "numpy" : 
            return input.size
        elif self.backend == "torch":
            return torch.numel(input)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def cov(self,m, rowvar=False):
        if self.backend == "numpy" : 
            return np.cov(m, rowvar=rowvar)
        elif self.backend == "torch":
            # """Estimate a covariance matrix (np.cov)"""
            # m = m - m.mean(dim=-1, keepdim=True)
            # factor = 1 / (m.shape[-1] - 1)
            # return factor * m @ m.transpose(-1, -2).conj()


            if m.dim() > 2:
                raise ValueError('m has more than 2 dimensions')
            if m.dim() < 2:
                m = m.view(1, -1)
            if not rowvar and m.size(0) != 1:
                m = m.t()
            # m = m.type(torch.double)  # uncomment this line if desired
            fact = 1.0 / (m.size(1) - 1)
            m -= torch.mean(m, dim=1, keepdim=True)
            mt = m.t()  # if complex: mt = m.t().conj()
            return fact * m.matmul(mt).squeeze()


        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def mean(self,m):
        if self.backend == "numpy" : 
            return np.mean(m)
        elif self.backend == "torch":
            return torch.mean(m)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")

    def exp(self,x):
        if self.backend == "numpy" : 
            return lb.exp(x)
        elif self.backend == "torch":
            return lb.exp(x)
        else:
            raise NotImplemented("Other linear backend packages not wrapped yet.")



# FIXME change back to cuda when needed
lb = Linear_Backend(backend_options = {"backend" : "torch", "device" : "cpu"})  # cuda
#lb = Linear_Backend(backend_options = {"backend" : "numpy"})


    

if __name__ == "__main__":

    lb = Linear_Backend(backend_options = {"backend" : "torch"})
    a = lb.tensor([[1,2,3,4],[0.1,0.1,0.1,0.1]])
    b = a
    c = lb.stack([a,b],0)
    print(c)

