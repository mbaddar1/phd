import numpy as np
from scipy.linalg import solve_triangular

# from linearbackendrbackend import Linear_Backend
#backend_options = {  "backend": "numpy"/"torch", 
#                     "device" : "cpu"/"cuda"}
from TT.linearbackend import Linear_Backend

device = "cpu"
lb = Linear_Backend(backend_options =  {"backend" : "torch", "device" : device})


def poly_inner(poly1, poly2, a=-1., b=1., norm='H1'):
    """computes the H^1 inner product of the polynomials p1 and p2 on the 
    intervall [a,b].

    Parameters
    ----------
    poly1 : numpy.poly1d class instance
        first polynomial.
    poly2 : numpy.poly1d class instance
        second polynomial.
    a : float
        left boundary of the integration inervall.
    b : float
        right boundary of the integration intervall.

    Returns
    -------
    inner_product : float
        H^1 inner product of poly1 and poly2 on [a,b].

    """
    # indefinite integral of the product of the polynomials for L^2-Term
    integral_p = np.polyint(poly1 * poly2)
    L2_term = integral_p(b) - integral_p(a) 
    inner_product = L2_term

    if (norm == 'H1' or norm == 'H2'):

        # get derivatives for H^1-Term
        derivative_p1 = np.polyder(poly1)
        derivative_p2 = np.polyder(poly2)

        # indefinite integral of the product of the derivatives for the H^1-Term
        integral_derivative_p = np.polyint(derivative_p1 * derivative_p2)
        
        H1_term = integral_derivative_p(b) - integral_derivative_p(a)
        inner_product += + H1_term
    
    if norm == 'H2':
        # get second derivatives for H^2-Term
        sec_derivative_p1 = np.polyder(derivative_p1)
        sec_derivative_p2 = np.polyder(derivative_p2)
        
        # indefinite integral of the product of the derivatives for the H^1_0-Term
        integral_sec_derivative_p = np.polyint(sec_derivative_p1 * sec_derivative_p2)
        
        H2_term = integral_sec_derivative_p(b) - integral_sec_derivative_p(a)
        inner_product += + H2_term
        

    return inner_product


def monomial(degree):
    """returns the 1d monomial x^k as a numpy.poly1d function.

    Parameters
    ----------
    degree : int

    Returns
    -------
    mon : numpy.poly1d class instance
        monomial of degree 'degree'

    """
    mon = np.poly1d([1] + [0] * degree)
    return mon


def gramian(degree,a=-1.,b=1.,norm='H1'):
    """builds the gramian matrix of the monomials x^k up to 'degree' w.r.t
    the H^1 inner product on [a,b]

    Parameters
    ----------
    degree : int
        maximal monomial degree.
    a : float
        left boundary of the integration inervall.
    b : float
        right boundary of the integration intervall.

    Returns
    -------
    gram : np.array
        gram matrix w.r.t H^1 inner product. Query ij is the inner product
        of x^i and x^j.

    """
    dim = degree # + 1
    gram = np.empty((dim, dim), dtype=np.float)

    for j in range(dim):
        for k in range(j + 1):
            gram[j, k] = gram[k, j] = poly_inner(monomial(j), monomial(k),a,b,norm)

    return gram


def coeffs(degree,a=-1.,b=1.,norm='H1'):
    """computes the coefficients of the H^1 orthonormal polynomials up to
    degree 'degree' on [a,b] by Gram-Schmidt orthogonalization.
    # Gram-Schmidt orthogonalization process
    # https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    # Orthogonal Polynomials
    # https://www.sydney.edu.au/science/chemistry/~mjtj/CHEM3117/Resources/poly_etc.pdf

    # Sobolev H1 Spaces
    # https://www.mat.tuhh.de/veranstaltungen/isem18/pdf/Lecture04.pdf
    # https://pnp.mathematik.uni-stuttgart.de/iadm/Weidl/fa-ws04/Suslina_Sobolevraeume.pdf

    Parameters
    ----------
    degree : int
        maximum degree.
    a : float, optional
        left boundary of integration interval. The default is 0..
    b : float, optional
        right boundary of integration interval. The default is 0.5*np.pi.

    Returns
    -------
    coefficients : np.array
        coefficient matrix. Query ij is the coefficient of x^j to the i-th
        H^1 orthonormal basis function.

    """
    gram = gramian(degree,a,b,norm)
    # print('cond. of gram matrix is ', np.linalg.cond(gram))
    #TODO: The gram matrix is relative to a Hilbert matrix which is very ill conditioned, 
    # this step here might be arbitrary inaccurate

    # print("COND : ",np.linalg.cond(gram))
    #exit()

    L = np.linalg.cholesky(gram)
    I = np.eye(degree)
    
    coefficients = solve_triangular(L, I, lower=True)

    return coefficients

def coeffs_grad(degree,a=-1.,b=1.,norm='H1'):
    """computes the coefficients of the derivatives of the H^1 orthonormal 
    polynomials up to degree 'degree' on [a,b] by Gram-Schmidt orthogonalization.

    Parameters
    ----------
    degree : int
        maximum degree.
    a : float, optional
        left boundary of integration intervall. The default is 0..
    b : float, optional
        right boundary of integration interval. The default is 0.5*np.pi.

    Returns
    -------
    coefficients : np.array
        coefficient matrix. Query ij is the coefficient of x^j to the i-th
        H^1 orthonormal basis function.

    """
    coeff = coeffs(degree,a,b,norm)
    for j in range(coeff.shape[1]-1):
        coeff[:,j] = (j+1)*coeff[:,j+1]
    coeff[:,-1] = 0*coeff[:,-1]
    return coeff



class orthpoly_basis(object):   

    # TODO: allow different domains in different dimensions
    def __init__(self, degrees,domain=[-1.,1.],norm='H1'):
        assert norm in ['L2','H1','H2']
        assert len(domain) == 2

        self.d = len(degrees) # order of tensor
        self.degs = degrees
        self.a = domain[0]
        self.b = domain[1]
        self.norm = norm

        self.coeffs = []
        for k in range(self.d):
            self.coeffs.append(lb.tensor(coeffs(degrees[k],a=self.a,b=self.b,norm=self.norm)))
        self.coeffs_grad = []
        for k in range(self.d):
            self.coeffs_grad.append(lb.tensor(coeffs_grad(degrees[k],a=self.a,b=self.b,norm=self.norm)))

    def __call__(self, x):
        """lifts the inputs to feature space.

        Parameters
        ----------
        x : lb.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of lb.tensor
            inputs lifted to feature space defined by the feature and
            basis_coeffs attributes. 
            Query [i][jk] is the k-th basis function evaluated at the j-th sample's
            i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []

        for k in range(self.d):
            exponents = lb.arange(0,self.degs[k],1, dtype=lb.float) 
            embedded_data.append(x[:, k, None] ** exponents)
            if self.coeffs is not None:
                embedded_data[k] = lb.einsum('oi, bi -> bo', self.coeffs[k], embedded_data[k])
        return embedded_data

    def grad(self, x):
        """lifts the inputs to feature-derivative space.

        Parameters
        ----------
        input_data : lb.tensor
            batched inputs of size (batch_size,input_dim)

        Returns
        -------
        embedded_data : list of lb.tensor
            inputs lifted to feature-derivative space defined by the feature and
            grad_coeffs attributes. 
            Query Query [i][jk] is the first derivative of the k-th basis function evaluated 
            at the j-th sample's i-th component.

        """
        assert x.shape[1] == self.d
        embedded_data = []

        for k in range(self.d):
            exponents = lb.arange(0,self.degs[k],1, dtype=lb.float)
            embedded_data.append(x[:, k, None] ** exponents)
            if self.coeffs is not None:
                embedded_data[k] = lb.einsum('oi, bi -> bo', self.coeffs_grad[k], embedded_data[k])
        return embedded_data

if __name__ == '__main__':
    coefs = coeffs(4,a=-1.,b=1.,norm='H2')
    print(coefs)

    poly = orthpoly_basis([5,5,5],norm='H2')
    x = lb.random.rand(10,3)
    print(poly(x)[1])
