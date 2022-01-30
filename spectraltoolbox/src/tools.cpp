/*
# This file is part of SpectralToolbox.
#
# SpectralToolbox is free software: you can redistribute it and/or modify
# it under the terms of the LGNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SpectralToolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LGNU Lesser General Public License for more details.
#
# You should have received a copy of the LGNU Lesser General Public License
# along with SpectralToolbox.  If not, see <http://www.gnu.org/licenses/>.
#
# DTU UQ Library
# Copyright (C) 2014-2015 The Technical University of Denmark
# Scientific Computing Section
# Department of Applied Mathematics and Computer Science
#
# Copyright (C) 2015-2016 Massachusetts Institute of Technology
# Uncertainty Quantification group
# Department of Aeronautics and Astronautics
#
# Author: Daniele Bigoni
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/ndarraytypes.h>
#include <math.h>

#if PY_MAJOR_VERSION >= 3
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#else
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
#endif

using namespace std;

static PyObject *polymodError;

/* .... C vector utility functions ..................*/
PyArrayObject *pyvector(PyObject *objin);
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin);
int  not_doublevector(PyArrayObject *vec);

/* .... C matrix utility functions ..................*/
PyArrayObject *pymatrix(PyObject *objin);
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin);
double **ptrvector(long n);
void free_Carrayptrs(double **v);
int  not_doublematrix(PyArrayObject *mat);

/* .... C 2D int array utility functions ..................*/
PyArrayObject *pyint2Darray(PyObject *objin);
int **pyint2Darray_to_Carrayptrs(PyArrayObject *arrayin);
int **ptrintvector(long n);
void free_Cint2Darrayptrs(int **v);
int  not_int2Darray(PyArrayObject *mat);

/* polymod functions declaration */
static PyObject* Py_polyeval (PyObject *self, PyObject *args);
static PyObject* Py_monomials (PyObject *self, PyObject *args);

/* PyPOLYMOD list of methods definition */
static PyMethodDef POLYMODMethods[] = {
  {"polyeval", Py_polyeval, METH_VARARGS, "Given a set of points and the N+1 recursion coefficients, evaluate the polynomial of order N, normalized."},
  {"monomials", Py_monomials, METH_VARARGS, "Given the N+1 recursion coefficients, evaluate the monomials coefficients."},
  {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* #### Vector Utility functions ######################### */

/* ==== Make a Python Array Obj. from a PyObject, ================
     generates a double vector w/ contiguous memory which may be a new allocation if
     the original was not a double type or contiguous 
  !! Must DECREF the object returned from this routine unless it is returned to the
     caller of this routines caller using return PyArray_Return(obj) or
     PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pyvector(PyObject *objin)  {
	return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, 1,1);
}
/* ==== Create 1D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.             */
double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
  return (double *) PyArray_DATA(arrayin);  /* pointer to arrayin data as double */
}
/* ==== Check that PyArrayObject is a double (Float) type and a vector ==============
    return 1 if an error and raise exception */ 
int  not_doublevector(PyArrayObject *vec)  {
  if (PyArray_DESCR(vec)->type_num != NPY_DOUBLE || PyArray_NDIM(vec) != 1)  {
    PyErr_SetString(PyExc_ValueError,
                    "In not_doublevector: array must be of type Float and 1 dimensional (n).");
    return 1;  }
  return 0;
}

/* #### Matrix Utility functions ######################### */

/* ==== Make a Python Array Obj. from a PyObject, ================
     generates a double matrix w/ contiguous memory which may be a new allocation if
     the original was not a double type or contiguous 
  !! Must DECREF the object returned from this routine unless it is returned to the
     caller of this routines caller using return PyArray_Return(obj) or
     PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pymatrix(PyObject *objin)  {
	return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
		NPY_DOUBLE, 2,2);
}
/* ==== Create Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
	double **c, *a;
	int i,n,m;
	
	n=PyArray_DIMS(arrayin)[0];
	m=PyArray_DIMS(arrayin)[1];
	c=ptrvector(n);
	a=(double *) PyArray_DATA(arrayin);  /* pointer to arrayin data as double */
	for ( i=0; i<n; i++)  {
		c[i]=a+i*m;  }
	return c;
}
/* ==== Allocate a double *vector (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(double ** )                  */
double **ptrvector(long n)  {
	double **v;
	v=(double **)malloc((size_t) (n*sizeof(double)));
	if (!v)   {
		printf("In **ptrvector. Allocation of memory for double array failed.");
		exit(0);  }
	return v;
}
/* ==== Free a double *vector (vec of pointers) ========================== */ 
void free_Carrayptrs(double **v)  {
	free((char*) v);
}
/* ==== Check that PyArrayObject is a double (Float) type and a matrix ==============
    return 1 if an error and raise exception */ 
int  not_doublematrix(PyArrayObject *mat)  {
  if (PyArray_DESCR(mat)->type_num != NPY_DOUBLE || PyArray_NDIM(mat) != 2)  {
    PyErr_SetString(PyExc_ValueError,
                    "In not_doublematrix: array must be of type Float and 2 dimensional (n x m).");
    return 1;  }
  return 0;
}

/* #### Integer Array Utility functions ######################### */

/* ==== Make a Python int Array Obj. from a PyObject, ================
     generates a 2D integer array w/ contiguous memory which may be a new allocation if
     the original was not an integer type or contiguous 
  !! Must DECREF the object returned from this routine unless it is returned to the
     caller of this routines caller using return PyArray_Return(obj) or
     PyArray_BuildValue with the "N" construct   !!!
*/
PyArrayObject *pyint2Darray(PyObject *objin)  {
	return (PyArrayObject *) PyArray_ContiguousFromObject(objin,
		NPY_LONG, 2,2);
}
/* ==== Create integer 2D Carray from PyArray ======================
    Assumes PyArray is contiguous in memory.
    Memory is allocated!                                    */
int **pyint2Darray_to_Carrayptrs(PyArrayObject *arrayin)  {
	int **c, *a;
	int i,n,m;
	
	n=PyArray_DIMS(arrayin)[0];
	m=PyArray_DIMS(arrayin)[1];
	c=ptrintvector(n);
	a=(int *) PyArray_DATA(arrayin);  /* pointer to arrayin data as int */
	for ( i=0; i<n; i++)  {
		c[i]=a+i*m;  }
	return c;
}
/* ==== Allocate a a *int (vec of pointers) ======================
    Memory is Allocated!  See void free_Carray(int ** )                  */
int **ptrintvector(long n)  {
	int **v;
	v=(int **)malloc((size_t) (n*sizeof(int)));
	if (!v)   {
		printf("In **ptrintvector. Allocation of memory for int array failed.");
		exit(0);  }
	return v;
}
/* ==== Free an int *vector (vec of pointers) ========================== */ 
void free_Cint2Darrayptrs(int **v)  {
	free((char*) v);
}
/* ==== Check that PyArrayObject is an int (integer) type and a 2D array ==============
    return 1 if an error and raise exception
    Note:  Use NY_LONG for NumPy integer array, not NP_INT      */ 
int  not_int2Darray(PyArrayObject *mat)  {
  if (PyArray_DESCR(mat)->type_num != NPY_LONG || PyArray_NDIM(mat) != 2)  {
    PyErr_SetString(PyExc_ValueError,
                    "In not_int2Darray: array must be of type int and 2 dimensional (n x m).");
    return 1;  }
  return 0;
}


/* Init Package function definition */
MOD_INIT(polymod)
{
    PyObject *m;

#if PY_MAJOR_VERSION >= 3
    static struct PyModuleDef moduledef = {
      PyModuleDef_HEAD_INIT,
      "polymod",           /* m_name */
      "Low-level routines of SpectralToolbox",  /* m_doc */
      -1,                  /* m_size */
      POLYMODMethods,      /* m_methods */
      NULL,                /* m_reload */
      NULL,                /* m_traverse */
      NULL,                /* m_clear */
      NULL,                /* m_free */
    };
#endif
    
#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;
#else
    m = Py_InitModule3("polymod", POLYMODMethods, "Low-level routines of SpectralToolbox");
    if (m == NULL) return;
#endif

    import_array();

    polymodError = PyErr_NewException(const_cast<char *>("polymod.error"), NULL, NULL);
    Py_INCREF(polymodError);
    PyModule_AddObject(m, "error", polymodError);

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

/* polymod functions definitions */
static PyObject*
Py_polyeval(PyObject *self, PyObject *args)
{
  /** 
      Evaluate polynomials of the form:
      P_{n+1}(x) = (x - alpha_n)P_{n}(x) - beta_n P_{n-1}(x)
   **/

  /* VARIABLES */
  // INPUT
  int n, normalized; // normalized = 0 False, normalized = 1 True
  PyObject *py_alpha_arg=NULL, *py_beta_arg=NULL; // Recursion coefficients
  PyObject *py_rs_arg=NULL; 	// Evaluation points
  PyArrayObject *py_alpha=NULL, *py_beta=NULL; // Recursion coefficients
  PyArrayObject *py_rs=NULL; 	// Evaluation points
  double *alpha, *beta_j, *beta_jm1, *rs, sqbeta_j, sqbeta_jm1;
  // OUTPUT
  PyArrayObject *py_out;
  // INTERNAL
  npy_intp *dims = new npy_intp[1];
  double *old1=NULL, *old2=NULL, *outinner, *tmp;

  // if (!PyArg_ParseTuple(args, "O!iO!O!i", &PyArray_Type, &py_rs, 
  //                       &n, &PyArray_Type, &py_alpha, &PyArray_Type, &py_beta,
  //                       &normalized))
  //   return NULL;
  // if (NULL == py_rs) return NULL;
  // if (NULL == py_alpha) return NULL;
  // if (NULL == py_beta) return NULL;
  if (!PyArg_ParseTuple(args, "OiOOi", &py_rs_arg, &n, 
			&py_alpha_arg, &py_beta_arg, &normalized))
    return NULL;

  py_rs = (PyArrayObject *)PyArray_FROM_OTF(py_rs_arg, NPY_DOUBLE, 
                                            NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (py_rs == NULL) return NULL;
  py_alpha = (PyArrayObject *)PyArray_FROM_OTF(py_alpha_arg, NPY_DOUBLE, 
                                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (py_alpha == NULL) goto fail;
  py_beta = (PyArrayObject *)PyArray_FROM_OTF(py_beta_arg, NPY_DOUBLE, 
                                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (py_beta == NULL) goto fail;
  
  if (n < 0)
    PyErr_SetString(polymodError, "The order must be >= 0.");
  if ((PyArray_NDIM(py_rs) > 1) || (PyArray_NDIM(py_alpha) > 1) || (PyArray_NDIM(py_beta) > 1))
    PyErr_SetString(polymodError, "The array provided must be 1 dimensional.");
  if ((PyArray_DIMS(py_alpha)[0] < n) or (PyArray_DIMS(py_beta)[0] < n))
    PyErr_SetString(polymodError, "The recurrence coefficeints must be more than n.");

  dims[0] = PyArray_DIMS(py_rs)[0];

  // Allocate output variables
  py_out = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  // Allocate auxiliary space
  outinner = new double[dims[0]];
  
  // Use three term recursion to compute the polynomial
  if (n >= 0) {
    if (normalized == 0){
      for (int i = 0; i < dims[0]; i++) outinner[i] = 1.;
    } else {
      beta_j = (double *)PyArray_GETPTR1(py_beta, 0);
      sqbeta_j = sqrt(*beta_j);
      for (int i = 0; i < dims[0]; i++) outinner[i] = 1./sqbeta_j;
    }
  }
  if (n >= 1) {
    old1 = outinner;		// Copy pointer
    outinner = new double[dims[0]];
    if (normalized == 0){
      alpha = (double *)PyArray_GETPTR1(py_alpha, 0);
      for (int i = 0; i < dims[0]; i++) {
        rs = (double *)PyArray_GETPTR1(py_rs, i);
        outinner[i] = *rs - *alpha;
      }
    } else {
      alpha = (double *)PyArray_GETPTR1(py_alpha, 0);
      beta_j = (double *)PyArray_GETPTR1(py_beta, 1);
      sqbeta_j = sqrt(*beta_j);
      for (int i = 0; i < dims[0]; i++) {
        rs = (double *)PyArray_GETPTR1(py_rs, i);
        outinner[i] = (*rs - *alpha) * old1[i] / sqbeta_j;
      }
    }
  }
  if (n >= 2) {
    old2 = new double[dims[0]];
  }
  for (int j=2; j <= n; j++){
    // Swap pointers
    tmp = old2;
    old2 = old1;
    old1 = outinner;
    outinner = tmp;		// Here we put the new values (save in allocation)
    if (normalized == 0){
      alpha = (double *)PyArray_GETPTR1(py_alpha, j-1);
      beta_jm1 = (double *)PyArray_GETPTR1(py_beta, j-1);
      for (int i = 0; i < dims[0]; i++) {
        rs = (double *)PyArray_GETPTR1(py_rs, i);
        outinner[i] = (*rs - *alpha) * old1[i] - *beta_jm1 * old2[i];
      }
    } else {
      alpha = (double *)PyArray_GETPTR1(py_alpha, j-1);
      beta_j = (double *)PyArray_GETPTR1(py_beta, j);
      beta_jm1 = (double *)PyArray_GETPTR1(py_beta, j-1);
      sqbeta_j = sqrt( *beta_j );
      sqbeta_jm1 = sqrt( *beta_jm1 );
      for (int i = 0; i < dims[0]; i++) {
        rs = (double *)PyArray_GETPTR1(py_rs, i);
        outinner[i] = (*rs - *alpha) * old1[i] / sqbeta_j - 
          sqbeta_jm1 / sqbeta_j * old2[i];
      }
    }
  }

  // Free space and prepare output
  if (old1 != NULL) delete [] old1;
  if (old2 != NULL) delete [] old2;
  
  for (int i = 0; i < dims[0]; i++) {
    double *v = (double *)PyArray_GETPTR1(py_out, i);
    *v = outinner[i];
  }

  delete [] outinner;
  delete [] dims;
  
  Py_DECREF(py_rs);
  Py_DECREF(py_alpha);
  Py_DECREF(py_beta);
  return Py_BuildValue("N", py_out);

 fail:
  Py_XDECREF(py_rs);
  Py_XDECREF(py_alpha);
  Py_XDECREF(py_beta);
  return NULL;
}

static PyObject*
Py_monomials(PyObject *self, PyObject *args)
{
  /**
     Evaluate the monomials coefficients from the recursion coefficients
   **/

  /* VARIABLES */
  // INPUT
  int normalized; // normalized = 0 False, normalized = 1 True
  PyObject *py_alpha_arg=NULL, *py_beta_arg=NULL; // Recursion coefficients
  PyArrayObject *py_alpha=NULL, *py_beta=NULL; // Recursion coefficients
  double *alpha_m1, *beta, *beta_m1, sqbeta, sqbeta_m1;
  // OUTPUT
  PyArrayObject *py_out;
  // INTERNAL
  npy_intp *na = new npy_intp[1];
  npy_intp *nb = new npy_intp[1];
  npy_intp *dim = new npy_intp[1];
  double *old1=NULL, *old2=NULL, *outinner, *tmp;

  if (!PyArg_ParseTuple(args, "OOi", &py_alpha_arg, &py_beta_arg, &normalized))
    return NULL;

  py_alpha = (PyArrayObject *)PyArray_FROM_OTF(py_alpha_arg, NPY_DOUBLE, 
                                               NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (py_alpha == NULL) goto fail;
  py_beta = (PyArrayObject *)PyArray_FROM_OTF(py_beta_arg, NPY_DOUBLE, 
                                              NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  if (py_beta == NULL) goto fail;

  if ((PyArray_NDIM(py_alpha) > 1) || (PyArray_NDIM(py_beta) > 1))
    PyErr_SetString(polymodError, "Array alpha and beta provided must be 1 dimensional.");  
  na[0] = PyArray_DIMS(py_alpha)[0];
  nb[0] = PyArray_DIMS(py_beta)[0];
  if (na[0] != nb[0])
    PyErr_SetString(polymodError, "The size of alpha and beta must be the same");
  dim[0] = na[0];

  // Allocate output variables
  py_out = (PyArrayObject *)PyArray_SimpleNew(1, dim, NPY_DOUBLE);

  // Auxiliary space
  outinner = new double[dim[0]];
  for (int i=0; i < dim[0]; i++) outinner[i] = 0.;

  // Unroll three term recursion to compute coefficients
  if (dim[0] > 0){
    if (normalized == 0){ 
      outinner[0] = 1.;
    } else {
      beta = (double *)PyArray_GETPTR1(py_beta, 0);
      sqbeta = sqrt(*beta);
      outinner[0] = 1./sqbeta;
    }
  }

  if (dim[0] > 1){
    old1 = outinner;  
    outinner = new double[dim[0]];
    for (int i=0; i < dim[0]; i++) outinner[i] = 0.;
    if (normalized == 0){
      alpha_m1 = (double *)PyArray_GETPTR1(py_alpha, 0);
      outinner[0] = - *alpha_m1;
      outinner[1] = 1.;
    } else {
      alpha_m1 = (double *)PyArray_GETPTR1(py_alpha, 0);
      beta = (double *)PyArray_GETPTR1(py_beta, 1);
      sqbeta = sqrt(*beta);
      outinner[0] = - *alpha_m1 / sqbeta * old1[0];
      outinner[1] = old1[0] / sqbeta;
    }
  }

  if (dim[0] > 2){
    old2 = new double[dim[0]];
    for (int i=0; i < dim[0]; i++) old2[i] = 0.;
  }
  for (int j=2; j < dim[0]; j++){
    // Swap
    tmp = old2;
    old2 = old1;
    old1 = outinner;
    outinner = tmp;
    if (normalized == 0){
      alpha_m1 = (double *)PyArray_GETPTR1(py_alpha, j-1);
      beta_m1 = (double *)PyArray_GETPTR1(py_beta, j-1);
      outinner[0] = - *alpha_m1 * old1[0] - *beta_m1 * old2[0];
      for (int k=1; k <= j; k++)
	outinner[k] = old1[k-1] - *alpha_m1 * old1[k] - *beta_m1 * old2[k];
    } else {
      alpha_m1 = (double *)PyArray_GETPTR1(py_alpha, j-1);
      beta_m1 = (double *)PyArray_GETPTR1(py_beta, j-1);
      beta = (double *)PyArray_GETPTR1(py_beta, j);
      sqbeta_m1 = sqrt( *beta_m1 );
      sqbeta = sqrt( *beta );
      outinner[0] = ( - *alpha_m1 * old1[0] - sqbeta_m1 * old2[0] ) / sqbeta;
      for (int k=1; k <= j; k++)
	outinner[k] = (old1[k-1] - *alpha_m1 * old1[k] - sqbeta_m1 * old2[k]) / sqbeta;
    }
  }
  
  // Free space and prepare output
  if (old1 != NULL) delete [] old1;
  if (old2 != NULL) delete [] old2;
  
  for (int i = 0; i < dim[0]; i++) {
    double *v = (double *)PyArray_GETPTR1(py_out, i);
    *v = outinner[i];
  }

  delete [] outinner;
  delete [] dim;
  
  Py_DECREF(py_alpha);
  Py_DECREF(py_beta);
  return Py_BuildValue("N", py_out);

 fail:
  Py_XDECREF(py_alpha);
  Py_XDECREF(py_beta);
  return NULL;
}


