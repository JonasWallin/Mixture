
import numpy as np
cimport numpy as np
cimport cython
import cPickle as pickle



cdef extern void bessel1(const double* x,  double* res, const int n) nogil
cdef extern void bessel1order(const double* x,  double* res, const int n) nogil



@cython.boundscheck(False)
@cython.wraparound(False)
def Bessel1approx(np.ndarray[np.double_t, ndim=1, mode='c']  x):
    """
        Approximation of modifed bessel function of the second kind with,
        nu = 1.
        (returns zero above 1000)
    """
    
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] res = np.empty_like(x)
    bessel1(<double *>  x.data, <double *>  res.data, x.shape[0])
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def Bessel1approxOrdered(np.ndarray[np.double_t, ndim=1, mode='c']  x):
    """
        Approximation of modifed bessel function of the second kind with
        nu = 1.
        (returns zero above 1000)
        assumes x is orderd
    """
    
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] res = np.empty_like(x)
    bessel1order(<double *>  x.data, <double *>  res.data, x.shape[0])
    return res