'''
testing speedup of code
Created on Sep 17, 2016

@author: jonaswallin
'''
from Mixture.density import mNIG
from Mixture.density.purepython import mNIG as pmNIG
from Mixture import mixOneDims
import numpy as np
import numpy.random as npr
import timeit

# most speed here is used startup (iteration = 500, n = 1000)
# Cython:
#    2000    0.152    0.000    0.268    0.000 NIG.py:82(EV)
#    2000    0.098    0.000    0.145    0.000 NIG.py:39(dens)
#    2000    0.051    0.000    0.051    0.000 {Mixture.util.cython_Bessel.Bessel0approx}
#    2000    0.037    0.000    0.037    0.000 {Mixture.util.cython_Bessel.Bessel1approx}
  
# Pure Python:
#     2000    1.201    0.001    1.264    0.001 NIG.py:208(EV)
#     2000    1.195    0.001    1.201    0.001 NIG.py:248(dens)

# Pure Python, no precompute:
#     2000    2.322    0.001    2.387    0.001 NIG.py:208(EV)
#     2000    1.205    0.001    1.211    0.001 NIG.py:248(dens)

npr.seed(10)

def speed_python(pure_python=False, precompute = True): 
    K = 2
    d = 2
    iteration = 500
    mixObj = mixOneDims(K=K, d=d)
    
    if pure_python:
        mixObj.set_densites([pmNIG(d=d) for k in range(K)])  # @UnusedVariable
    else:
        mixObj.set_densites([mNIG(d=d) for k in range(K)])  # @UnusedVariable
    paramMat_true = [np.array([[1.1, 1.12, 0.1, 0],
                                 [-1,  0,2 , -4] ]),
                np.array([[-2, 0, 0.3, 0],
                                 [1,  0, 2 , -4] ])]
    alpha_true = [0]
    mixObj.set_paramMat(alpha = alpha_true,paramMat = paramMat_true)
    
    
    Y = mixObj.sample(n = 1000)
    
    
   
    mixObj.set_data(Y)
    
    paramMat = [npr.randn(2,4),npr.randn(2,4)]
    paramMat[0][0,0] = 1.1 
    paramMat[1][0,0] = -2 
    alpha = np.array(alpha_true)
    for i in range(iteration):  # @UnusedVariable
        p, alpha, paramMat = mixObj.EMstep(alpha = alpha, paramMat = paramMat , precompute = precompute)  # @UnusedVariable


if __name__ == "__main__":
    
    