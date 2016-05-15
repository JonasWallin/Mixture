'''
Testing NIG on Flow data
Created on May 1, 2016

@author: jonaswallin
'''

from Mixture.density import NIG, mNIG
from Mixture import mixOneDims
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numpy.random as npr

K = 4
if __name__ == "__main__":
    
    Y = np.loadtxt("../data/dat1.txt")

    Y_red = Y[:,[0,3]]
    
    
    #fig, axarr = plt.subplots(2, 1)
    #axarr[0].hist(Y[:,0], 200,normed=True, histtype='stepfilled', alpha=0.2)
    #axarr[0].plot(x_1, np.exp(dens1), color = 'red')
    #axarr[1].hist(Y[:,3], 200,normed=True, histtype='stepfilled', alpha=0.2) 
    #axarr[1].plot(x_2, np.exp(dens2), color = 'red')
    #plt.show()
    d = Y_red.shape[1]
    simObjs  = [mNIG(paramvec = 100000*npr.randn(d*4)) for k in range(K)]
    mixObj = mixOneDims(K = K, d = Y_red.shape[1])
    mixObj.set_densites(simObjs)
    mixObj.set_data(Y_red)

    def f(x):
        lik =  - np.sum(mixObj(x))
        if np.isnan(lik):
            return np.Inf
        return lik
    
    x0 = np.array([ 266.74328189,  197.27638216,  198.06260285,  197.71155534,
        197.44634898,  197.08434816,  197.68358472,  198.06329792,
        198.09895742,  197.44734574,  197.39679396,    0.3874771 ,
          0.39292952,    5.24902432,   -1.21524397,    0.40066076,
          0.56159078,    1.3224249 ,   -2.71225716,  196.82375815,
        197.9560146 ,  197.56184291,  196.62450118,  198.26623712,
        198.23906905,  198.45845469,  197.8798628 ,  197.80798558,
        196.68550239,  197.2746013 ,  198.20249954,  198.03895281,
        197.59264208,  197.60377769,  197.38745821])
    mixObj(x0)
    x = sp.optimize.fmin_powell(f, x0,disp=True )
    
    
    x_1 = np.linspace(np.min(mixObj.y[:,0]),np.max(mixObj.y[:,0]), num = 1000)
    x_2 = np.linspace(np.min(mixObj.y[:,1]),np.max(mixObj.y[:,1]), num = 1000)
    mixObj.set_param_vec(x)
    dens1 = mixObj.density_1d(dim = 0, y = x_1)
    dens2 = mixObj.density_1d(dim = 1, y = x_2)
    fig, axarr = plt.subplots(2, 1)
    axarr[0].hist(mixObj.y[:,0], 200,normed=True, histtype='stepfilled', alpha=0.2)
    axarr[0].plot(x_1, np.exp(dens1), color = 'red')
    axarr[1].hist(mixObj.y[:,1], 200,normed=True, histtype='stepfilled', alpha=0.2) 
    axarr[1].plot(x_2, np.exp(dens2), color = 'red')  
    plt.show()