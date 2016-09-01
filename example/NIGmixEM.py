'''
Testing if the model can recover the parameter for multi univariate
regular NIG density
Created on May 1, 2016

@author: jonaswallin
'''

from Mixture.density import NIG, mNIG
from Mixture import mixOneDims
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import numpy.random as npr
n  = 100000


if __name__ == "__main__":
    simObj  = mNIG(paramvec = np.array([1.1, 2.12, 1., 2.,0, 2.12, .1, .1]))
    simObj2 = mNIG(paramvec = np.array([ -1.1, -2.12, .4, .5,-4, 2.12, -2, -.4]))
    #Y = simObj.simulate(n = n)
    
    
    mixObj = mixOneDims(K = 2, d = 2)
    mixObj.set_densites([simObj, simObj2])
    x_true = np.array([.5, 2.1, 2.12, 1., 2.,-4, .12, .1, .1,
                                   -1.1, -2.12, .4, .5,-1, .12, -1, -.4])
    mixObj.set_param_vec(x_true)
    
    
    Y = mixObj.sample(n = n)
    mixObj.set_data(Y)
    x_1 = np.linspace(np.min(Y[:,0]),np.max(Y[:,0]), num = 1000)
    x_2 = np.linspace(np.min(Y[:,1]),np.max(Y[:,1]), num = 1000)
   

    def f(x):
        lik =  - np.sum(pik*mixObj.dens_componentwise(np.hstack((0, x))))
        if np.isnan(lik):
            return np.inf
        return lik
    
    x0 = npr.randn(4*2*2)
    for iter_ in range(20):
        # E step
        pik = mixObj.weights(log=False)
        pi_k = np.sum(np.sum(pik, axis=1), axis=1)

        # M step
        #x = sp.optimize.fmin_cg(f, x0, epsilon=1e-4, maxiter=3)
        x = sp.optimize.fmin_powell(f, x0, maxiter=5)
        alpha = np.log(pi_k[1:]) - np.log(pi_k[0])
        #print(optim)
        mixObj.set_param_vec(np.hstack((alpha.ravel(), x)))
        print "Total log likelihood: {}".format(np.sum(mixObj()))
        x0 = x
    pi_k /= np.sum(pi_k, axis=0)
    allocations = mixObj.sample_allocations() #np.argmax(pik, axis=0)
    dens1 = mixObj.density_1d(dim = 0, y = x_1)
    dens2 = mixObj.density_1d(dim = 1, y = x_2)
    fig, axarr = plt.subplots(2, 2)
    comp_colors = ['blue', 'green']
    axarr[0, 0].hist(Y[:,0], 200, normed=True, histtype='stepfilled', alpha=0.2)
    axarr[0, 0].plot(x_1, np.exp(dens1), color = 'red', LineWidth=2)
    for i, (dens, pi_kk, col) in enumerate(zip(mixObj.dens, pi_k, comp_colors)):
        axarr[0, 0].plot(x_1, pi_kk*dens.dens_d(dim=0, y=x_1, log_=False), color=col)
        axarr[0, 1].hist(Y[allocations[:, 0] == i, 0], np.linspace(x_1[0], x_1[-1], 200), normed=True, color=col, alpha=0.4)
        axarr[0, 1].plot(x_1, dens.dens_d(dim=0, y=x_1, log_=False), color=col)
    axarr[1, 0].hist(Y[:,1], 200, normed=True, histtype='stepfilled', alpha=0.2) 
    axarr[1, 0].plot(x_2, np.exp(dens2), color = 'red', LineWidth=2)
    for i, (dens, pi_kk, col) in enumerate(zip(mixObj.dens, pi_k, comp_colors)):
        axarr[1, 0].plot(x_2, pi_kk*dens.dens_d(dim=1, y=x_2, log_=False), color=col)
        axarr[1, 1].hist(Y[allocations[:, 1] == i, 1], np.linspace(x_2[0], x_2[-1], 200), normed=True, color=col, alpha=0.4)
        axarr[1, 1].plot(x_2, dens.dens_d(dim=1, y=x_2, log_=False), color=col)
    plt.show()

