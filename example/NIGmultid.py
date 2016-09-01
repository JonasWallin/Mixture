'''
Testing if the model can recover the parameter for multi univariate
regular NIG density
Created on May 1, 2016

@author: jonaswallin
'''

from Mixture.density import NIG, mNIG
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
n  = 1000


if __name__ == "__main__":
    simObj = NIG(paramvec = [1.1, 2.12, 1., 2.])
    Y_1 = simObj.simulate(n = n)
    simObj2 = NIG(paramvec = [0, 2.12, .1, .1])
    Y_2 = simObj2.simulate(n = n)
    f, axarr = plt.subplots(2, 1)
    axarr[0].hist(Y_1, 200,normed=True, histtype='stepfilled', alpha=0.2)
    axarr[1].hist(Y_2, 200,normed=True, histtype='stepfilled', alpha=0.2)
    #plt.show()

    multiObj = mNIG(d = 2)
    multiObj.set_data(np.vstack((Y_1, Y_2)).transpose())


    def f(x):
        lik =  - np.sum(multiObj(paramvec= x))
        if np.isnan(lik):
            return np.Inf
        return lik

    optim = sp.optimize.minimize(f, np.zeros(8),method='CG')
    multiObj.set_param_vec(optim.x)
    x_1 =np.linspace(np.min(Y_1),np.max(Y_1), num = 1000)
    x_2 =np.linspace(np.min(Y_2),np.max(Y_2), num = 1000)
    x = np.vstack((x_1, x_2)).transpose()
    dens = multiObj.dens_dim(y = x, log_ = False)
    axarr[0].plot(x_1, dens[:, 0])
    axarr[1].plot(x_2, dens[:, 1])
    plt.show()