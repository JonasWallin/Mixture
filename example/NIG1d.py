'''
Testing if the model can recover the parameter for regular NIG density
Created on May 1, 2016

@author: jonaswallin
'''

from Mixture.density import NIG
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
n  = 1000


if __name__ == "__main__":
    simObj = NIG(paramvec = [1.1, 2.12, 1., 2.])
    Y = simObj.simulate(n = n)
    plt.hist(Y, 200,normed=True, histtype='stepfilled', alpha=0.2)
    x_ =np.linspace(np.min(Y),np.max(Y), num = 1000)
    logf = simObj(y = x_)
    plt.plot(x_, np.exp(logf), color = 'blue')
    simObj.set_data(Y)
    def f(x):
        lik =  - np.sum(simObj(paramvec= x))
        if np.isnan(lik):
            return np.Inf
        return lik
    optim = sp.optimize.minimize(f, np.zeros(4),method='CG')
    print(optim)
    #plt.show()
    print(f(optim.x))
    simObj.set_param_vec(optim.x)
    logf = simObj(y = x_)
    plt.plot(x_, np.exp(logf), color = 'red')
    plt.show()