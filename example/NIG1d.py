'''
Testing if the model can recover the parameter for regular NIG density
Created on May 1, 2016

@author: jonaswallin
'''

from Mixture.density import NIG
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
n  = 6000


if __name__ == "__main__":
    simObj = NIG(paramvec = [0,0,0,0,0])
    Y = simObj.simulate(n = n)
    plt.hist(Y, 200,normed=True, histtype='stepfilled', alpha=0.2)
    x_ =np.linspace(np.min(Y),np.max(Y), num = 1000)
    logf = simObj(y = x_)
    plt.plot(x_, np.exp(logf))
    #TODO see that mean variance is equal
    #plot check
    f = lambda x: np.exp(simObj(y = x) )
    res=  sp.integrate.quad(f, -np.inf, np.inf)
    print(res)
    plt.show()
    