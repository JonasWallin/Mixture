'''
Testing if the model can recover the parameter for mixture of multi univariate
regular NIG density
Created on May 1, 2016

@author: jonaswallin
'''

from Mixture.density import mNIG
from Mixture import mixOneDims, SwarmOptimMixtured, swarm
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr

n = 10000


if __name__ == "__main__":
    # Setup
    simObj = mNIG(paramvec=np.array([1.1, 2.12, 1., 2., 0, 2.12, .1, .1]))
    simObj2 = mNIG(paramvec=np.array([-1.1, -2.12, .4, .5, -4, 2.12, -2, -.4]))

    mixObj = mixOneDims(K=2, d=2)
    mixObj.set_densites([simObj, simObj2])
    x_true = np.array([.5, 2.1, 2.12, 1., 2., -4, .12, .1, .1,
                       -1.1, -2.12, .4, .5, -1, .12, -1, -.4])
    mixObj.set_param_vec(x_true)
    Y = mixObj.sample(n=n)
    mixObj.set_data(Y)

    # Random initialization
    x0 = npr.randn(1+4*2*2)
    mixObj.set_param_vec(x0)
    so_mixObj = SwarmOptimMixtured(mixObj)

    # Optimization
    swarm(so_mixObj, mutate_iteration = 3, burst_iteration = 3)

    # Get parameters
    pi_k, _, _ = mixObj.get_paramMat()
    allocations = mixObj.sample_allocations()

    # Plot
    x_is = [np.linspace(np.min(Y[:, i]), np.max(Y[:, i]), num=1000) for i in range(mixObj.d)]
    dens_is = [mixObj.density_1d(dim=i, y=x_) for i, x_ in enumerate(x_is)]

    fig, axs = plt.subplots(mixObj.K+1, mixObj.d, sharex=True)
    comp_colors = ['blue', 'green']
    for dd in range(mixObj.d):
        axs[0, dd].hist(Y[:, dd], 200, normed=True, histtype='stepfilled', alpha=0.2)
        axs[0, dd].plot(x_is[dd], np.exp(dens_is[dd]), color='red', LineWidth=2)
        for i, (dens, pi_kk, col) in enumerate(zip(mixObj.dens, pi_k, comp_colors)):
            axs[0, dd].plot(x_is[dd], pi_kk*dens.dens_d(dim=dd, y=x_is[dd], log_=False), color=col)

    for k in range(mixObj.K):
        for dd in range(mixObj.d):
            axs[k+1, dd].hist(Y[allocations == k, dd], np.linspace(x_is[dd][0], x_is[dd][-1], 200), normed=True, color=comp_colors[k], alpha=0.4)
            axs[k+1, dd].plot(x_is[dd], mixObj.dens[k].dens_d(dim=dd, y=x_is[dd], log_=False), color=comp_colors[k])

plt.show()
