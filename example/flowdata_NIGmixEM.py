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

host = 'ke'
flowdirs = {'jw': '/Users/jonaswallin/Dropbox/articles/FlowCap/',
            'ke': '/Users/johnsson/Dropbox/FlowCap (2)/'}

if __name__ == "__main__":

    n = 10000
    K = 5
    dims = (7, 4, 3, 9)
    EM_iter = 5
    EMM_iter = 3
    d = np.shape(dims)[0]
    x0 = npr.randn(4*K*d)

    flowdir = flowdirs[host]
    files = ['data/labex/preprocessed/panel1_v3/week 51/donor 015panel 1 V1LX12012-12-17.001_tcells.npy',
             'data/labex/preprocessed/panel1_v3/week 51/donor 016panel 1 V1LX12012-12-18.001_tcells.npy']
    for iter_files in range(np.shape(files)[0]):
        datafile = flowdir + files[iter_files]
        Y = np.load(datafile).astype(np.float_)
        print "Y.shape = {}".format(Y.shape)
        Y = Y[np.random.choice(range(Y.shape[0]), n, replace=False), :][:, dims]
        Y = Y[(Y[:, 0] < np.percentile(Y[:, 0], 99)) &
              (Y[:, 1] < np.percentile(Y[:, 1], 99)) &
              (Y[:, 0] > -2000) & (Y[:, 1] > -2000), :]
        Y -= np.mean(Y, axis=0)
        Y /= np.std(Y, axis=0)

        mixObj = mixOneDims(K=K, d=d)
        mixObj.set_densites([mNIG(d=d) for k in range(K)])

        mixObj.set_data(Y)
        x_is = []
        dens_is = []
        for i in range(d):
            x_is.append( np.linspace(np.min(Y[:, i]), np.max(Y[:, i]), num=1000))
        #plt.hist2d(Y[:, 0], Y[:, 1], bins=200, norm=colors.LogNorm(), vmin=1)
        #plt.show()

        def f(x):
            lik = - np.sum(pik *
                           np.sum(mixObj.dens_componentwise(np.hstack((np.zeros(K-1), x))), axis=2))
            if np.isnan(lik):
                return np.inf
            return lik

        
        alpha = np.zeros(K-1)
        mixObj.set_param_vec(np.hstack((alpha.ravel(), x0)))
        for iter_ in range(EM_iter):
            # E step
            print('iter = {}'.format(iter_))
            pik = mixObj.weights(log=False)
            pi_k = np.sum(pik, axis=1)

            # M step
            #x = sp.optimize.fmin_cg(f, x0, epsilon=1e-4, maxiter=3)
            x = sp.optimize.fmin_powell(f, x0, maxiter=EMM_iter)
            alpha = np.log(pi_k[1:]) - np.log(pi_k[0])
            #print(optim)
            mixObj.set_param_vec(np.hstack((alpha.ravel(), x)))
            print "Total log likelihood: {}".format(np.sum(mixObj()))
            x0 = x
        print('done file = {}'.format(iter_files))
        pi_k /= np.sum(pi_k, axis=0)
        allocations = mixObj.sample_allocations()
        for i in range(d):
            dens_is.append(mixObj.density_1d(dim=i, y=x_is[i]))

        fig, axarr = plt.subplots(2, 2)
        comp_colors = ['blue', 'green', 'yellow', 'pink', 'grey', 'purple']
        axarr[0, 0].hist(Y[:, 0], 200, normed=True, histtype='stepfilled', alpha=0.2)
        axarr[0, 0].plot(x_is[0], np.exp(dens_is[0]), color='red', LineWidth=2)
        for i, (dens, pi_kk, col) in enumerate(zip(mixObj.dens, pi_k, comp_colors)):
            axarr[0, 0].plot(x_is[0], pi_kk*dens.dens_d(dim=0, y=x_is[0], log_=False), color=col)
            #axarr[0, 1].hist(Y[allocations[:, 0] == i, 0], np.linspace(x_1[0], x_1[-1], 200), normed=True, color=col, alpha=0.4)
            axarr[0, 1].hist(Y[allocations == i, 0], np.linspace(x_is[0][0], x_is[0][-1], 200), normed=True, color=col, alpha=0.4)
            axarr[0, 1].plot(x_is[0], dens.dens_d(dim=0, y=x_is[0], log_=False), color=col)
        axarr[1, 0].hist(Y[:, 1], 200, normed=True, histtype='stepfilled', alpha=0.2)
        axarr[1, 0].plot(x_is[1], np.exp(dens_is[1]), color='red', LineWidth=2)
        for i, (dens, pi_kk, col) in enumerate(zip(mixObj.dens, pi_k, comp_colors)):
            axarr[1, 0].plot(x_is[1], pi_kk*dens.dens_d(dim=1, y=x_is[1], log_=False), color=col)
            # axarr[1, 1].hist(Y[allocations[:, 1] == i, 1], np.linspace(x_2[0], x_2[-1], 200), normed=True, color=col, alpha=0.4)
            axarr[1, 1].hist(Y[allocations == i, 1], np.linspace(x_is[1][0], x_is[1][-1], 200), normed=True, color=col, alpha=0.4)
            axarr[1, 1].plot(x_is[1], dens.dens_d(dim=1, y=x_is[1], log_=False), color=col)

        fig, axs = plt.subplots(K+1, d, sharex=True)
        for dd in range(d):
            axs[0, dd].hist(Y[:, dd], 200, normed=True, histtype='stepfilled', alpha=0.2)
            axs[0, dd].plot(x_is[dd], np.exp(dens_is[dd]), color='red', LineWidth=2)
            for i, (dens, pi_kk, col) in enumerate(zip(mixObj.dens, pi_k, comp_colors)):
                axs[0, dd].plot(x_is[dd], pi_kk*dens.dens_d(dim=dd, y=x_is[dd], log_=False), color=col)

        for k in range(K):
            for dd in range(d):
                axs[k+1, dd].hist(Y[allocations == k, dd], np.linspace(x_is[dd][0], x_is[dd][-1], 200), normed=True, color=comp_colors[k], alpha=0.4)
                axs[k+1, dd].plot(x_is[dd], mixObj.dens[k].dens_d(dim=dd, y=x_is[dd], log_=False), color=comp_colors[k])

plt.show()