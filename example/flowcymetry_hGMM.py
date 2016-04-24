'''
Created on Jul 10, 2014

@author: jonaswallin
'''
from __future__ import division
import numpy as np
import BayesFlow as bm
import os
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from Mixture import swarm
from Mixture.Bayesflow  import GMMoptim, merge
from mpi4py import MPI
def invlogit(alpha):
    """
        inverse logit
    """
        
    p = np.hstack((1.,np.exp(alpha.flatten())))
    p /= np.sum(p)
    return p.flatten()

majorFormatter = FormatStrFormatter('%.1e')

def plot(GMM, ax ,dim):
    data= GMM.data[:,dim]
    x = GMM.x
    cm = plt.get_cmap('gist_rainbow')
    ax.set_color_cycle([cm(1.*i/GMM.K) for i in range(GMM.K)])
    if len(dim) == 2:
        for k in range(K):
            plt.plot(data[x==k,0],data[x==k,1],'+',label='k = %d'%(k+1))



def plot_prob(ps, axs = None, colors = None, **figargs):
    """
        ps -  (J x K) probabililites for each class
        axs - (K x 1) the axes on which to plot
        colors     - (J x 3?) the colors for indv
        figargs    - extra arguments to the figures

    """
    J = ps.shape[0]
    K = ps.shape[1]

    if axs is None:
        fig, axs = plt.subplots(K, **figargs)

    if colors  is None:
        colors = 'black'
    for k in range(K):
        axs[k].scatter(range(1, J + 1),
                   ps[:,k],
                   color= colors)

        axs[k].axes.yaxis.set_ticks( [np.min(ps[:,k]),
                                    np.mean(ps[:,k]),
                                    np.max(ps[:,k])])
        #axs[k].set_yscale('log')
        axs[k].set_xlim([0.5, J + 1.5])
        axs[k].set_ylim([np.min(ps[:,k])*0.9, np.max(ps[:,k])*1.1])

        axs[k].axes.xaxis.set_ticks( [])
        axs[k].yaxis.set_major_formatter(majorFormatter)
        axs[k].yaxis.tick_right()


def mean_center(musin, axs = None, dim_scale = True, colors =None, **figargs):
    """
        mus        - (J x K x d) the mean of J indv, K classes, d dimensions
        axs - (K x 1) the axes on which to plot
        dim_scale  - scale everything dimension to [-1, 1]
        colors     - (K x 3?) the colors for each class
        figargs    - extra arguments to the figures

    """
    mus = np.copy(musin)
    J = mus.shape[0]
    K = mus.shape[1]
    d = mus.shape[2]
    if colors is None:
        cm = plt.get_cmap('gist_rainbow')
        colors = [cm(1. * i / K) for i in range(K)]
    if axs is None:
        fig, axs = plt.subplots(K, **figargs)


    for k in range(K):
        if dim_scale:
            mus[:,k,:] -= np.min(mus[:,k,:])
            mus[:,k,:] /= np.max(mus[:,k,:])

        for j in range(J):
            axs[k].plot(range(1,d + 1), mus[j, k,:], color = colors[k])      

        axs[k].plot([1, d],[.5, .5], color = 'grey' )

        if k == K -1:
            axs[k].axes.xaxis.set_ticks( range(1, d + 1))
    return axs

       
if __name__ == '__main__':
    silent = True
    comm = MPI.COMM_WORLD  # @UndefinedVariable
    rank = comm.Get_rank()
    np.random.seed(2 + rank)
    repeat = 3
    K = 3
    iteration = 10
    sim = 10
    plt.close('all')
    plt.ion()
    
    data = None
    if rank == 0:
        data = []
        for file_ in os.listdir("../data/flow_dataset/"):
            if file_.endswith(".dat"):
                data.append(np.ascontiguousarray(np.loadtxt("../data/flow_dataset/" + file_)))


    hGMM = bm.hierarical_mixture_mpi(K = K)
    hGMM.set_data(data)
    hGMM.set_logisticdata()
    hGMM.set_prior_param0()
    hGMM.update_GMM()
    hGMM.update_prior()
    hGMM.set_p_labelswitch(1.)
    n = len(hGMM.GMMs)
    for i in range(n):
        if rank == 0:
            print(' i = {0}/{1}'.format(i, n))
        mix =  hGMM.GMMs[i]
        mixOptim = GMMoptim(mix)
        swarm(mixOptim, iteration = iteration, silent = True)

    mergeObj = merge.HGMMMerge(hGMM)

    for j in range(repeat):
        mergeObj.startup()
        mergeObj.run(1e-7)
        mergeObj.reset()
        if rank == 0:
            print('*********************************')
            print('PRE RUN')
            print('logit^-1(mu_alpha) = {}'.format(invlogit(hGMM.alpha_prior.beta_mu)))
            print('diag(Sigmas_alpha) = {}'.format(np.diag(hGMM.alpha_prior.Sigma)))
            print('beta_alpha        = {}'.format(hGMM.alpha_prior.beta_mu))
        
        for i in range(sim):
            if rank == 0:
                if not silent:
                    print(' sim = {}'.format(i))
            hGMM.sample()
        if rank == 0:
            print('POST RUN')
            print('logit^-1(mu_alpha) = {}'.format(invlogit(hGMM.alpha_prior.beta_mu)))
            print('diag(Sigmas_alpha) = {}'.format(np.diag(hGMM.alpha_prior.Sigma)))
            print('beta_alpha        = {}'.format(hGMM.alpha_prior.beta_mu))
    
    
            
    fig = plt.figure()
    fig.subplots_adjust(hspace=.7)

    for i in range(n):
        ax = fig.add_subplot(np.ceil(n/2),2,i + 1)
        plot(hGMM.GMMs[i], ax ,[0,1])
        ax.set_title('GMM = {}'.format(i))
        plt.savefig('GMM_{}.png'.format(rank))

    mus = hGMM.get_mus()
    ps  = hGMM.get_ps()
    sigma = hGMM.get_Sigmas()
    sigma_d = np.zeros_like(mus)
    if rank == 0:
        for j in range(sigma.shape[0]):
            for k in range(sigma.shape[1]):
                sigma_d[j, k, :] = np.diag(sigma[j, k, :])
        
        fig, axs = plt.subplots(hGMM.K, 3, figsize=(15,15))
        fig.subplots_adjust(hspace=.7)
        mean_center(mus, axs[:,0], dim_scale = True)
        mean_center(sigma_d, axs[:,1], dim_scale = False)
        plot_prob(ps, axs[:,2])
        plt.savefig('MEAN_analysis.png')
        #plt.show()