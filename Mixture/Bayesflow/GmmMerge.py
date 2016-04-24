"""
    function for merging objects of Bayesflow Gaussian mixture models
"""

import numpy as np
import copy as cp
from logisticnormal import tMd
import scipy.optimize as spo
from mpi4py import MPI

class HGMMMerge(object):
    """
        Object used to merging classes in a hGMM obj
        if one wants non deafult inital guess one needs to set
        self.param, see startup
    """
    def __init__(self, hGMM, dist_type = 0):
        """
            setups the main object
            
            dist_type - distance compared between mixtures
                        0 : (deafult) compare on distnace between mus    
        """
        self.hGMM = hGMM
        self.param = None
        self.K = self.hGMM.GMMs[0].K
        self.index_start = 0
        self.dist_type   = dist_type
        self.comm = MPI.COMM_WORLD  # @UndefinedVariable
    
    
    def reset(self):
        
        self.param = None
        self.startup()
    
    def startup(self):
        """
            sets up the needed parameters
            returns index_start which tells which index should start with
        """
        rank = self.comm .Get_rank() 
        if rank == 0:
            
            if self.param is None:
                self.index_start = 1
                
                self.param = {'mus':   [], 
                              'sigmas':[],
                              'ps'    :[]}
                for k in range(self.K):
                    mu_    = np.atleast_2d(self.hGMM.GMMs[0].mu[k])
                    sigma_ = self.hGMM.GMMs[0].sigma[k][np.newaxis,:]
                    p_     = self.hGMM.GMMs[0].p[k]
                    self.param['mus'].append(cp.deepcopy(mu_ ))
                    self.param['sigmas'].append(cp.deepcopy(sigma_ ))
                    self.param['ps'].append(p_)
        


    def gather(self):
        """
            Gather the mu, sigma, p from all data
        """
        
        rank = self.comm .Get_rank()
        params = []
        for i in range(len(self.hGMM.GMMs)):
            param_ = {'mu':     self.hGMM.GMMs[i].mu,
                      'sigma':  self.hGMM.GMMs[i].sigma,
                      'p':      self.hGMM.GMMs[i].p,
                      'rank':   rank}
            params.append(param_)
            
        params = self.comm.gather(params, root=0)
        
        if rank == 0:
            self.params=[]
            for parList in params:
                for i, par in enumerate(parList):
                    if (par['rank'] != 0) or (i > 0) or (self.index_start == 0):
                        self.params.append(par)
    def run(self, shrinkage  =0.1):
        """
            runs the main loop
            
            shrinkage - two shrink the prior covariance towards the mean for alpha in
                        logistic normal
        """
        self.gather()
        
        #push all the data to rank one
        rank = self.comm .Get_rank()
        if rank == 0:
            mus    = self.param['mus']
            sigmas = self.param['sigmas']
            ps     = self.param['ps']
        
            for param in self.params:
                index_k = np.argsort(param['p'][:self.K])[::-1] #sort by probabilility, largest first
                
                mus_t    =  np.array([np.mean(mu, axis=0) for mu in  mus])
                sigmas_t =  np.array([np.mean(sigma, axis=0) for sigma in  sigmas])
                ps_t     =  np.array([np.mean(p) for p in ps])
                list_temp = [None for k in range(self.K)]   # @UnusedVariable
                
                
                for index in index_k: #looping over index
                    i, mus_t, sigmas_t, ps_t = self.dist(index, param, mus_t, sigmas_t, ps_t)
                    list_temp[index] = i 
                    
                list_temp = np.argsort(np.array(list_temp))
                mus = [np.vstack((mu,param['mu'][list_temp[i]])) for i,mu in enumerate(mus) ]
                sigmas = [np.vstack((sigma,
                      param['sigma'][list_temp[i]][np.newaxis,:])) for i, sigma in enumerate(sigmas)]
                ps = [np.vstack((p,param['p'][list_temp[i]])) for i, p in enumerate(ps) ]
        
            self.param['mus']    = mus
            self.param['sigmas'] = sigmas 
            self.param['ps']     = ps
        
            # if rank not equal to zero send in mus[1:], sigmas[1:], ps[1:]
            
        self.optim(shrinkage) 
    
    def optim(self, shrinkage = 0.1):
        """
            merges the classes using a multi t distribution
            for the mus (could also use on for other distance!
            
            shrinkage - how much two shrinkge the prior with
        """
        
        rank = self.comm .Get_rank()
        if rank == 0: 
            GMMmus    = []
            GMMsigmas = []
            GMMps     = [] 
            #only rank one
            for i in range(len(self.param['mus'])):
                tmObj = tMd()
                tmObj.nu = 1
                if self.dist_type == 0:
                    tmObj.set_data(self.param['mus'][i]) #here could use anything
                    L   = np.eye(self.param['mus'][i].shape[1])
                    mu = np.mean(self.param['mus'][i], 1)
                    index_L = np.tril_indices(self.param['mus'][i].shape[1], k = -1)
                    x0 =  np.hstack((mu, np.log(np.diag(L)),L[index_L]))
                
                xest = spo.fmin(lambda x: tmObj.f_lik(x), 
                                x0, 
                                maxiter=15000, 
                                maxfun=35000)
                tmObj.set_theta(xest)
                iV = tmObj.weights()
                mu_est_w =  np.sum(iV.reshape((iV.shape[0],1)) * self.param['mus'][i]/np.sum(iV),0)
                sigma_w  =  np.sum(iV.reshape((iV.shape[0],1,1)) * self.param['sigmas'][i]/np.sum(iV),0)
                p_w      =  np.sum(iV.reshape((iV.shape[0],1)) * self.param['ps'][i]/np.sum(iV),0)
            
                GMMmus.append(mu_est_w)
                GMMsigmas.append(sigma_w)
                GMMps.append(p_w)
            GMMparam = {'mus'   : GMMmus, 
                        'sigmas': GMMsigmas,
                        'ps'    : GMMps}
        else:
            GMMparam = None
            # push to other ranks, GMMmus, GMMsigmas, GMMps
        GMMparam = self.comm.bcast(GMMparam,    root=0)
        
        
        for i, GMM in enumerate(self.hGMM.GMMs):
            for k in range(self.K):
                GMM.mu[k]    = GMMparam['mus'][k]
                GMM.sigma[k] = GMMparam['sigmas'][k]
                GMM.p[k]     = GMMparam['ps'][k] 
        for GMM in self.hGMM.GMMs:
            GMM.logisticNormal.set_alpha_p(GMM.p)
    
    
        self.hGMM.update_prior()
        
        [GMM.updata_mudata() for GMM in self.hGMM.GMMs]
        [GMM.sample_x()      for GMM in self.hGMM.GMMs] 
        alphas = self.hGMM.get_alphas()  
        if rank == 0: 
            A = alphas.reshape(-1, 1)
            B = np.vstack(self.hGMM.alpha_prior.Bs_mu)
            self.hGMM.alpha_prior.beta_mu = np.linalg.lstsq(B, A)[0].reshape(-1)
            self.hGMM.alpha_prior.Sigma = shrinkage * np.eye(self.K-1)
        self.hGMM.update_GMM() 
        [GMM.sample()  for GMM in self.hGMM.GMMs] 
        self.hGMM.update_prior()
    
    def dist(self,i, param, mus, sigmas, ps):
        """
            computes the cloestdistance between class i in the GMM
            and the elements in mus, sigmas, ps
            
            modfies mus, sigmas, ps so they can not be double compared
        """
        
        if self.dist_type == 0: # distance computed by distance between mus
            
            
            dist = np.linalg.norm(mus - param['mu'][i],axis=1)
            
            i = np.argsort(dist)[0]
            mus[i,:] = np.inf
            
            return i, mus, sigmas, ps
         
