'''
    standard implimentation of multivariate NIG density
'''

from __future__ import division
import scipy.special as sps
import numpy as np
import numpy.random as npr
from scipy.stats import invgauss





#TODO: change parametrization so that the paramtrization has mean, variance , shape, kurtoties
class NIG(object):
    """
        Univariate density generated from
        Y = \delta - mu + \mu V  + \sqrt{V} \sigma^{1/2}Z 
        Z = N(0,1)
        V = IG( \nu , \nu)
        densities:
            f(v) \propto \nu^{1/2} v^{-3/2} e^{- \frac{\nu}{2} V^{-1} - \frac{\nu}{2} V   + \nu} 
            f(y) = \sqrt{\nu} \sigma^{-1/2} \pi^{-1} \exp(\nu) ...
                   \exp( \frac{1}{\sigma^2}(y - \delta + \mu) \mu )    ...
                   \frac{a }{ b} K_{ 1 }(\sqrt{ ab })
            a = \frac{ 1 }{ \sigma^2} \mu^2 +  \nu
            b =  \frac{1}{ \sigma^2 }  (x -  \delta + \mu)^2  + \nu
    """
    
    def __init__(self, param = None, paramvec = None):
    
        if param is not None:
            self.set_param(param)
        elif paramvec is not None:
            self.set_param_vec(paramvec)
        else:
            self.delta = None
            self.mu    = None
            self.nu    = None
            self.sigma = None
        self.k     = 4
    
    def set_param(self, param):
        """

            priror - dictonary contaning:

            delta  - (1x1) mean
            mu     - (1x1) assymetric parameter
            sigma  - (1x1) the std (in log format)
            nu     - (1)   shape parameter (in log format)
        """

        self.delta = param['delta']
        self.mu    = param['mu']
        self.nu    = param['nu']
        self.sigma = param['sigma']
        self.update_param()
    
    def set_data(self, y):
        """
            setting data
            y - ( n x 1) numpy array
        """
        self.y = y.flatten()
        
        
    def set_param_vec(self, paramvec):
        """

            priror set through vec

            [0] - delta
            [1] - mu  
            [2] - nu  (in log)
            [3] - sigma (in log)
        """

        self.delta = paramvec[0]
        self.mu    = paramvec[1]
        self.nu    = paramvec[2]
        self.sigma = paramvec[3]
        self.update_param()
    
    def update_param(self):
        """
         updating constants
        """
        
        self.sigma  = np.exp(self.sigma)
        self.nu     = np.exp(self.nu)
        
    def dens(self, y =None , log_ = True, paramvec = None):
        """
            computes the density
            
            y        - (n x 1) densites to computed
            log_     - (bool)  return logarithm of density
            paramvec - (k x 1) the parameter to evalute the density
            
             f(y) = \sqrt{\nu} \sigma^{-1} \pi^{-1} \exp(\nu) ...
                   \exp( \frac{1}{\sigma^2}(y - \delta + \mu) \mu )    ...
                   \frac{a }{ b} K_{ 1 }(\sqrt{ ab })
        """
        
        if y is None:
            y = self.y
            
        delta, mu, nu, sigma = self._paramvec(paramvec)

        a        = nu + mu**2 / sigma**2  
        delta_mu = delta - mu
        
        c0       = - np.log(np.pi) + 0.5 * np.log(nu) + nu - np.log(sigma)
        
        
        #n  = y.shape[0]
        y_ = (y - delta_mu ) /sigma
        b = nu +  y_**2 
        
        const = c0 #* n
        logf = y_ * ( mu / sigma)
        logf += const
        logf += 0.5 * (np.log(a) - np.log(b))
        logf += np.log(sps.kn(1, np.sqrt(a * b)))
        
        if not log_:
            return np.exp(logf)
        
        return logf
    
    
    def _paramvec(self, paramvec):
        """
            hidden function that returns paramvec
        """
        if paramvec is None:
            mu    = self.mu
            delta = self.delta
            nu    = self.nu
            sigma = self.sigma
        else:
            delta = paramvec[0]
            mu    = paramvec[1]
            nu    = np.exp(paramvec[2])
            sigma = np.exp(paramvec[3])
            
        return delta, mu, nu, sigma
    
    def __call__(self, paramvec = None, y = None):
        
        return self.dens(paramvec = paramvec, y = y)
    
    
    def simulate(self, n = 1, paramvec = None):
        """
            simulating n random variables from prior
        """
        #invGaussian scale like
        # tX \sim IG(t \mu, t, \lambda)
        # scipy uses
        # X \sim IG( \mu, 1)         
        delta, mu, nu, sigma = self._paramvec(paramvec)
        V = nu * invgauss.rvs( 1. / nu , size= (n,1) )
        Z = npr.randn(n,1)
        X = (delta - mu) + mu * V + sigma * np.sqrt(V) * Z
        X = X.flatten()
        return X
        
        


class multi_univ_NIG(object):   
    """
        for d-dimensional object where each dimension is iid
    """
    
    def __init__(self, d = None, param = None, paramvec = None):
    
    
    
        self.NIGs = None
        self.k    = None
        if param is not None:
            
            self.d = len(param)
            d = self.d
            self.set_objects()
            self.set_param(param)
            
        elif paramvec is not None:
            
            self.d = np.int(paramvec.flatten().shape[0]/4.)
            d = self.d
            self.set_objects()
            self.set_param_vec(paramvec)
            
        
        if d is not None:
            self.d = d
            
            self.k     = 4  * d 
            if self.NIGs is None:
                self.set_objects()
    
    def set_objects(self, d = None):
        """
            sets up the basic objects
        """
        if d is None:
            d = self.d
            
        if d is None:
            raise Exception('dimesnion must be set before set_objects')
        
        self.NIGs = [ NIG() for i in range(d)]  # @UnusedVariable
      
    def set_param(self, paramList):
        """

            paramList - list of dictonary contaning:

            delta  - (1x1) mean
            mu     - (1x1) assymetric parameter
            sigma  - (1x1) the std (in log format)
            nu     - (1)   shape parameter (in log format)
        """
        
        [nig.set_param(paramList[i]) for i, nig in enumerate(self.NIGs)]
        
    
    def set_data(self, y):
        """
            setting data
            y - ( n x d) numpy array
        """
        if self.d is not None:
            if y.shape[1] != self.d:
                raise Exception('y must have {} columns'.format(self.d))
            
        self.y = np.copy(y)
        
    
    def set_param_vec(self, paramvec):
        
        paramMat = self.paramMatToparamVec(paramvec)
        self.set_param_Mat(paramMat)
        
    def set_param_Mat(self, paramMat):
        """

           ParamMat (d x 4)

            [i,0] - delta
            [i,1] - mu  
            [i,2] - nu  (in log)
            [i,3] - sigma (in log)
        """
        
        [nig.set_param_vec(paramMat[i,]) for i, nig in enumerate(self.NIGs)]
    
    
    def dens_d(self, dim, y = None, log_ = True, paramMat = None, paramvec = None):
        """
            evaluetes density at dim 
        
        """
        if paramvec is not None:
            paramMat = self.paramMatToparamVec(paramvec)
            
            
        if y is None:
            y = self.y
            
        if len(y.shape) > 1 and (y.shape[1] > 1):
            y_ = y[:,dim]
        else:
            y_ = y
        if paramMat is None:
            res = self.NIGs[dim].dens(y = y_, log_ = log_) 
        else:
            res = self.NIGs[dim].dens(y = y_, log_ = log_, paramvec = paramMat[dim, ]) 
            
        return res
    
    def dens_dim(self, y =None, log_ = True, paramMat = None, paramvec = None):
        
        
        if paramvec is not None:
            paramMat = self.paramMatToparamVec(paramvec)
            
        if y is None:
            y = self.y
        
        if paramMat is None:
            res = np.array([nig.dens(y = y[:,i], log_ = True) for i, nig in enumerate(self.NIGs)])
        else:
            res = np.array([nig.dens(y = y[:,i], paramvec = paramMat[i, ] ,log_ = True) for i, nig in enumerate(self.NIGs)])
        
        if log_ is True:
            return res.transpose()
        else:
            return np.exp(res.transpose())
    
    def density(self, y =None , log_ = True, paramMat = None):
        """
            computes the joint density
        """
        res = self.dens_dim(y = y, log_ = True, paramMat = paramMat)
        res = np.sum(res, 1)
        
        if y is None:
            y = self.y

        if log_ is True:
            return res
        else:
            return np.exp(res)
        
    def __call__(self, paramvec = None, y = None):
        """
            used for optimization
            paramvec - (d * 4 x 1)
        """
        if self.d is None:
            self.d = y.shape[1]
        paramMat = self.paramMatToparamVec(paramvec)
        return self.density(paramMat = paramMat, y = y)
    
    def paramMatToparamVec(self, paramvec):
        
        if paramvec is None:
            return None
        return  paramvec.reshape((self.d, 4))
    
    def simulate(self, n = 1, paramMat = None, paramvec = None):
        """
            simulating n random variables from prior
        """
        if paramvec is not None:
            paramMat = self.paramMatToparamVec(paramvec)
            
        if paramMat is None:
            X = np.array([nig.simulate(n = n ) for i, nig in enumerate(self.NIGs)]).transpose()
        else:
            X = np.array([nig.simulate(n = n, paramvec = paramMat[i, ] ) for i, nig in enumerate(self.NIGs)]).transpose()
    
        return X
    
class multivariateNIG(object):
    """
        multivariate nig can be generated from

        Y = \delta - mu + \mu V  + \sqrt{V} \Sigma^{1/2} Z 
        Z = N(0,I)
        V = IG(\nu, \nu)
        density:

            f(v) \propto \nu^{1/2} v^{-3/2} e^{- \nu/(2V) - \nu V/2   + \nu} 
            f(y) = 2\sqrt{\nu} |\Sigma|^{-1/2} (2\pi)^{d+1/2} \exp(\nu) ...
                   \exp( (y - \delta + \mu)^T \Sigma^{-1} \mu )    ...
                   (a/b)^{(d+1)/2} K_{(d+1)/2} ( \sqrt{ab})
            a = \mu^T \Sigma^{-1} \mu + \nu
            b = (x -  \delta + \mu)^T \Sigma^{-1} (x -  \delta + \mu) + \nu
    """


    def __init__(self, d = None):


        self.d = None
        pass


    def set_prior(self, prior):
        """

            priror - dictonary contaning:

            delta - (dx1) mean
            mu    - (dx1) assymetric parameter
            Sigma - (dxd) covariance 
            nu    - (1)   shape parameter
        """

        pass

    