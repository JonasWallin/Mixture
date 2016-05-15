'''
Created on May 15, 2016

@author: jonaswallin
'''

from .purepython.NIG import NIG as NIGpy
from Mixture.util import Bessel1approx
import numpy as np
#import line_profiler

#    58     41960       551857     13.2     17.5  : logf += 0.5 * (np.log(a) - np.log(b))
#    59     41960      1458115     34.8     46.3  :logf += np.log(Bessel1approx( np.array(np.sqrt(a * b)).flatten()))
#    profiling result  (pure python)
#   126     62936       807035     12.8      2.4  : logf += 0.5 * (np.log(a) - np.log(b))
#   127     62936     31469201    500.0     92.8  : logf += np.log(sps.kn(1, np.sqrt(a * b))


#compared to pure python
class NIG(NIGpy):
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
        logf += np.log(Bessel1approx( np.array(np.sqrt(a * b)).flatten()))
        
        if not log_:
            return np.exp(logf)
        
        return logf
    
    def __call__(self, paramvec = None, y = None):
        
        return self.dens(paramvec = paramvec, y = y)
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
    