'''
Created on May 15, 2016

@author: jonaswallin
'''

from .purepython.NIG import NIG as NIGpy
from .purepython.NIG import multi_univ_NIG as multi_univ_NIGpy
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
 

class multi_univ_NIG(multi_univ_NIGpy):
    """
        multivariate vertsion
    
    """
    
    def set_objects(self, d = None):
        """
            sets up the basic objects
        """
        if d is None:
            d = self.d
            
        if d is None:
            raise Exception('dimesnion must be set before set_objects')
        
        self.NIGs = [ NIG() for i in range(d)]  # @UnusedVariable   
    