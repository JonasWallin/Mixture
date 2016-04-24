'''
    standard implimentation of multivariate NIG density
'''

from __future__ import division
import scipy.special as sps
import numpy as np


class NIG(object):
    """
        Univariate density generated from
        Y = \delta - mu + \mu V  + \sqrt{V} \Sigma^{1/2}Z 
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
    
    def __init__(self):
    
        self.delta = None
        self.mu    = None
        self.nu    = None
        self.sigma = None
    
    def set_prior(self, prior):
        """

            priror - dictonary contaning:

            delta - (1x1) mean
            mu    - (1x1) assymetric parameter
            Sigma - (1x1) covariance 
            nu    - (1)   shape parameter
        """

        self.delta = prior['delta']
        self.mu    = prior['mu']
        self.nu    = prior['nu']
        self.sigma = prior['sigma']
        self.update_param()
    
    def set_prior_vec(self, priorvec):
        """

            priror set through vec

            [0] - delta
            [1] - mu  
            [2] - nu  
            [3] - sigma
        """

        self.delta = priorvec[0]
        self.mu    = priorvec[1]
        self.nu    = priorvec[2]
        self.sigma = priorvec[3]
        self.update_param()
    
    def update_param(self):
        """
         updating constants
        """
        
        self.a        = self.nu + self.mu**2 / self.sigma**2  
        self.delta_mu = self.delta - self.mu
        
        self.c0       = - np.log(np.pi) + 0.5 * np.log(self.nu) + self.nu 
        
    def dens(self, y, log_ = True):
        """
            computes the density
            
            y    - (n x 1) densites to computed
            log_ - (bool)  return logarithm of density
        """
        n  = y.shape[0]
        y_ = (y - self.delta_mu ) /self.sigma
        b = self.nu +  y_**2 
        
        const = self.c0 * n
        logf = y_ * self.mu
        logf += const
        logf += self.a / b
        logf += sps.kn(1, np.sqrt(self.a * b))
        
        if not log_:
            return np.exp(logf)
        
        return logf
        
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

    