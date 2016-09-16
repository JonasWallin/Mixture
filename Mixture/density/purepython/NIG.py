'''
    standard implimentation of multivariate NIG density
'''

from __future__ import division
import scipy.special as sps
import numpy as np
import numpy.random as npr
from scipy.stats import invgauss




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
    
    def get_param(self):
        """
            get the parameters
        """
        out = {"nu":np.log(self.nu), 
               "mu": self.mu,
               "delta": self.delta,
               "sigma": np.log(self.sigma)}
        return(out)
    
    def get_param_vec(self):
        """
            get the parameters in vector format
        """
        
        out = np.array([self.delta, self.mu, np.log(self.nu), np.log(self.sigma)])
        return(out)
       
    def Mstep(self, EV, EiV, p = None, y = None, paramvec = None, update = [1,1,1,1]):
        """
            Takes an Mstep in a EM algorithm
            
            log(l(\theta|Y) ) = \sum_i p_i (\log(\sigma ^{-1}) - (Y_i - delta + mu - mu V_i)^2/(2 \sigma^2 V_i) 
                                 - \log(\nu)/2 - V_i^{-1}/2 \nu - V_i/2 \nu + \nu )
                                 + C   
            
            EV       - (n x 1) the expectation of the latent variance parameter
            EiV      - (n x 1) the expectation of the latent 1/variance parameter
            p        - (n x 1) weight of the observations p_i \in [0,1]
            y        - (n x 1) the observations
            paramvec - (k x 1) the parameter to evalute the density
            update   - (k x 1) which of the parameters should be updated, order
                               delta, mu, nu ,sigma 
        """
    
    
        if y is None:
            y = self.y
    
        if p is None:
            p = np.ones((max([1, np.prod(np.shape(y))]), 1))
                                 
        delta, mu, nu, sigma = self._paramvec(paramvec)
        
        # 
        sum_p  = np.sum(p)
        p_EV   = p.flatten() * EV.flatten()
        p_EiV  = p.flatten() * EiV.flatten()
        sum_pEiV  = np.sum(p_EiV)
        sum_pEV   = np.sum(p_EV)
        sum_pY    = np.sum(np.multiply(y.flatten(), p.flatten()))
        sum_pYEiv = np.sum(np.multiply(y.flatten() , p_EiV.flatten()))
        #delta, mu step
        # the log likelhiood quadratic function
        #- \frac{1}{2} [\delta, mu]^T Q  [\delta, mu] +  b^T [\delta, mu]
        Q = np.array([[sum_pEiV,            (-sum_pEiV + sum_p)], 
                      [ (-sum_pEiV + sum_p),  sum_pEiV  + sum_pEV - 2 * sum_p]])
        b = np.array([ [sum_pYEiv], [-sum_pYEiv + sum_pY]])
        Qinvb = np.linalg.solve(Q, b) 
        
        if update[0] > 0:
            delta = Qinvb.flatten()[0]
        if update[1] > 0:
            mu   = Qinvb.flatten()[1]
        
        # sigma step
        
        # 
        if update[3] > 0:
            delta_mu = np.array([[delta],[mu]])
            H = np.dot( -0.5*np.dot(delta_mu.transpose(), Q) + b.transpose(), delta_mu) - 0.5*np.sum(np.multiply(y.flatten()**2 , p_EiV.flatten()))  
            sigma = np.sqrt(-2*H[0,0]/sum_p)
        
        
        # nu step
        if update[2]  > 0:
            nu = sum_p / (np.sum(p_EV) + sum_pEiV - 2*sum_p )
               
        return [delta, mu, np.log(nu), np.log(sigma)]
    
    def EMstep(self, 
               y = None, 
               paramvec = None,  
               update  = [1,1,1,1],
               p   = None,
               compute_E = True, 
               EV = None, 
               EiV = None,
               update_param = True):
        
        """
            Takes an Mstep in a EM algorithm
            y            - (n x 1) the observations
            p            - (n x 1) weight of the observations p_i \in [0,1] 
            paramvec     - (k x 1) the parameter to evalute the density
            update       - (k x 1) which of the parameters should be updated, order
                               delta, mu, nu ,sigma 
            compute_E    - (bool) Compute the expectations
            EV           - (n x 1) the expectation of the latent variance parameter
            EiV          - (n x 1) the expectation of the latent 1/variance parameter
            update_param - (bool) update the parameters in the object
        """
        
        if compute_E:
            EV, EiV = self.EV(y, paramvec)
        
        
        res = self.Mstep(EV, EiV, p = p, y = y, paramvec = paramvec, update = update)
        if update_param:
            self.set_param_vec(res)
        
        return res
        
        
    def EV(self, y = None, paramvec = None):
        """
            compute the EV|y and EV^{-1}|y where V_i is the variance component of y_i
            Used for EM algorithm
        
            y        - (n x 1) the observations
            paramvec - (k x 1) the parameter to evalute the density
        """
        
        
        
        if y is None:
            y = self.y
        delta, mu, nu, sigma = self._paramvec(paramvec)
        a = nu + mu**2 /sigma**2
        delta_mu = delta - mu
        y_ = (y - delta_mu ) /sigma
        b = nu + y_**2 
        
        sqrt_ab = np.sqrt(a * b)
        K1 = sps.kn(1, sqrt_ab) # really -1 but K_-1(x) = K_1(x) 
        K0 = sps.kn(0, sqrt_ab)
        sqrt_a_div_b = np.sqrt(a/b)
        EV = np.zeros((np.int(np.max([1,np.prod(np.shape(y))])), 1))
        EV[:,0]         = K0 / K1
        EiV = np.zeros_like(EV)
        EiV[:,0]             = ( K0 + 2 * K1/sqrt_ab) /K1
        EV[:, 0]          /= sqrt_a_div_b
        EiV[:, 0]         *= sqrt_a_div_b
        
        return EV, EiV
    
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
        self.V_sim = V
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

    