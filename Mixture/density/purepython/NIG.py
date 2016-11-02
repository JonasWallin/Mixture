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
               update_param = True,
               precompute   = False):
        
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
            precompute - (bool)  is the bessel1 alredy caculated
        """
        
        if compute_E:
            EV, EiV = self.EV(y, paramvec, precompute = precompute)
        
        
        res = self.Mstep(EV, EiV, p = p, y = y, paramvec = paramvec, update = update)
        if update_param:
            self.set_param_vec(res)
        
        return res
        
        
    def EV(self, y = None, paramvec = None, precompute = False):
        """
            compute the EV|y and EV^{-1}|y where V_i is the variance component of y_i
            Used for EM algorithm
        
            y          - (n x 1) the observations
            paramvec   - (k x 1) the parameter to evalute the density
            precompute - (bool)  is the bessel1 alredy caculated
        """
        
        
        
        if y is None:
            y = self.y
        delta, mu, nu, sigma = self._paramvec(paramvec)
        a = nu + mu**2 /sigma**2
        delta_mu = delta - mu
        y_ = (y - delta_mu ) /sigma
        b = nu + y_**2 
        
        sqrt_ab = np.sqrt(a * b)
        if precompute:
            K1e = self.K1e
        else:
            K1e = sps.k1e(sqrt_ab) # really -1 but K_-1(x) = K_1(x) 
            
        K0e = sps.k0e( sqrt_ab)
        
        sqrt_a_div_b = np.sqrt(a/b)
        EV = np.zeros((np.int(np.max([1,np.prod(np.shape(y))])), 1))
        EV[:,0]         = K0e / K1e
        EV[K1e==0, 0]     = 1.
        EiV = np.zeros_like(EV)
        EiV[:,0]          = ( K0e + 2 * K1e/sqrt_ab) /K1e
        EiV[K1e==0, 0]     = 1 + 2/sqrt_ab[K1e==0]
        EV[:, 0]          /= sqrt_a_div_b
        EiV[:, 0]         *= sqrt_a_div_b
        
        return EV, EiV
    
    def dens(self, y =None , log_ = True, paramvec = None, precompute = False):
        """
            computes the density
            
            y        - (n x 1) densites to computed
            log_     - (bool)  return logarithm of density
            paramvec - (k x 1) the parameter to evalute the density
            precomptue - (bool) store the bessel cacululation
            
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
        sqrt_ab = np.sqrt(a * b)
        K1e = sps.k1e( sqrt_ab)
        if precompute:
            self.K1e = K1e
        logf += np.log(K1e) - sqrt_ab
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
    
    def __str__(self):
        
        str_ = " delta = {0:+2.2}, mu = {1:+2.2}, nu = {2:+2.2}, sigma = {3:+2.2}".format(self.delta, self.mu, self.nu, self.sigma)
        return str_
    
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
        



# class multivariateNIG(object):
#     """
#         multivariate nig can be generated from
# 
#         Y = \delta - mu + \mu V  + \sqrt{V} \Sigma^{1/2} Z 
#         Z = N(0,I)
#         V = IG(\nu, \nu)
#         density:
# 
#             f(v) \propto \nu^{1/2} v^{-3/2} e^{- \nu/(2V) - \nu V/2   + \nu} 
#             f(y) = 2\sqrt{\nu} |\Sigma|^{-1/2} (2\pi)^{d+1/2} \exp(\nu) ...
#                    \exp( (y - \delta + \mu)^T \Sigma^{-1} \mu )    ...
#                    (a/b)^{(d+1)/2} K_{(d+1)/2} ( \sqrt{ab})
#             a = \mu^T \Sigma^{-1} \mu + \nu
#             b = (x -  \delta + \mu)^T \Sigma^{-1} (x -  \delta + \mu) + \nu
#     """
# 
# 
#     def __init__(self, d = None):
# 
# 
#         self.d = None
#         pass
# 
# 
#     def set_prior(self, prior):
#         """
# 
#             priror - dictonary contaning:
# 
#             delta - (dx1) mean
#             mu    - (dx1) assymetric parameter
#             Sigma - (dxd) covariance 
#             nu    - (1)   shape parameter
#         """
# 
#         pass    


class multi_univ_NIG(object):   
    """
        for d-dimensional object where each dimension is iid
    """
    
    def __init__(self, d, param = None, paramvec = None):
    
    
    
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
        
    def __str__(self):
        
        str_ = ""
        for i in range(self.d):
            str_ +="{}:".format(i) + self.NIGs[i].__str__() + "\n"
        return str_   
    
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
        
        paramMat = self.paramvecToparamMat(paramvec)
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
 
    def get_paramvec(self):
        
        paramvec = np.zeros(self.d*4)
        for i in range(self.d):
            paramvec[4*i:4*(i+1)] = self.NIGs[i].get_param_vec()
            
        return paramvec
    
    
    def get_paramMat(self):
        """
          returns:
           ParamMat (d x 4)

            [i,0] - delta
            [i,1] - mu  
            [i,2] - nu  (in log)
            [i,3] - sigma (in log)
        """
        paramvec = np.zeros((self.d, 4))
        for i in range(self.d):
            paramvec[i, ] = self.NIGs[i].get_param_vec()
            
        return paramvec 
 
    
    def EMstep(self, 
               y = None, 
               paramMat = None, 
               paramvec = None,
               update  = [1,1,1,1],
               p   = None,
               update_param = True,
               precompute   = False):
        
        """
            Takes an Mstep in a EM algorithm
            y            - (n x 1) the observations
            p            - (n x d) weight of the observations p_i \in [0,1] 
            paramvec     - (d*k x 1) the parameter to evalute the density
            paramvec     - (d   x k) the parameter to evalute the density
            update       - (k x 1) which of the parameters should be updated, order
                               delta, mu, nu ,sigma 
            compute_E    - (bool) Compute the expectations
            update_param - (bool) update the parameters in the object
            precompute   - (bool) is the Bessel function alredy calculated?
        """
        if paramvec is not None:
            paramMat = self.paramMatToparamVec(paramvec)
            
            
        if y is None:
            y = self.y
            
            
        
        out_ = np.zeros((self.d, 4))
            
        for i in range(self.d):
            if paramMat is not None:
                paramvec = paramMat[i, ]
            res = self.NIGs[i].EMstep(y = y[:,i],
                                p = p,
                                paramvec = paramvec,
                                update = update,
                                update_param = update_param,
                                precompute = precompute) 
        
            out_[i,:] = res
        
        
        return out_
       
    
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
    
    def dens_dim(self, y =None, log_ = True, paramMat = None, paramvec = None, precompute = False):
        
        
        if paramvec is not None:
            paramMat = self.paramMatToparamVec(paramvec)
            
        if y is None:
            y = self.y
        
        if paramMat is None:
            res = np.array([nig.dens(y = y[:,i], log_ = True, precompute = precompute) for i, nig in enumerate(self.NIGs)])
        else:
            res = np.array([nig.dens(y = y[:,i], paramvec = paramMat[i, ] ,log_ = True,  precompute = precompute) for i, nig in enumerate(self.NIGs)])
        
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
    
    def paramvecToparamMat(self, paramvec):
        
        if paramvec is None:
            return None
        paramMat = np.array(paramvec)
        return  paramMat.reshape((self.d, 4))
    
    def paramMatToparamVec(self, paramMat):
        
        if paramMat is None:
            return None
        paramvec = np.array(paramMat)
        return  paramvec.flatten()
    
    def simulate(self, n = 1, paramMat = None, paramvec = None):
        """
            simulating n random variables from prior
        """
        
        if paramvec is not None:
            paramMat = self.paramvecToparamMat(paramvec)
            
        if paramMat is None:
            X = np.array([nig.simulate(n = n ) for i, nig in enumerate(self.NIGs)]).transpose()
        else:
            X = np.array([nig.simulate(n = n, paramvec = paramMat[i, ] ) for i, nig in enumerate(self.NIGs)]).transpose()
    
        return X

