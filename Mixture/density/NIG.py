'''
Created on May 15, 2016

@author: jonaswallin
'''

from .purepython.NIG import NIG as NIGpy
from .purepython.NIG import multi_univ_NIG as multi_univ_NIGpy
from Mixture.util import Bessel1approx, Bessel0approx, Bessel0eapprox, Bessel1eapprox
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
    
    def dens(self, y =None , log_ = True, paramvec = None, precompute = False):
        """
            computes the density
            
            y          - (n x 1) densites to computed
            log_       - (bool)  return logarithm of density
            paramvec   - (k x 1) the parameter to evalute the density
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
        logf +=  (0.25 * np.log(a) - 0.75 * np.log(b))
        #logf -= np.log(b)
        sqrt_ab = np.array(np.sqrt(a * b)).flatten()
        
        K1e = Bessel1eapprox( sqrt_ab) 
        if precompute:
            self.K1 = K1e
        logf += np.log( K1e) - sqrt_ab
        
        if not log_:
            return np.exp(logf)
        
        return logf

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
            K1 = self.K1
        else:
            K1 = Bessel1eapprox(sqrt_ab) # really -1 but K_-1(x) = K_1(x) 
            
        K0 = Bessel0eapprox(sqrt_ab) 
        sqrt_a_div_b = np.sqrt(a/b)
        EV = np.zeros((np.int(np.max([1,np.prod(np.shape(y))])), 1))
        EV[:,0]         = K0 / K1
        EV[K1==0, 0]     = 1.
        EiV = np.zeros_like(EV)
        EiV[:,0]          = ( K0 + 2 * K1/sqrt_ab) /K1
        EiV[K1==0, 0]     = 1 + 2/sqrt_ab[K1==0]
        EV[:, 0]          /= sqrt_a_div_b
        EiV[:, 0]         *= sqrt_a_div_b
        
        return EV, EiV
    
       
    def __call__(self, paramvec = None, y = None):
        
        return self.dens(paramvec = paramvec, y = y)
 


class NIG_conj(NIGpy):
    """
        NIG with conjugate prior 
        
        assuming 
        \delta -  N (\theta_01, theta_11^-1)
        \mu    -  N (\theta_02, theta_12^-1)
        \sigma -  IG(alpha, beta)
        \nu    -  IG(alpha, beta)
    """
    def __init__(self, param = None, paramvec = None, prior = None):
        """
        
            prior - (4 x 2) 
                    prior[0, :] - mean delta, var delta
                    prior[1, :] - mean mu   , var mu
                    prior[2, :] -  beta/(alpha+1), alpha for Inverse Gamma distribution for \sigma^2
                    prior[3, :] - alpha/beta, alpha for  Gamma distribution for \nu
        """
        super(NIG_conj, self).__init__(param = param, paramvec = paramvec)
        
        if prior is None:
            self.prior = None
        else:
            self.prior = np.array(prior)
      
    def set_prior(self, prior = None):
        
        
        if prior is None:
            self.prior = None
        else:
            self.prior = np.array(prior)
        
    def _prior(self, prior_ = None):
        
        if prior_ is None:
            return self.prior
        
        return prior_  
        
    def Mstep(self, EV, EiV, p = None, y = None, paramvec = None, update = [1,1,1,1], prior = None):
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
        
        prior = self._prior(prior_ = prior)
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
        
        if prior is not None:
            
            Q[0, 0] += prior[0,1]
            Q[1, 1] += prior[1,1]
            b[0]    += prior[0,0] *prior[0,1]
            b[1]    += prior[1,0] *prior[1,1]    
                
        Qinvb = np.linalg.solve(Q, b) 
        
        if update[0] > 0:
            delta = Qinvb.flatten()[0]
        if update[1] > 0:
            mu   = Qinvb.flatten()[1]
        
        if prior is not None:
            
            Q[0, 0] -= prior[0,1]
            Q[1, 1] -= prior[1,1]
            b[0]    -= prior[0,0] *prior[0,1]
            b[1]    -= prior[1,0] *prior[1,1]      
            
        
        # nu step
        if update[2]  > 0:
            c  = 0.5 * (np.sum(p_EV) + sum_pEiV - 2*sum_p  )
            c0 = 0.5 * sum_p 
            if prior is not None:
                c  +=  prior[2,1]/prior[2,0]
                c0 +=  (prior[2,1]-1)
            nu = c0 / c
                 
        # sigma step
        # 
        if update[3] > 0:
            delta_mu = np.array([[delta],[mu]])
            H = np.dot( -0.5*np.dot(delta_mu.transpose(), Q) + b.transpose(), delta_mu) - 0.5*np.sum(np.multiply(y.flatten()**2 , p_EiV.flatten()))  
            H[0,0] *= -1.
            n_ = sum_p*0.5
            if prior is not None:
                H[0,0] += (prior[3,1] + 1) * prior[3,0]
                n_ += prior[3,1] + 1
            
            sigma = np.sqrt(H[0,0]/n_)

               
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
               precompute   = False,
               prior = None):
        
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
        
        
        res = self.Mstep(EV, EiV, p = p, y = y, paramvec = paramvec, update = update, prior = prior)
        if update_param:
            self.set_param_vec(res)
        
        return res

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
    
    
class multi_univ_NIG_conj(multi_univ_NIGpy):
    """    
        multivariate version of conjugate NIG
    
    """ 


    def __init__(self, d , param = None, paramvec = None, prior = None):
        
        
        super(multi_univ_NIG_conj, self).__init__(d = d, param = param, paramvec = paramvec)
        
        if prior is None:
            [ self.NIGs[i].set_prior(None) for i in range(self.d)] 
        else:
            [ self.NIGs[i].set_prior(prior[i]) for i in range(self.d)] 
          
    
    
    def set_prior(self, prior):
        """
            prior - (d x 1) list of priors
        
        """
        [ self.NIGs[i].set_prior(prior[i]) for i in range(self.d)]
        

    def set_objects(self, d = None):
        """
            sets up the basic objects
        """
        if d is None:
            d = self.d
            
        if d is None:
            raise Exception('dimesnion must be set before set_objects')
        
        self.NIGs = [ NIG_conj() for i in range(d)]  # @UnusedVariable   

    def _prior(self, prior_ = None):
        
        if prior_ is None:
            return [ self.NIGs[i].prior for i in range(self.d)]  # @UnusedVariable
        
        return prior_       

    def EMstep(self, 
               y = None, 
               paramMat = None, 
               paramvec = None,
               update  = [1,1,1,1],
               p   = None,
               update_param = True,
               precompute   = False,
               prior        = None):
        
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
            prior        - 
        """
        if paramvec is not None:
            paramMat = self.paramMatToparamVec(paramvec)
            
            
        if y is None:
            y = self.y
            
        
        prior = self._prior(prior)  
        
        out_ = np.zeros((self.d, 4))
            
        for i in range(self.d):
            if paramMat is not None:
                paramvec = paramMat[i, ]
            res = self.NIGs[i].EMstep(y = y[:,i],
                                p = p,
                                paramvec = paramvec,
                                update = update,
                                update_param = update_param,
                                precompute = precompute,
                                prior        = prior[i]) 
        
            out_[i,:] = res
        
        
        return out_