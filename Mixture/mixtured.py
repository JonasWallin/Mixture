'''
    A generic class for densites of class of 1d independent densites to make d dimensional model
Created on Apr 27, 2016

@author: jonaswallin
'''


from __future__ import division
import numpy as np
import copy as cp
import numpy.random as npr
from scipy.misc import logsumexp


#TODO: 1,setup general parameter setup 2,EMstep
class mixtured(object):
    """
        the object consits of d one dimensional densities (class).
        the denisties (class) needs the following funcitons
        - set_param     -  set the prior for the density
        - __call__      -  returns the log density
        - set_param_vec -  same as set_prior but now the data is in vector form 
        - set_data      - set the data
        - k             - number of parameters
    
    """
    
    def __init__(self, K , d = None):
        """
            K - number of classes
            d - dimension of data
        """
        self.K  = K
        self.d = d
        self.dens = None
        self.alpha = None
        self.EMM_iter = 3

    def __call__(self, paramvec = None, paramMat = None, alpha=  None, p = None):
        """
            paramavec (d * k + (d-1) x 1) the prior vectors in matrix format
                     (0:d-2)            :  alpha which defines the prior prob as:
                                          \pi_i = np.exp( alpha_i) / 1 + sum(np.exp(alpha_i))
        """
        
        
        return np.sum(logsumexp(self.weights(normalized = False,
                                             paramvec = paramvec,
                                             paramMat = paramMat,
                                             alpha = alpha,
                                             p = p), axis=0))
        
    def set_data(self, y):
        """
            the densites needs to be set first
            y - (n x d) the data
        """
        
        
        if self.dens is None:
            raise Exception('need to set denity first')
        
        
        if self.d is not None:
            if np.shape(y)[1] != self.d:
                raise Exception('the dimenstion of the data is {0} and d = {1}'.format(np.shape(y)[1], self.d))
        
        
        if self.d is None:
            self.d = np.shape(y)[1]
        
        self.y = cp.deepcopy(y)
        self.n = np.shape(y)[0]
            
    
    def set_densites(self, dens):
        """
            dens - (K x 1) list of densites - denites needs that __call__(x) returns loglikehood of x
        """
        
        if len(dens) != self.K:
            raise Exception('the dimenstion of the dens is {0} and K = {1}'.format( len(dens), self.K))
            
        self.dens = dens
        
        
    def _check_parameters(self, paramvec, paramMat , alpha , p ):
        """
            interall function for checking parameters are ok, paramvec is dominating paramMat
        """
        if paramvec is not None:
            p, alpha, paramMat = self.paramvec_to_paramMat(paramvec)  # @UnusedVariable
        
        if (paramMat is not None) and (alpha is None and p is None):
            raise Exception("if paramMat is used must spesifiy either alpha or p")
        
        if alpha is not None:
            p = self.alpha_to_p(alpha)
        
        if paramMat is None:
            p, alpha, paramMat = self.get_paramMat()
            
        if p is None:
            raise Exception("the parameter must be set")
        
        if alpha is None:
            alpha = self.p_to_alpha(p)
        
        return p, alpha, paramMat
     
             
    def get_paramMat(self):
        """
            returns:
            
            p        -  (k x 1)   probabililites
            alpha    -  (k-1 x 1) log format p
            paramMat - (k x 1)    list of parameter matrix for the parametes
        """
        if self.alpha is None:
            return None, None, None
        p = self.alpha_to_p(self.alpha)
        alpha = self.alpha
        paramMat  = []
        for den in self.dens:
            paramMat.append(den.get_paramMat())
            
        return p, alpha, paramMat
    
    def p_to_alpha(self, p):
        """
            converting p to alpha
        
            p - (K x 1) probabilites of beloning to a class
        """  
        
        if p is None:
            return None
        
        p = np.array(p).flatten()
        alpha = np.zeros(self.K - 1)
        for i,p_ in enumerate(p[1:]):
            alpha[i] = np.log(p_ / p[0])
            
        return alpha
    
    def alpha_to_p(self, alpha):
        """
            converting p to alpha
        
            alpha - (K - 1 x 1) probabilites of beloning to a class
        """  
        
        if alpha is None:
            return None
        alpha = np.array(alpha).flatten()
        p = np.zeros(self.K)
        sum_expalpha = np.sum(np.exp(alpha)) + 1
        p[0] = 1 / sum_expalpha
        
        for i,alpha_ in enumerate(alpha):
            p[i+1] = np.exp(alpha_) / sum_expalpha
            
        return p
            
        
    
    def set_paramMat(self, paramMat, p = None, alpha = None):
        """
            setting the parameter where:
            
            p        - (K x 1) probability of each class
            alpha    - (K-1 x 1) optional
            paramMat - (k x 1) list each entry is matrix of parameters
        
        """
        
        
        if p is None and alpha is None:
            raise ValueError("either alpha or P must not be None")
        
        if p is not None:
            
            if len(p) != (self.K):
                raise ValueError("p length must be K ={}".format(self.K))
            alpha = self.p_to_alpha(p)
            
            
        if alpha is not None:
            if len(alpha) != (self.K-1):
                raise ValueError("alpha length must be K-1 ={}".format(self.K-1))
            self.alpha = np.zeros_like(alpha)
            self.alpha[:] = alpha[:]
        
        for i, parMat_ in enumerate(paramMat):
            self.dens[i].set_param_Mat(parMat_)
                
    def set_param_vec(self, paramvec):  
        """
        
        """
        
        self.paramvec = np.copy(paramvec)
        s = 0
        self.alpha =  paramvec[s:(self.K-1)]
        s += self.K-1
        for  den in self.dens:
            den.set_param_vec(paramvec = paramvec[s:(s + den.k)]) 
            s += den.k
    
    def paramMat_to_paramvec(self, paramMat, alpha = None, p = None):
        
        if p is not None:
            alpha  = self.p_to_alpha(p)
        paramvec = alpha
        for  den in self.dens:
            paramvec = np.hstack((paramvec, den.get_paramvec()))
            
        return paramvec
    
      
    def paramvec_to_paramMat(self, paramvec):
        
        s = 0
        alpha = np.hstack((0, paramvec[s:(self.d-1)]))
        p = self.alpha_to_p(alpha)
        paramMat = []
        
        for i in range(self.K):
            paramMat.append(self.dens[i].paramvecToparamMat(paramvec[s:(s + self.dens[i].k)]))
            s += self.dens[i].k
            
        return p, alpha, paramMat
    
    def get_paramvec(self):
        """
            collects the parameter vector
        """
        
        paramvec    = np.zeros_like(self.alpha)
        paramvec[:] = self.alpha[:] 
        
        for den in self.dens:
            paramvec = np.hstack((paramvec, den.get_paramvec()))
        
        return(paramvec)
         
    
    
    def sample(self, n = 1, paramvec = None, paramMat = None, alpha=  None, p = None):
        


        p, alpha, paramMat = self._check_parameters(paramvec, paramMat , alpha , p )
        
        P = np.hstack((0.,p))
        P = np.cumsum(P)
        
        U = npr.rand(n)
        y = np.zeros((n,self.d))
        for i in range(self.K ): 
            index = (P[i] <= U) * (U <= P[i + 1]) == 1
            y[index,:] = self.dens[i].simulate(np.sum(index), paramMat = paramMat[i])
            
        return y
            
    
    
    def density(self, paramvec = None, paramMat = None, alpha=  None, p = None, log = True):
        """
        
        """
        
        p, alpha, paramMat = self._check_parameters(paramvec, paramMat , alpha , p )
        
        
        
        logf  = np.zeros((self.n, self.d))
        alpha = np.hstack((0, alpha))
        for i, den in enumerate(self.dens):
            logf += p[i] * np.exp(den.dens_dim( y = self.y,  paramMat = paramMat[i]))
        
        logf = np.log(logf)
        if log == True:
            return logf
        else:
            return np.exp(logf)

    def density_1d(self, dim, y = None,  paramvec = None, paramMat = None, alpha=  None, p = None,  log = True):
        """
            evalutes one dimension of the density
            
            dim - int the dimension of data
            y   (n x d) or (n x 1) the data of full dimension or only dimension of intresset
        """
        
        if y is None:
            y = self.y
            
        
        if len(y.shape) > 1 and (y.shape[1] > 1):
            y_ = y[:,dim]
        else:
            y_ = y
        
        p, alpha, paramMat = self._check_parameters(paramvec, paramMat , alpha , p )
        
        n = len(y_)
        logf  = np.zeros((n, 1))
        alpha = np.hstack((0, alpha))
        for i, den in enumerate(self.dens):
            logf[:,0] += p[i] * np.exp(den.dens_d( dim = dim, y = y_, paramMat = paramMat[i]) )
        
        
        if log == True:
            return np.log(logf)
        else:
            return logf

    def weights(self, y = None, paramvec = None, paramMat = None, alpha=  None, p = None,
                log=True, precompute = False, normalized=True):
        
        
        if y is None:
            y = self.y
            
        p, alpha, paramMat = self._check_parameters(paramvec, paramMat , alpha , p )

        n = y.shape[0]
        pik = np.zeros((self.K, n))
        alpha = np.hstack((0, alpha))
        for i, den in enumerate(self.dens):
            pik[i, :] = np.sum(den.dens_dim(y = y,
                                            paramMat = paramMat[i],
                                            precompute = precompute),
                                axis=1) + np.log(p[i])

        if normalized:
            pik -= logsumexp(pik, axis=0)[np.newaxis, :]
        #else:
        #    pik += np.log(p[0])

        if log:
            return pik
        else:
            return np.exp(pik)



    def sample_allocations(self, paramvec = None, paramMat = None, alpha=  None, p = None):
        pik = self.weights(paramvec = paramvec,
                           paramMat = paramMat,
                           alpha    = alpha,
                           p        = p,
                            log=False)
        
        P = np.cumsum(pik, axis=0)
        U = np.random.rand(self.n)
        alloc = np.zeros((self.n,), dtype=np.int)
        for k in range(self.K-1):
            alloc[U > P[k, :]] = k+1
        return alloc

    def dens_componentwise(self, paramvec = None, paramMat = None, alpha=  None, p = None, log=True):
        """
            Component-wise densities
        """
        p, alpha, paramMat = self._check_parameters(paramvec, paramMat , alpha , p )


        logfk = np.zeros((self.K, self.n, self.d))
        for i, den in enumerate(self.dens):
            logfk[i, :, :] = den.dens_dim(y=self.y, paramMat=paramMat[i])


        if log:
            return logfk
        else:
            return np.exp(logfk)
    
    def EMstep(self, 
               y          = None, 
               paramvec   = None,
               paramMat   = None,
               alpha      = None, 
               p          = None,
               update_param = True, 
               precompute = True):
        """
             EM algorithm step
             
             y            - (n x d) the data
             precompute   - (bool) store various object when computing the weights
             update_param - (bool) update the parameter in the object
        """
        
        return_paramvec = True
        if paramvec is None:
            return_paramvec = False
        
        if y is None:
            y = self.y
        
        if y is None:
            raise Exception("y must set or be in the object")
        
        p, alpha, paramMat = self._check_parameters(paramvec, paramMat , alpha , p )
        
        pik = self.weights(paramMat = paramMat,
                           p        = p,
                           log=False,
                           precompute = precompute)
        p = np.mean(pik, axis=1)
        if update_param:
            self.alpha = self.p_to_alpha(p)
        
        paramMat_out = []
        for i, den in enumerate(self.dens): 
            if p[i] > 0: 
                paramMat_out.append(den.EMstep(y = y, 
                                               p = pik[i,:].flatten(),
                           paramMat = paramMat[i], 
                           precompute = precompute, 
                           update_param = update_param))
            
        if return_paramvec:
            paramvec = self.paramMat_to_paramvec(p = p, paramMat = paramMat_out)
            return paramvec
        
        return p, self.p_to_alpha(p), paramMat_out 
   
