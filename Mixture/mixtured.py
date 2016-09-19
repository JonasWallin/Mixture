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
        self.store_attr = ['_paramvec', '_p', '_hardClass', '_loglik',
                           '_computeProb']
        self.EMM_iter = 3

    def __call__(self, paramvec = None):
        """
            paramavec (d * k + (d-1) x 1) the prior vectors in matrix format
                     (0:d-2)            :  alpha which defines the prior prob as:
                                          \pi_i = np.exp( alpha_i) / 1 + sum(np.exp(alpha_i))
        """
        
        
        
        return np.sum(self.density(paramvec))
        
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
    
    
    def set_param_vec(self, paramvec):  
        """
        
        """
        
        self.paramvec = np.copy(paramvec)
        s = 0
        self.alpha = np.hstack((0, paramvec[s:(self.K-1)]))
        s += self.K-1
        self.alpha -= np.log(np.sum( np.exp(self.alpha))) 
        for  den in self.dens:
            den.set_param_vec(paramvec = paramvec[s:(s + den.k)]) 
            s += den.k
            
    def sample(self, n = 1, paramvec = None):
        
        if paramvec is None:
            paramvec = self.paramvec
        
        
        s = 0
        alpha = np.hstack((0, paramvec[s:(self.d-1)]))
        s += self.d-1
        alpha -= np.log(np.sum( np.exp(alpha))) 
        
        P = np.hstack((0.,np.exp(alpha)))
        P = np.cumsum(P)
        U = npr.rand(n)
        y = np.zeros((n,self.d))
        for i in range(self.K ): 
            index = (P[i] <= U) * (U <= P[i + 1]) == 1
            y[index,:] = self.dens[i].simulate(np.sum(index), paramvec = paramvec[s:(s + self.dens[i].k)])
            s += self.dens[i].k
            
        return y
            
    
    
    def density(self, paramvec = None, log = True):
        """
        
        """
        
        if paramvec is None:
            paramvec = self.paramvec
            
        if paramvec is None:
            raise Exception('need to set paramvec')
        
        s = 0
        alpha  = np.hstack((0, paramvec[s:(self.K-1)]))
        s     += self.K-1
        alpha -= np.log(np.sum( np.exp(alpha))) 
        
        logf  = np.zeros((self.n, self.d))
        for i, den in enumerate(self.dens):
            logf += np.exp(den.dens_dim( y = self.y, paramvec = paramvec[s:(s + den.k)]) + alpha[i])
            s += den.k
        
        logf = np.log(logf)
        if log == True:
            return logf
        else:
            return np.exp(logf)

    def density_1d(self, dim, y = None, paramvec = None, log = True):
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
        
        if paramvec is None:
            paramvec = self.paramvec
            
        if paramvec is None:
            raise Exception('need to set paramvec')
        
        n = len(y_)
        s = 0
        alpha = np.hstack((0, paramvec[s:(self.K-1)]))
        s += self.K-1
        alpha -= np.log(np.sum( np.exp(alpha))) 
        
        logf  = np.zeros((n, 1))
        for i, den in enumerate(self.dens):
            logf[:,0] += np.exp(den.dens_d( dim = dim, y = y_, paramvec = paramvec[s:(s + den.k)]) + alpha[i])
            s += den.k
        logf = np.log(logf)  
        
        if log == True:
            return logf
        else:
            return np.exp(logf)

    def weights(self, paramvec=None, log=True):
        if paramvec is None:
            paramvec = self.paramvec

        if paramvec is None:
            raise Exception('need to set paramvec')

        s = 0
        alpha  = np.hstack((0, paramvec[s:(self.K-1)]))
        s     += self.K-1
        alpha -= np.log(np.sum( np.exp(alpha)))

        # pik = np.zeros((self.K, self.n, self.d))
        # for i, den in enumerate(self.dens):
        #     pik[i, :, :] = den.dens_dim(y=self.y, paramvec=paramvec[s:(s + den.k)]) + alpha[i]
        #     s += den.k
        # pik -= logsumexp(pik, axis=0)[np.newaxis, :, :]

        pik = np.zeros((self.K, self.n))
        for i, den in enumerate(self.dens):
            pik[i, :] = np.sum(den.dens_dim(y=self.y, paramvec=paramvec[s:(s + den.k)]), axis=1) + alpha[i]
            s += den.k
        pik -= logsumexp(pik, axis=0)[np.newaxis, :]

        if log:
            return pik
        else:
            return np.exp(pik)

    def sample_allocations(self, paramvec=None):
        # pik = self.weights(paramvec, log=False)
        # P = np.cumsum(pik, axis=0)
        # U = np.random.rand(self.n, self.d)
        # alloc = np.zeros((self.n, self.d), dtype=np.int)
        # for k in range(self.K-1):
        #     alloc[U > P[k, :, :]] = k+1
        # return alloc
        pik = self.weights(paramvec, log=False)
        P = np.cumsum(pik, axis=0)
        U = np.random.rand(self.n)
        alloc = np.zeros((self.n,), dtype=np.int)
        for k in range(self.K-1):
            alloc[U > P[k, :]] = k+1
        return alloc

    def dens_componentwise(self, paramvec=None, log=True):
        """
            Component-wise densities
        """
        if paramvec is None:
            paramvec = self.paramvec

        if paramvec is None:
            raise Exception('need to set paramvec')

        s = self.K-1

        logfk = np.zeros((self.K, self.n, self.d))
        for i, den in enumerate(self.dens):
            logfk[i, :, :] = den.dens_dim(y=self.y, paramvec=paramvec[s:(s + den.k)])
            s += den.k

        if log:
            return logfk
        else:
            return np.exp(logfk)

    @property
    def paramvec(self):
        return self._paramvec

    @paramvec.setter
    def paramvec(self, paramvec):
        self._paramvec = paramvec
        self._p = None
        self._hardClass = None
        self._loglik = None
        self._computeProb = None

    def storeParam(self):
        self._storedParam = {par: getattr(self, attr) for attr in self.store_attr}

    def restoreParam(self):
        for attr in self.store_attr:
            setattr(self, self._storedParam[attr])

    @property
    def p(self):
        if self._p is None:
            self._p = self.weights(log=False)
        return self._p

    @property
    def Y(self):
        return self.y

    @property
    def loglik(self):
        if self._loglik is None:
            self._loglik = np.sum(self.computeProb)
        return self._loglik

    @property
    def F(self):
        return self.loglik

    @property
    def hardClass(self):
        if self._hardClass is None:
            self._hardClass = self.sample_allocations()
        return self._hardClass

    @property
    def computeProb(self):
        '''
            log likelihood pointwise
        '''
        if self._computeProb is None:
            self._computeProb = self.__call__()
        return self._computeProb

    def step(self):
        '''
            Optimization step
        '''

        def f(x):
            lik = - np.sum(self.p *
                           np.sum(self.dens_componentwise(np.hstack((np.zeros(K-1), x))), axis=2))
            if np.isnan(lik):
                return np.inf
            return lik

        # E step
        pi_k = np.sum(self.p, axis=1)

        # M step
        x = sp.optimize.fmin_powell(f, self.paramvec[self.K-1:], maxiter=self.EMM_iter)
        alpha = np.log(pi_k[1:]) - np.log(pi_k[0])
        self.set_param_vec(np.hstack((alpha.ravel(), x)))
