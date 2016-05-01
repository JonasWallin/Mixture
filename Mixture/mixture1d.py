'''
    A generic class for densites of class of 1d independent densites to make d dimensional model
Created on Apr 27, 2016

@author: jonaswallin
'''


from __future__ import division
import numpy as np
import copy as cp

class mixture1d(object):
    """
        the object consits of d one dimensional densities (class).
        the denisties (class) needs the following funcitons
        - set_param     -  set the prior for the density
        - __call__      -  returns the log density
        - set_param_vec -  same as set_prior but now the data is in vector form 
        - set_data      - set the data
        - k             - number of parameters
    
    """
    
    def __init__(self, d = None):
        """
            d - dimension of data
        """

        self.d = d
        self.dens = None
        
        
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
            self.d = np.shape(y)[2]
        
        
        for i, den in enumerate(self.dens):
            den.set_prior_vec(y[:,i])
            
        self.n = np.shape(y)[0]
            
    
    def set_densites(self, dens):
        """
            dens - (d x 1) list of densites - denites needs that __call__(x) returns loglikehood of x
        """
        
        if self.d is not None:
            if len(dens) != self.d:
                raise Exception('the dimenstion of the data is {0} and d = {1}'.format( len(dens), self.d))
            
        self.dens = cp.deepcopy( dens) # maybe better not use copy?
        
    def set_param(self, params):
        """
            params -  (d x 1) list of param corresponding to denisty in self.dens
        """
        if self.dens is None:
            raise Exception('need to set denity first')
        
        for den, param in zip(self.dens, params):
            den.set_prior(param)
            
            
    def set_param_vec(self, paramvecs):
        """
            paramvecs (d x k) matrix where each row corresponds to the prior in each vector
        """
        
        
        if self.dens is None:
            raise Exception('need to set denity first')
        
        for i, den in enumerate(self.dens):
            den.set_param_vec(paramvecs[i,:])
            
            
    def __call__(self, priorvec):
        """
            priorvec (d * k + (d-1) x 1) the prior vectors in matrix format
                     (0:d-2)            :  alpha which defines the prior prob as:
                                          \pi_i = np.exp( alpha_i) / 1 + sum(np.exp(alpha_i))
        """
        alpha = np.hstack((0, priorvec[:(self.d-1)]))
        alpha -= np.log(np.sum( np.exp(alpha))) 
        priorvecs = priorvec[self.d:].reshape(self.d, self.dens[0].k)
        logf = np.zeros(self.n, self.d)
        
        for i, den in enumerate(self.dens):
            logf[:,i] = den(priorvecs[i,:]) + alpha[i]
         
        return np.sum(logf)
        
        
                