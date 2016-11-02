'''
Created on May 1, 2016

@author: jonaswallin
'''
import unittest

from Mixture.density import NIG, NIGc
from Mixture.density.purepython import NIG as pNIG
import scipy as sp
import numpy.random as npr
import scipy.special as sps
import numpy as np


def f_GIG(x, p, a ,b):
    """
        the univariate density for generalised inverse Gaussian distribution
    """
    f = np.sqrt(a / b)**p /(2  *sps.kn(p, np.sqrt(a * b)) )
    f *= x**(p-1)
    f *= np.exp( - (a * x + b/x)/2)
    return f
    
def EV(NIG_obj, y, power = 1):
            
    delta_mu = NIG_obj.delta - NIG_obj.mu
    y_ = (y - delta_mu ) /NIG_obj.sigma
    b = NIG_obj.nu +  y_**2 
    p  = -1.
    a = NIG_obj.mu**2 / NIG_obj.sigma**2 + NIG_obj.nu
    def f(x):
        return x**power * f_GIG(x, p , a, b)
    res=  sp.integrate.quad(f, 0, np.inf)[0]
    return res 


class Test_NIG_conj(unittest.TestCase):
    
 

    def setUp(self):
        npr.seed(12346)
        n  = 10000
        self.simObj = NIGc(paramvec = [1.1, 2.12,0.1,0.1])
        self.Y = self.simObj.simulate(n = n)
        
    
    def testEM(self):
        """
            without prior
        """
        paramvec_true = self.simObj.get_param_vec()
        #test nu
        if 1:
            
            paramvec      = np.array(paramvec_true)
            paramvec[2]   = 0
            # test nu
            for i in range(100): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [0,0,1,0])
            
            np.testing.assert_approx_equal(paramvec[2], paramvec_true[2], significant = 0.5)
        
        
        # test delta, mu
        if 1:
            paramvec      = np.array(paramvec_true)
            paramvec[0]   = 0
            paramvec[1]   = 0
            # test nu
            for i in range(20): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [1,1,0,0])
            np.testing.assert_approx_equal(paramvec[0], paramvec_true[0], significant = 2)
            np.testing.assert_approx_equal(paramvec[1], paramvec_true[1], significant = 2)

        # test sigma
        if 1:
            
            paramvec      = np.array(paramvec_true)
            paramvec[3]   = 0
           
            for i in range(40): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [0,0,0,1])
            
            np.testing.assert_approx_equal(paramvec[3], paramvec_true[3], significant = 0.5)  
            
        # test all   if 0:
            
        paramvec      = np.zeros_like(paramvec_true)
       
        for i in range(100): 
            paramvec = self.simObj.EMstep(y = self.Y, 
                                          paramvec = paramvec,
                                          update = [1,1,1,1])
        
        np.testing.assert_array_almost_equal(paramvec, paramvec_true, decimal = 1)  
              

    def testEM_withP(self):
        """
            with prior
        """
        paramvec_true = self.simObj.get_param_vec()
        #test nu
        if 1:
            
            paramvec      = np.array(paramvec_true)
            paramvec[2]   = 0
            
            prior = np.array([[0., 2.],
                              [0., 2.],
                              [10., 2.],
                              [10., 2.],])
            prior2 = np.array([[10, 10.**5],
                              [10, 10.**5],
                              [ 10., 10**5],
                              [ 10., 10**5],])
            # test nu
            for i in range(100): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [0,0,1,0],
                                              prior  = prior)
            
            np.testing.assert_approx_equal(paramvec[2], paramvec_true[2], significant = 0.5)
            
            paramvec2 = np.array(paramvec)
            for i in range(20): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [0,0,1,0],
                                              prior  = prior2)            
            np.testing.assert_array_less(paramvec2[2], paramvec[2])
            
        # test delta, mu
        if 1:
            paramvec      = np.array(paramvec_true)
            paramvec[0]   = 0
            paramvec[1]   = 0
            # test nu
            for i in range(20): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [1,1,0,0],
                                              prior= prior)
            np.testing.assert_approx_equal(paramvec[0], paramvec_true[0], significant = 2)
            np.testing.assert_approx_equal(paramvec[1], paramvec_true[1], significant = 2)
            
            paramvec2 = np.array(paramvec)
            for i in range(20): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [1,1,0,0],
                                              prior  = prior2)   
            np.testing.assert_array_less(paramvec2[0], paramvec[0])    
            np.testing.assert_array_less(paramvec2[1], paramvec[1])

        # test sigma
        if 1:
            
            paramvec      = np.array(paramvec_true)
            paramvec[3]   = 0
           
            for i in range(40): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [0,0,0,1],
                                              prior = prior)
            
            np.testing.assert_approx_equal(paramvec[3], paramvec_true[3], significant = 0.5)  
            paramvec2 = np.array(paramvec)
            for i in range(20): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [0,0,0,1],
                                              prior  = prior2)   
            np.testing.assert_array_less(paramvec2[3], paramvec[3]) 
        # test all   if 0:
            
        paramvec      = np.zeros_like(paramvec_true)
       
        for i in range(100): 
            paramvec = self.simObj.EMstep(y = self.Y, 
                                          paramvec = paramvec,
                                          update = [1,1,1,1],
                                          prior = prior)

      
        np.testing.assert_array_almost_equal(paramvec, paramvec_true, decimal = 1)  
         

class Test_pNIG(unittest.TestCase):
    
 

    def setUp(self):
        npr.seed(123456)
        n  = 4000
        self.simObj = pNIG(paramvec = [1.1, 2.12,0.1,0.1])
        self.Y = self.simObj.simulate(n = n)
        
        
    def testEV(self):
        """
            testing if the expectation of the latent variance is correct
        """
        y  = 4*npr.randn(1)
        EV_, EiV_ = self.simObj.EV(y)
        np.testing.assert_approx_equal(EV_, EV(self.simObj, y), significant = 7)
        np.testing.assert_approx_equal(EiV_, EV(self.simObj, y, power=-1), significant = 7)
    
    def testEM(self):
        """
            pass
        """
        paramvec_true = self.simObj.get_param_vec()
        #test nu
        if 1:
            
            paramvec      = np.array(paramvec_true)
            paramvec[2]   = 0
            # test nu
            for i in range(80): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [0,0,1,0])
            np.testing.assert_approx_equal(paramvec[2], paramvec_true[2], significant = 0)
        
        
        # test delta, mu
        if 1:
            paramvec      = np.array(paramvec_true)
            paramvec[0]   = 0
            paramvec[1]   = 0
            # test nu
            for i in range(20): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [1,1,0,0])
            np.testing.assert_approx_equal(paramvec[0], paramvec_true[0], significant = 2)
            np.testing.assert_approx_equal(paramvec[1], paramvec_true[1], significant = 2)

        # test sigma
        if 1:
            
            paramvec      = np.array(paramvec_true)
            paramvec[3]   = 0
           
            for i in range(20): 
                paramvec = self.simObj.EMstep(y = self.Y, 
                                              paramvec = paramvec,
                                              update = [0,0,0,1])
            
            np.testing.assert_approx_equal(paramvec[3], paramvec_true[3], significant = 0)  
            
        # test all   if 0:
            
        paramvec      = np.zeros_like(paramvec_true)
       
        for i in range(100): 
            paramvec = self.simObj.EMstep(y = self.Y, 
                                          paramvec = paramvec,
                                          update = [1,1,1,1])
        
        np.testing.assert_array_almost_equal(paramvec, paramvec_true, decimal = 1)  
               

 

class Test_NIG(unittest.TestCase):


    def setUp(self):
        n  = 1000
        self.simObj = NIG(paramvec = [1.1, 2.12,0.1,0.1])
        self.Y = self.simObj.simulate(n = n)
        
    def testIntegrare(self):
        """
            testing that the function approxiamte integrate to one
        """
        f = lambda x: np.exp(self.simObj(y = x) )
        res=  sp.integrate.quad(f, -np.inf, np.inf)
        np.testing.assert_approx_equal(1., res[0], significant = 5)

    def testSimulate(self):
        """
            testing that the mean is approx delta
        """
        np.testing.assert_approx_equal( self.simObj.delta , np.mean(self.Y), 1e-2)
 
class Test_NIG_vs_pNIG(unittest.TestCase):


    def setUp(self):
        n  = 1000
        self.simObj = NIG(paramvec = [1.1, 2.12,0.1,0.1])
        self.simObjp = pNIG(paramvec = [1.1, 2.12,0.1,0.1])
        self.Y = self.simObj.simulate(n = n)
        
    def test_f(self):
        """
            testing that the function approxiamte integrate to one
        """
        np.testing.assert_array_almost_equal(self.simObj(y = self.Y), self.simObjp(y = self.Y), decimal = 5)                
                                        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()