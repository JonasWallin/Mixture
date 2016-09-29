'''
Created on Sep 16, 2016

@author: jonaswallin
'''
import unittest


from Mixture.density import NIG
from Mixture.density import mNIGc
from Mixture.density import mNIG as muNIG
from Mixture.density.purepython import mNIG as muNIG_python
import scipy as sp
import numpy.random as npr
import scipy.special as sps
import numpy as np


paramMatTrue = np.array([[1.1, 2.12, 0.1, 0.1],
                             [-2,  2.12, -1.5 , -1]   ,
                             [0,  0, 0.4 , 1] 
                            ])
class Test_mNIG(unittest.TestCase):


    def setUp(self):
        npr.seed(12326)
        n  = 5000
        self.paramMat = paramMatTrue
        d = self.paramMat.shape[0]
        self.simObj = muNIG(d = d)
        self.simObj_py = muNIG_python(d = d)
        self.simObj.set_param_Mat(self.paramMat)
        self.Y = self.simObj.simulate(n = n)


    def tearDown(self):
        pass


    def testEM(self):
        
        
        paramMat = np.zeros_like(self.paramMat)
        for i in range(300): 
            paramMat = self.simObj.EMstep(y = self.Y, paramMat = paramMat)
        np.testing.assert_array_almost_equal(paramMat, self.paramMat, decimal = 1)  
    def testEM_py(self):
        
        
        paramMat = np.zeros_like(self.paramMat)
        for i in range(300): 
            paramMat = self.simObj_py.EMstep(y = self.Y, paramMat = paramMat)
        np.testing.assert_array_almost_equal(paramMat, self.paramMat, decimal = 1)  

class Test_mNIGc(unittest.TestCase):


    def setUp(self):
        npr.seed(12326)
        n  = 5000
        self.paramMat = paramMatTrue
        d = self.paramMat.shape[0]
        self.simObj = mNIGc(d = d)
        self.simObj_py = muNIG_python(d = d)
        self.simObj.set_param_Mat(self.paramMat)
        self.Y = self.simObj.simulate(n = n)


    def tearDown(self):
        pass


    def testEM(self):
        
        
        paramMat = np.zeros_like(self.paramMat)
        for i in range(300): 
            paramMat = self.simObj.EMstep(y = self.Y, paramMat = paramMat)
        np.testing.assert_array_almost_equal(paramMat, self.paramMat, decimal = 1) 
    
    
    def testEM_wprior(self):
        
        prior = list()
        prior.append(np.array([[1.,1],
                               [1,1],
                               [1,1],
                               [1,1]]))
        prior.append(np.array([[1.,1],
                               [1,1],
                               [1,1],
                               [1,1]]))
        prior.append(np.array([[1.,1],
                               [1,1],
                               [1,1],
                               [1,1]]))
        self.simObj.set_prior(prior)
        paramMat = np.zeros_like(self.paramMat)
        for i in range(300): 
            paramMat = self.simObj.EMstep(y = self.Y, paramMat = paramMat)
        np.testing.assert_array_almost_equal(paramMat, self.paramMat, decimal = 1) 
        
         



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()