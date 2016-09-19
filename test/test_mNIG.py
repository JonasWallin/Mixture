'''
Created on Sep 16, 2016

@author: jonaswallin
'''
import unittest


from Mixture.density import NIG
from Mixture.density import mNIG as muNIG
import scipy as sp
import numpy.random as npr
import scipy.special as sps
import numpy as np



class Test_mNIG(unittest.TestCase):


    def setUp(self):
        npr.seed(12326)
        n  = 4000
        self.paramMat = np.array([[1.1, 2.12, 0.1, 0.1],
                             [-2,  2.12, -2 , -1]   ,
                             [0,  0, 0.4 , 1] 
                            ])
        d = self.paramMat.shape[0]
        self.simObj = muNIG(d = d)
        self.simObj.set_param_Mat(self.paramMat)
        self.Y = self.simObj.simulate(n = n)


    def tearDown(self):
        pass


    def testName(self):
        
        
        paramMat = np.zeros_like(self.paramMat)
        for i in range(100): 
            paramMat = self.simObj.EMstep(y = self.Y, paramMat = paramMat)
        
        np.testing.assert_array_almost_equal(paramMat, self.paramMat, decimal = 1)  


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()