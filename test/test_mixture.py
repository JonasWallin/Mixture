'''
Created on Sep 17, 2016

@author: jonaswallin
'''
import unittest

from Mixture import mixOneDims
import numpy as np
import numpy.random as npr
from Mixture.density import mNIG
class Test(unittest.TestCase):


    def setUp(self):
        self.K = 3
        self.d = 2
        self.mixObj = mixOneDims(K = self.K, d = self.d)


    def tearDown(self):
        pass

    
    def test_alpha_to_p_and_back(self):
        
        alpha = [0,0]
        p_out = self.mixObj.alpha_to_p(alpha)
        alpha_out = self.mixObj.p_to_alpha(p_out)
        np.testing.assert_array_equal(alpha, alpha_out)
        
        p = np.random.rand(3)
        p /= np.sum(p)
        alpha_out = self.mixObj.p_to_alpha(p)
        p_out     = self.mixObj.alpha_to_p(alpha_out)
        np.testing.assert_almost_equal(p, p_out, decimal = 10)
        
    def test_simulate(self):
        """
            testing if simulation works
        """
        K = 2
        d = 2
        npr.seed(16)
        mixObj = mixOneDims(K=K, d=d)

        mixObj.set_densites([mNIG(d=d) for k in range(K)])  # @UnusedVariable
        paramMAt = [np.array([[1.1, 1.12, 0.1, 0],
                             [-1,  0,2 , -4] ]),
            np.array([[-2, 0, 0.3, 0],
                             [1,  0, 2 , -4] ])]
        mixObj.set_paramMat(alpha = [0],paramMat = paramMAt )
        
        Y = mixObj.sample(n = 10)  # @UnusedVariable

    def test_simple_estimate(self):
        """
            estimation on very well seperated dta set
        """
        K = 2
        d = 2
        iteration = 50
        npr.seed(11)
        mixObj = mixOneDims(K=K, d=d)
        
        mixObj.set_densites([mNIG(d=d) for k in range(K)])
        paramMat_true = [np.array([[1.1, 1.12, 0.1, 0],
                                     [-1,  0,2 , -4] ]),
                    np.array([[-2, 0, 0.3, 0],
                                     [1,  0, 2 , -4] ])]
        alpha_true = [0]
        mixObj.set_paramMat(alpha = alpha_true,paramMat = paramMat_true )
        
        
        Y = mixObj.sample(n = 2000)
        
        
       
        mixObj.set_data(Y)
        
        paramMat = [npr.randn(2,4),npr.randn(2,4)]
        paramMat[0][0,0] = 1.1 
        paramMat[1][0,0] = -2 
        alpha = np.array(alpha_true)
        for i in range(iteration):
            p, alpha, paramMat = mixObj.EMstep(alpha = alpha, paramMat = paramMat )
        
        
        np.testing.assert_array_almost_equal(np.array(paramMat), np.array(paramMat_true), decimal = 0) 
        np.testing.assert_array_almost_equal(np.array(alpha), np.array(alpha_true), decimal = 1)  

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()