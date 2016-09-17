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


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()