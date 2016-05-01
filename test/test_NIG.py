'''
Created on May 1, 2016

@author: jonaswallin
'''
import unittest

from Mixture.density import NIG
import scipy as sp
import numpy.random as npr
import numpy as np
class Test(unittest.TestCase):


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
        np.testing.assert_approx_equal(1., res[0], 1e-5)

    def testSimulate(self):
        """
            testing that the mean is approx delta
        """
        np.testing.assert_approx_equal( self.simObj.delta , np.mean(self.Y), 1e-2)
                                        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()