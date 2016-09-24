'''
Created on Sep 16, 2016

@author: jonaswallin
'''
import unittest


from Mixture.density import NIG
from Mixture.density.purepython import NIG as NIG_py
from Mixture.density import mNIG as muNIG
#from Mixture.density.purepython import mNIG as muNIG
import scipy as sp
import numpy.random as npr
import scipy.special as sps
from Mixture.util import Bessel1approx, Bessel0approx
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
        self.simObj.set_param_Mat(self.paramMat)
        self.Y = self.simObj.simulate(n = n)


    def tearDown(self):
        pass


    def testCompare_1(self):
        
        dens = NIG()
        dens_py = NIG_py()
        np.testing.assert_array_almost_equal( dens.dens(self.Y[:,1],  paramvec = paramMatTrue[1,:]),
                                              dens_py.dens(self.Y[:,1], paramvec = paramMatTrue[1,:]), 
                                              6)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    
    