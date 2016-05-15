'''
Created on May 15, 2016

@author: jonaswallin
'''
import unittest
from Mixture.util import Bessel1approx, Bessel1approxOrdered
import numpy as np
import numpy.random as npr
import scipy.special as sps
import time



class Test(unittest.TestCase):

    n = 10000

    def Bessel_inLine(self, lower, upper , dec = 7):
        
        x = np.linspace(lower, upper, self.n + 1, endpoint = False)[1:]
        y_approx = Bessel1approx(x.flatten())
        y = sps.kv(1, x)
        
        print('sup_[{0},{1}] = {2}'.format(lower, upper, np.max(np.abs(y - y_approx))))
        
        np.testing.assert_array_almost_equal( y, y_approx, dec)
        
        x = (upper - lower) * npr.rand(self.n) + lower
        y_approx = Bessel1approx(x.flatten())
        y = sps.kv(1, x)
        np.testing.assert_array_almost_equal( y, y_approx, dec)
        
    def testSmallBessel(self):
        self.Bessel_inLine(0,0.001)
        self.Bessel_inLine(0.001,0.08)
        self.Bessel_inLine(0.08,0.5)
        self.Bessel_inLine(0.5,1.)
        
    def testLargeBessel(self):
        self.Bessel_inLine(1.,1.5)
        self.Bessel_inLine(1.5, 2)
        self.Bessel_inLine(2, 5)
        self.Bessel_inLine(5, 40)
        self.Bessel_inLine(40, 200)
        self.Bessel_inLine(200, 1e4)

    def test_speed(self):
        start =0.1
        x = np.linspace(start, start  + 100,  10**6)
        t0 = time.time()
        y_approx = Bessel1approx(x.flatten())  # @UnusedVariable
        t1 = time.time()
        time_approx = t1-t0
        t0 = time.time()
        y_approx = Bessel1approxOrdered(x.flatten())  # @UnusedVariable
        t1 = time.time()
        time_approxOrderd = t1-t0
        t0 = time.time()
        y = sps.kv(1, x)  # @UnusedVariable
        t1 = time.time()
        time_bessel = t1-t0
        print('approx(orded) = {0:.2e} ratios = {1:.2f}, {2:.2f} (unorded, exact)'.format(time_approxOrderd, 
                                                             time_approx/time_approxOrderd, 
                                                             time_bessel/ time_approxOrderd))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testSmallBessel']
    unittest.main()