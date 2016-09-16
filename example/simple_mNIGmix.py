
import numpy as np
import matplotlib.pyplot as plt
from Mixture.density import NIG, mNIG
from Mixture import mixOneDims
import numpy.random as npr


K = 2
d = 2
npr.seed(16)
x0 = npr.randn(4*K*d  + (K - 1))
mixObj = mixOneDims(K=K, d=d)
mixObj.set_densites([mNIG(d=d) for k in range(K)])

Y = mixObj.sample(n = 10, paramvec = x0)