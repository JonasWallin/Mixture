
import numpy as np
import matplotlib.pyplot as plt
from Mixture.density import NIG, mNIG
from Mixture import mixOneDims
import numpy.random as npr


K = 2
d = 2
iteration = 100
npr.seed(16)
mixObj = mixOneDims(K=K, d=d)

mixObj.set_densites([mNIG(d=d) for k in range(K)])
paramMat = [np.array([[1.1, 1.12, 0.1, 0],
                             [-1,  0,2 , -4] ]),
            np.array([[-2, 0, 0.3, 0],
                             [1,  0, 2 , -4] ])]
alpha = [0]
mixObj.set_paramMat(alpha = alpha,paramMat = paramMat )


Y = mixObj.sample(n = 20000)


fig, axarr = plt.subplots(2, 1)

axarr[0].hist(Y[:, 0], 200, normed=True, histtype='stepfilled', alpha=0.2)

axarr[1].hist(Y[:, 1], 200, normed=True, histtype='stepfilled', alpha=0.2)

mixObj.set_data(Y)

paramMat = [npr.randn(2,4),npr.randn(2,4)]
for i in range(d):
    x_0 = np.linspace(np.min(Y[:,i]), np.max(Y[:,i]), 200)
    d_0 = mixObj.density_1d(dim = i, y = x_0, log=False, alpha = alpha, paramMat = paramMat)
    axarr[i].plot(x_0, d_0, color='red')
for i in range(iteration):
    p, alpha, paramMat = mixObj.EMstep(alpha = alpha, paramMat = paramMat )
    

for i in range(d):
    x_0 = np.linspace(np.min(Y[:,i]), np.max(Y[:,i]), 200)
    d_0 = mixObj.density_1d(dim = i, y = x_0, log=False, alpha = alpha, paramMat = paramMat)
    axarr[i].plot(x_0, d_0)
plt.show()

