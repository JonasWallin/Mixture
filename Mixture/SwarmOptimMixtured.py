import numpy as np
from scipy.misc import logsumexp

from . import SwarmOptimMixObj

# TODO: Reset precomputed at better times.


class SwarmOptimMixtured(SwarmOptimMixObj):

    def __init__(self, mixture):
        super(SwarmOptimMixtured, self).__init__(mixture)
        self._store_attr = ['paramvec']
        self.reset_precomputed()

    def reset_precomputed(self):
        self._paramvec_curr = self._mixture.paramvec
        self._computedProb = None
        self._hardClass = None
        self._p = None
        self._F = None

    def step(self):
        '''
            Optimization step
        '''
        self._mixture.EMstep()
        self.reset_precomputed()

    def storeParam(self):
        param = super(SwarmOptimMixtured, self).storeParam()
        p, alpha, paramMat = self._mixture.get_paramMat()
        param['alpha'] = alpha
        param['paramMat'] = paramMat
        param['paramvec'] = self._mixture.get_paramvec()
        return param

    def restoreParam(self, param):
        paramMat = param.pop('paramMat')
        alpha = param['alpha']
        self._mixture.set_paramMat(paramMat, alpha=alpha)
        super(SwarmOptimMixtured, self).restoreParam(param)

    def computeProb(self):
        '''
            computes a vector of unormalised likelihood contribution
            for each observation Y

            pointwise log likelihood
        '''
        if not self._paramvec_curr is self._mixture.paramvec:
            self.reset_precomputed()
        if self._computedProb is None:
            self._computedProb = logsumexp(self._mixture.weights(normalized=False), axis=0)
        return self._computedProb

    def setMutated(self, ks, points):
        '''
            setup new classes for the ks centered at the points
        '''
        p, alpha, paramMats = self._mixture.get_paramMat()
        mu_mean = np.mean(np.vstack([paramMat[:, 1] for paramMat in paramMats]), axis=0)
        sigma_mean = np.mean(np.vstack([paramMat[:, 2] for paramMat in paramMats]), axis=0)
        nu_mean = np.mean(np.vstack([paramMat[:, 3] for paramMat in paramMats]), axis=0)
        p_median = np.median(p)
        points = points.reshape(-1, self._mixture.d)
        for i, k in enumerate(ks):
            paramMats[k] = np.vstack([points[i, :], np.zeros_like(mu_mean), sigma_mean, 2 * np.ones_like(nu_mean)]).T
            p[k] = p_median
        p /= np.sum(p)
        self._mixture.set_paramMat(paramMats, p=p)
        self.reset_precomputed()

    def dist(self, k_1):
        '''
            some distance measure between mixture objects and the other clases

            L1 distance between centers
        '''
        _, _, paramMats = self._mixture.get_paramMat()
        center_k1 = paramMats[k_1][:, 0]
        dist = np.zeros(self.K)
        for k, paramMat in enumerate(paramMats):
            dist[k] = np.sum(np.abs(paramMats[k][:, 0] - center_k1))
        return dist

    def hardClass(self):
        '''
            give a hard classification
        '''
        if not self._paramvec_curr is self._mixture.paramvec:
            self.reset_precomputed()
        if self._hardClass is None:
            self._hardClass = self._mixture.sample_allocations()
        return self._hardClass

    @property
    def K(self):
        '''
            (int) number of classes
        '''
        return self._mixture.K

    @property
    def p(self):
        '''
            (np.array) prob of belonging to a class
        '''
        if not self._paramvec_curr is self._mixture.paramvec:
            self.reset_precomputed()
        if self._p is None:
            self._p = np.sum(self._mixture.weights(log=False), axis=1)
        return self._p

    @property
    def F(self):
        '''
            current objective value: log likelihood
        '''
        if not self._paramvec_curr is self._mixture.paramvec:
            self.reset_precomputed()
        if self._F is None:
            self._F = np.sum(self.computeProb())
        return self._F

    @F.setter
    def F(self, F):
        '''
            set current objective value
        '''
        self._F = F

    @property
    def Y(self):
        '''
            (np.array) the actual data
        '''
        return self._mixture.y



