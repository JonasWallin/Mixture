import numpy as np

from . import SwarmOptimMixObj


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

    def computeProb(self):
        '''
            computes a vector of unormalised likelihood contribution
            for each observation Y

            pointwise log likelihood
        '''
        if not self._paramvec_curr is self._mixture.paramvec:
            self.reset_precomputed()
        if self._computedProb is None:
            self._computedProb = self._mixture.__call__()
        return self._computedProb

    def setMutated(self, ks, points):
        '''
            setup new classes for the ks centered at the points
        '''
        p, alpha, paramMats = self._mixture.get_paramMat()
        mu_mean = np.mean(np.hstack([paramMat[:, 1] for paramMat in paramMats]), axis=1)
        sigma_mean = np.mean(np.hstack([paramMat[:, 2] for paramMat in paramMats]), axis=1)
        nu_mean = np.mean(np.hstack([paramMat[:, 3] for paramMat in paramMats]), axis=1)
        p_median = np.median(p)
        for k, point in zip(ks, points):
            paramMats[k] = np.hstack([point.reshape(-1, 1), mu_mean, sigma_mean, nu_mean])
            p[k] = p_median
        p /= np.sum(p)
        self._mixture.set_paramMat(paramMats, p=p)

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
            self._hardClass = self.sample_allocations()
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
            self._p = self._mixture.weights(log=False)
        return self._p

    @property
    def F(self):
        '''
            current objective value: log likelihood
        '''
        if not self._paramvec_curr is self._mixture.paramvec:
            self.reset_precomputed()
        if self._F is None:
            self._F = np.sum(self.computeProb)
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



