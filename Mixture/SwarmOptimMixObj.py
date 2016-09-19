class SwarmOptimMixObj(object):
    '''
        Abstract class for MixObj to be used in SwarmOptim.
    '''

    def __init__(self, mixture):

        self._mixture = mixture
        self._store_attr = None

    def step(self):
        '''
            Optimization step
        '''
        raise NotImplementedError

    def storeParam(self):
        '''
            storing the parameters needed to restore dict
            should include ['F'] current object value
        '''
        store_dict = {par: getattr(self._mixture, attr) for attr in self._store_attr}
        store_dict['F'] = self.F
        return store_dict

    def restoreParam(self, param):
        '''
            using dict from .storeParam restores object
        '''
        for par in param:
            if not par == 'F':
                setattr(self._mixture, param[par])
        self.F = F

    def computeProb(self):
        '''
            computes a vector of unormalised likelihood contribution
            for each observation Y
        '''
        raise NotImplementedError

    def setMutated(self, ks, point):
        '''
            from one observation Y_i, should setup new classes
            for the ks starint at the points
        '''
        raise NotImplementedError

    def dist(self, k_1):
        '''
            some distance measure between mixture objects and the other clases
        '''
        raise NotImplementedError

    def hardClass(self):
        '''
            give a hard classification
        '''
        raise NotImplementedError

    @property
    def K(self):
        '''
            (int) number of classes
        '''
        raise AttributeError

    @property
    def p(self):
        '''
            (np.array) prob of belonging to a class
        '''
        raise AttributeError

    @property
    def F(self):
        '''
            current objective value
        '''
        raise AttributeError

    @F.setter
    def F(self):
        '''
            set current objective value
        '''
        raise NotImplementedError

    @property
    def Y(self):
        '''
            (np.array) the actual data
        '''
        raise AttributeError
