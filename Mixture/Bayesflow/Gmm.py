"""
	Settingup a swarmoptim object for Bayesflow, using a
	Gibbs sampler as step
	see swarmOptim instruction

"""
import copy as cp
import numpy as np

class GMMoptim(object):

	"""
		optimobject for GMM to run swarm

	"""


	def __init__(self, GMM):
		"""

		"""
		self.GMM = GMM

	def step(self):

		self.GMM.sample()

	def storeParam(self, F = None):

		
		if F is None:
			F = self.F
		
		res = {'F':     F,
			   'p':     cp.deepcopy(self.GMM.p),
		       'mu':    cp.deepcopy(self.GMM.mu), 
		       'sigma': cp.deepcopy(self.GMM.sigma)}

		return res

	def restoreParam(self, param):

		self.GMM.p = np.zeros_like(param['p'])
		self.GMM.p[:] = param['p']
		self.GMM.set_mu(param['mu'])
		self.GMM.set_sigma(param['sigma'])


	def computeProb(self):

		self.GMM.compute_ProbX(norm=False)
		l = np.zeros_like(self.GMM.prob_X)
		l[:] = self.GMM.prob_X[:]
		return l

	def setMutated(self, ks, points):

		for i, k in enumerate(ks):
			if len(points.shape) == 1:
				self.GMM.mu[k][:] = points[:]
			else:
				self.GMM.mu[k][:] = points[i,:]
			self.GMM.sigma[k] = 50 * np.diag(
									 np.diag(self.GMM.sigma[k])) 
		self.GMM.updata_mudata()
		self.GMM.sample_x()	


	def dist(self, k):

		d_ = np.random.randint(self.GMM.d)
		dist = np.abs(( np.array( self.GMM.mu)[k, d_]
					   - np.array(self.GMM.mu)[:, d_]))

		return dist

	def hardClass(self):

		x  = np.zeros_like(self.GMM.x)
		x[:] = self.GMM.x[:]
		return x

	@property
	def K(self):

		return self.GMM.K

	@property
	def p(self):

		p    = np.zeros_like(self.GMM.p)
		p[:] = self.GMM.p[:]
		return p 

	@property
	def F(self):

		return self.GMM.calc_lik()

	@property
	def Y(self):

		return self.GMM.data
