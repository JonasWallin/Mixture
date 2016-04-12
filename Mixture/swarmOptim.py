from __future__ import print_function
import numpy as np


def swarm(Mixobj, 
			   prec = 1, 
			   iteration = 10, 
			   mutate_iteration = 20, 
			   burst_iteration = 20, 
			   local_iter = 5, 
			   silent= False):
	"""	
		A mutation type algorithm to find good starting point for optimization or MCMC
		implimented for a general mixture model on the form:
					Y \sim \sum_{i=1}^K p_i f_i(Y)
		where f_i(Y) is a density

		*Mixobj*             - mixture object that needs to have:
							   - .step        -> otimization step
							   - .storeParam  -> storing the parameters needed to restore dict
							 				  should include ['F'] current object value

							   - .restorePram -> using dict from .storeParam restores object
							   - .computeProb -> computes a vector of unormalised likelihood contribution
							   				  -> for each observation Y
							   - .setMutated(ks, point)  -> from one observation Y_i, should setup new classes
							   							   for the ks starint at the points	
							   - .dist(k_1)        -> some distance measure between mixture objects and the other clases
							   - .hardClass        -> give a hard classification
							   - .K           -> (int) number of clases
							   - .p           -> (np.array) prob of beloning to a class
							   - .F           -> function that gives the current objective value
							   - .Y           -> (np.arry) the actual data

		*prec*               - precentage of data counting as outlier (deafult 1%)
		*iteration*          - number of iteration of mutation burst mutation
		*mutate_iteration*   - number of gibbs samples to run before testing if the mutation improved
		*burst_iteration*    - number of gibbs samples to run before testing if the mutation improved
	"""
	for j in range(iteration):  # @UnusedVariable
		if not silent:
			print('pre burnin iteration {j}'.format(j = j))
		mutate(Mixobj, prec, iteration = mutate_iteration, silent = silent)
		mutate(Mixobj, prec, iteration = mutate_iteration, silent = silent, rand_class =True)
		burst(Mixobj,        iteration = burst_iteration,   silent = silent)
		for k in range(local_iter):  # @UnusedVariable
			Mixobj.step()


def mutate(Mixobj, 
		   prec = 0.1, 
		   iteration = 10, 
		   silent = True, 
		   rand_class = False):
	'''
		mutate by setting a random class to outiler class
		*Mixobj*     - the main object
		*prec*       - [0,100] lower quantile what is defined as outlier (precentage)
		*iter*       - number of iteration in the Gibbs sampler 
  		*rand_class* - draw the class at random (else always take the smallest) 
	'''
	
	param0 = Mixobj.storeParam()
	point_ = drawOutlierPoint(Mixobj, prec)
	if rand_class:
		k  = np.random.randint(Mixobj.K)
	else:
		k  = np.argmin(Mixobj.p[:Mixobj.K])


	Mixobj.setMutated( [k], point_)
	
	for i in range(iteration):
		Mixobj.step()
		
	F = Mixobj.F
	if param0['F'] < F:
		if silent is False:
			if rand_class:
				print('random mutation %.2f < %.2f'%(param0['F'], F))
			else:
				print('min mutation %.2f < %.2f'%(param0['F'], F))
		return
	
	Mixobj.restoreParam( param0)

def drawOutlierPoint(Mixobj, prec = 0.1):
	"""
		draws a random outlier point (outlier defined through likelihood)
		*prec* - [0,1] lower quantile what is defined as outlier 
	"""
	l = Mixobj.computeProb( )
	l  = np.max(l, 1)
	index  = l < np.percentile(l, prec)
	
	points_ = Mixobj.Y[index,:]
	index_p = np.random.randint(points_.shape[0])
	point_ = points_[index_p,:]
	return(point_)


def burst(Mixobj, 
		 iteration = 10, 
		 randclass = False,
		 silent    = True):
	'''
		trying to burst two random classes (k, k2) and  then restimate the data

		*iteration* number of samples allowed after the burst
		*randclass* should the second class be chosen at random or through
					distance 
	'''
	param0 = Mixobj.storeParam()
	
	k  = np.random.randint(Mixobj.K)
	if randclass:
		k2 = k
		while k2 == k:
			k2 = np.random.randint(Mixobj.K)
	else:
		dist = Mixobj.dist(k)
		dist[k] = np.Inf
		k2 = np.argmin(dist)

	x = Mixobj.hardClass()
	index = (x == k) + (x == k2)
	y_index = Mixobj.Y[index,:]

	if y_index.shape[0] < 3: # two small samples
		return
	index_points = np.random.choice(y_index.shape[0], 2, replace=False)
	points = y_index[index_points,:]
	Mixobj.setMutated( [k, k2], points)


	for i in range(iteration):
		Mixobj.step()

	F = Mixobj.F
	if param0['F'] < F:
		if silent is False:
			print('burst mutation %.2f < %.2f'%(param0['F'], F))
		return

	Mixobj.restoreParam(param0)