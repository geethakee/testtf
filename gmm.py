"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Homework 3
The gmm function takes in as input a data matrix X and number of gaussians in the mixture model
The implementation assumes that the covariance matrix is shared and is a spherical diagonal covariance matrix
You have to fill in the pieces whereever you see ????
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.misc import logsumexp

def gmm(trainX, num_K, num_iter = 20):
	"""
		input trainX is a N by D matrix containing N datapoints, num_K is the number of clusters or mixture components desired.
		num_iter is the maximum number of EM iterations run over the dataset
		For the output:
			- mu which is K by D, the coordinates of the means
			- pk, which is K by 1 and represents the cluster proportions
			- zk, which is N by K, has at each z(n,k) the probability that the nth data point belongs to cluster k, specifying the cluster associated with each data point
			- si2 is the estimated (shared) variance of the data
			- BIC is the Bayesian Information Criterion (smaller BIC is better)
	"""
	N = trainX.shape[0]
	D = trainX.shape[1]

	try:
		if num_K >= N:
			raise AssertionError
	except AssertionError:
		print("You are trying too many clusters")
		raise

	si2 = 1 # Initialization of variance
	pk = np.ones([num_K,1]) # Uniformly initialize cluster proportions
	mu = np.random.randn(num_K, D) # Random initialization of clusters

	zk = np.zeros([N,num_K]) # Matrix containing cluster membership probability for each point

	for iter in range(0,num_iter):
		"""
			E-Step
			In the first step, we find the expected log-likelihood of the data which is equivalent to:
			finding cluster assignments for each point probabilistically
			In this section, you will calculate the values of zk(n,k) for all n and k according to current values of si2, pk and mu
		"""
		????

		"""
			M-step
			Compute the GMM parameters from the expressions which you have in your writeup
		"""

		# Estimate new value of pk
		????

		# Estimate new value for mu
		????

		# Estimate new value for sigma^2
		????

	# Computing the expected likelihood of data for the optimal parameters computed
	????

	# Compute the BIC for the current cluster using the expected log-likelihood
	????

	return mu, pk, zk, si2, BIC
