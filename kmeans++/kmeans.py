"""
Simple implementation of (linear) k-means algorithm with:
1. Random initial centroids
2. Initial centroids selected by kmeans++ heuristic.
"""

import math
import random
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def edist(p, q):
	""" Compute the Euclidean distance between two points p, q. """
	if len(p) != len(q):
		raise ValueError, "lengths must match!"
	
	sqSum = sum(map(lambda x,y: (x-y)**2, p, q))
	return math.sqrt(sqSum)
	
def pearson(x, y):
	""" Compute the Pearson correlation between two points x, y. 
		This can be used as a "distance" between x, y.
	"""
	if len(x) != len(y):
		raise ValueError, "lengths must match!"
	n = len(x)
	sumx = sum(x)
	sumy = sum(y)
	sumxy = sum(map(lambda a,b: a*b, x, y))
	sqSumx = sum(map(lambda a: a**2, x))
	sqSumy = sum(map(lambda a: a**2, y))
	
	nu = sumxy - float(sumx) * sumy/n
	de = math.sqrt((float(sqSumx) - sumx**2/n) * (float(sqSumy) - sumy**2/n))
	if de == 0: return 0 # no correlation
	else: return nu/de
		
def random_init_from_data(X, k, dist=None):
	""" Choose k initial *different* centroids randomly from input data X. 
		
		Params: 
			X:  n x d matrix of n samples. Type: np.ndarray  
				Each sample is d-dimensional.
			k: Number of centroids
		Returns:
			List of k centroids.
	"""
	n = X.shape[0] 
	if k > n: 
		k = n
	k_indices = random.sample(xrange(n), k)
	cs = X[k_indices]
	return cs.tolist()
	
def random_init2(X, k, dist=None):
	""" Choose k initial *different* centroids randomly and not necessarily from data X.
		
		Params: 
			X:  n x d matrix of n samples. Type: np.ndarray 
				Each sample is d-dimensional.
			k:  Number of centroids
		Returns:
			List of k centroids.
	"""
	
	n, d = X.shape
	if k > n:
		k = n
		
	# val_range[i] = (min_i, max_i) -> [min, max] range for i-th coordinate of X.	
	val_range = zip(np.min(X, 0), np.max(X, 0))
	
	# Select k random *different* initial centroids.
	set_c = set()
	
	while len(set_c) < k:
		set_c.add((random.random() * (val_range[i][1] - val_range[i][0]) + val_range[i][0] for i in range(d)))
	
	assert len(set_c) == k
	cs = map(list, set_c)
	return cs
	
def init_plusplus(X, k, dist=edist):
	""" Choose k initial *different* centroids randomly using the k-means++ heuristic. 
		See the paper: Sergei Vassilvitskii, and David Arthur, K-means++: The advantages of careful seeding.
    	This often gives better clustering results, but it is slower than random initial version.
	"""
	
	X = X.tolist()
	set_c = set()
	
	from utils import WeightedRandomSelector
	# Choose the first centroid randomly from data X
	cid = random.randrange(len(X))
	set_c.add(tuple(X[cid]))
	del X[cid]
	
	i = 0
	while len(set_c) < k and i < k*5:
		min_dists = [min(dist(c, p) for c in set_c) for p in X]
		selector = WeightedRandomSelector(min_dists)
		cid = selector()
		set_c.add(tuple(X[cid]))
		i += 1
		del X[cid]
	
	cs = map(list, set_c)
	if len(cs) < k:
		cs.extend([cs[0]]*(k-len(cs)))	
		
	return cs
	
def kmeans(X, init=init_plusplus, dist=edist, k=5, n_iter=1000, tol=1e-10):
	"""	k-Means algorithm of clustering data X into k clusters.
		Params:
			X: n x d matrix of n samples. np.ndarray type
			dist: Distance metric function
			init: specify how to select k initial centroids.
			n_iter: the maximum number of iterations of the algorithm.
			tol: tolerance value for convergence
		Returns:
			clusters: {cluster_id: [list of samples in the cluster]}
			cs: list of centroids.
	"""
	n, d = X.shape
	
	# Get k initial centroids
	cs = init(X, k, dist=dist)
	clusters = None
	
	for t in range(n_iter):
		print 'Iteration %d:' %t
		# First, assign each data point to a cluster specified by its nearest centroids.
		tmpClusters = defaultdict(list)
		
		for p in X:
			_, cid = min([(dist(p, cs[id]), id) for id in range(len(cs))])
			tmpClusters[cid].append(p)
		
		# Compute new centroid for each cluster. 
		for i in range(k):
			# Get the list of X that belong to i-th cluster
			cPoints = tmpClusters[i]
			# Get the size of i-th cluster
			cSize = len(cPoints)
			
			oldcs = cs[:]
			# New centroid for i-th cluster: simply computing the average of the cluster's X 
			if cSize > 0:
				total = map(sum, zip(*cPoints))
				avg = map(lambda x: float(x)/cSize, total)
				cs[i] = avg # new centroid of i-th cluster
		
		# Check if convergence
		diff = 0.0
		for i in range(k):
			diff += dist(cs[i], oldcs[i])
		
		if diff <= tol: 
			break
			
	clusters = tmpClusters
	for k in clusters:
		clusters[k] = np.asarray(clusters[k], dtype='float64')
	
	return clusters, cs
	
def gen_2D_data(n=1000):
	""" Generate 9*n data points from 9 Gaussian distributions for testing.
		Returns:
			X: 9n x 2 matrix of 9n samples. Each sample is 2-dimensional
			   Type: ndarray.
	"""
	X = []
	means = [(0,0), (0,3), (0,6), (3,0), (3,3,), (3,6), (-3,0), (-3,3), (-3,6)]
	cov = 0.1*np.eye(2)
	for i in range(len(means)):
		tmp = np.random.multivariate_normal(means[i], cov, n)	
		for p in tmp:
			X.append([p[0],p[1]])
	# randomly shuffle data points
	random.shuffle(X)
	return np.asarray(X, dtype='float64')

def plotClusters(clusters, centroids, figname):
	""" Plot clusters. 
		Assume that #clusters <= 6 for simplicity in color scheme.
		
		Params:
			clusters: {cluster_id: [list of samples in the cluster]}
			centroidss: list of centroids.
	"""
	n_cluster = len(centroids)
	
	#marker_list = ['o', '*', '+', 'x', 'v', 's', 'd', 'p', '^']
	marker_symbol = 'o'
	color_list = ['r', 'g', 'b', 'y', 'm', 'k', 'c', 'burlywood', 'chartreuse']
	
	fig = plt.figure()
	fig.clf()
	
	# Plot each cluster
	for i in range(n_cluster):
		plt.scatter(clusters[i][:,0], clusters[i][:,1], marker=marker_symbol, edgecolor=color_list[i], facecolor='white')
	
	# Plot the centroid of each cluster
	for i in range(n_cluster): 
		plt.scatter(centroids[i][0], centroids[i][1], marker=marker_symbol, facecolor=color_list[i], s=10)
		
	plt.axis('off')	
	plt.savefig(figname)
	plt.show()
	
def with_lecture_data():
	X = []
	with open('2d-7/2d-7.txt', 'rt') as fin:
		for line in fin:
			row = map(float, line.strip().split(','))
			X.append(row)
	X = np.asarray(X, dtype='float64')
	
	clusters, cs = kmeans(X, init=init_plusplus, k=2)
	plotClusters(clusters, cs, '2d-7/2d-7_random_init_from_data.png')
	
def with_9_gaussian_toy_data():
	X = gen_2D_data(n=500)
	clusters, centroids = kmeans(X, init=init_plusplus, k=9)
	plotClusters(clusters, centroids, '9_gaussian_plusplus2.png') 
	
if __name__ == '__main__':
	with_9_gaussian_toy_data()
	
	
	