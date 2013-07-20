"""
	Simple Implementation of K-nearest neighbor algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

def edist(x, y):
	""" Compute the Euclidean distance between two samples x, y \in R^d."""
	try:
		dist = np.sqrt(np.sum((x-y)**2))
	except ValueError:
		print 'Dimensionality of samples must match!'
	else:
		return dist

def manhattan_dist(x, y):
	""" Compute the Manhattan distance between two samples x, y \in R^d."""
	try:
		dist = np.sum(np.abs(x-y))
	except ValueError:
		print 'Dimensionality of samples must match!'
	else:
		return dist
		
def knn(query, data, k=5, metric=edist):
	"""
		Find [k] nearest neighbors of [query] from data.
		Data [data] is given as an np.array of size n x d, 
		where d is the dimensionality of sample, n is the number of samples.
		[metric]: distance function. Default to Euclidean distance.
	"""
	n, d = data.shape
	if k > n: k = n
	
	# compute distances from query to all data samples.
	dist_array = np.asarray([edist(query, x) for x in data])
	
	# return indices of k nearest neighbors of query.
	return np.argsort(dist_array)[:k]

def knn_classifier(query, data, labels, k=5, metric=edist):
	"""
		KNN-based classifier.
		[labels]: array of sample labels.
		Return sample labels and k-neighbors indices.
	"""
	k_indices = knn(query, data, k, metric)
	counts = np.bincount(labels[k_indices])
	
	return np.argmax(counts), k_indices

def test_knn():
	# Generate 2d uniform data samples  
	data = np.random.rand(200, 2)
	
	# Query sample
	query = np.random.rand(1, 2)[0]
	
	k_indices = knn(query, data, metric=manhattan_dist)
	
	# Plotting the data samples and query.
	plt.plot(data[:,0], data[:,1], 'ob', query[0],query[1], 'or')
	
	# Show k neighbors
	plt.plot(data[k_indices,0], data[k_indices,1], 'o',
			markerfacecolor='None', markersize=15, markeredgewidth=1)
		 
	plt.show()
	
def test_classifier():
	# Generate 2d uniform data samples
	data = np.random.rand(200, 2)
	labels = np.array(np.random.rand(200) > 0.5, dtype=int)
	
	# Query
	query = np.random.rand(1, 2)[0]
	
	# Classifying...
	qlabel, k_indices = knn_classifier(query, data, labels)
	print 'Classified label of query: %d' % (qlabel)
	
	# Visualization. Note that: label = 1(0) -> c = red(blue)
	plt.scatter(data[:,0], data[:,1], c=labels, alpha=0.8)
	plt.scatter(query[0], query[1], c='g', s=40)
	plt.plot(data[k_indices,0], data[k_indices,1], 'o', markerfacecolor='None', markersize=15, markeredgewidth=1)
	plt.show()

if __name__ == '__main__':
	#test_knn()
	test_classifier()