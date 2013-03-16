'''
Simple k-means with:
+ random initial starting points
+ kmeans++ heuristic
(c) Duong Nguyen
'''

import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt

def edist(p, q):
	'''
	Compute the Euclidean distance between two points p, q.
	'''
	if len(p) != len(q):
		raise ValueError, "lengths must match!"
	
	sqSum = sum(map(lambda x,y: (x-y)**2, p, q))
	return math.sqrt(sqSum)
	
def pearson(x, y):
	'''
	Compute the Pearson correlation between two points x, y. 
	This can be used a "distance" between x, y too.
	'''
	
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
		
def random_init1(points, k, dist=None):
	'''
	Choose k initial *different* centroids randomly from input data points.
	'''
	# Select k random points as initial centroids
	set_c = set(tuple(random.choice(points)) for _ in range(k))
		
	# Try to select k *different* centroids
	i = 0
	while len(set_c) < k and i < n and i < 4*k:
		set_c.add(tuple(points[i]))
		i += 1
		
	cs = map(list, set_c)
	
	# If we can't choose k *different* centroids from data points 
	if len(cs) < k:
		#c.extend( [cs[0]] * (k-len(cs)) )
		for i in range(k-len(c)):
			cs.append(random.choice(points))
			
	return cs
	
def random_init2(points, k, dist=None):
	'''
	Choose k initial *different* centroids randomly and not necessarily from data points.
	'''
	# val_range[i] = (min_i, max_i) -> [min, max] range for i-th coordinate of points.
	val_range = zip(map(min, zip(*points)), map(max, zip(*points)))
	
	pdim = len(points[0])
	# Select k random *different* initial centroids.
	set_c = set()
	
	while len(set_c) < k:
		set_c.add((random.random() * (val_range[i][1] - val_range[i][0]) + val_range[i][0] for j in range(pdim)))
	
	assert len(set_c) == k
	cs = map(list, set_c)
	
	return cs
	
def init_plusplus(points, k, dist=edist):
	'''
	Choose k initial *different* centroids randomly using the
    k-means++ heuristic by David Arthur and Sergei Vassilvitskii.
    This often gives better clustering results, but it is slower than the
    basic choice of starting points.
	'''
	set_c = set()
	
	from utils import WeightedRandomSelector
	# Choose the first centroid randomly from data points
	id = random.randrange(len(points))
	set_c.add(tuple(points[id]))
	del points[id]
	
	i = 0
	while len(set_c) < k and i < k*5:
		min_dists = [min(dist(c, p) for c in set_c) for p in points]
		selector = WeightedRandomSelector(min_dists)
		id = selector()
		set_c.add(tuple(points[id]))
		i += 1
		del points[id]
	
	cs = map(list, set_c)
	if len(cs) < k:
		cs.extend([cs[0]]*(k-len(cs)))	
	
	return cs
	
def kmeans(points, init=init_plusplus, dist=edist, k=5, iter=1000):
	'''
	k-means algorithm of clustering data points into k clusters.
	dist: Distance metric
	init: specify how to select k initial centroids.
	iter: maximum iteration of the algorithm.
	'''
	
	pdim = len(points[0])
	# Get k initial centroids
	cs = init(points, k)
	clusters = None
	
	for t in range(iter):
		print 'Iteration %d:' %t
		# First, assign each data point to a cluster specified by its nearest centroids.
		tmpClusters = defaultdict(list)
		
		for p in points:
			_, cid = min([(dist(p, cs[id]), id) for id in range(len(cs))])
			tmpClusters[cid].append(p)
		
		# Stop if convergence
		if tmpClusters == clusters: break
		# Update clusters
		clusters = tmpClusters
		
		# Compute new centroid for each cluster. 
		for i in range(k):
			# Get the list of points that belong to i-th cluster
			cPoints = tmpClusters[i]
			# Get the size of i-th cluster
			cSize = len(cPoints)
			
			# New centroid for i-th cluster: simply computing the average of the cluster's points 
			if cSize > 0:
				total = map(sum, zip(*cPoints))
				avg = map(lambda x: x/cSize, total)
				cs[i] = avg # new centroid of i-th cluster
				
	clusters = tmpClusters
	return clusters, cs
			
def genData(n=100, d=2):
	'''
	Generate n data points for testing.
	d: dimensionality of data point.
	For plotting purpose, use d = 2.
	'''
	points = []
	for i in range(n/2):
		points.append([random.random()*100+50 for j in range(d)])
	
	for i in range(n/2):
		points.append([random.random()*100-50 for j in range(d)])
	
	# randomly shuffle data points
	random.shuffle(points)
	
	return points

def plot2d(points, color='b', marker='o'):
	temp = zip(*points)
	assert len(temp[0]) == len(temp[1]), "lengths mismatched!"
	plt.scatter(temp[0], temp[1], c=color, marker=marker)
	#plt.show()

def plotClusters(clusters, cs):
	'''
	Plotting clusters for
	'''
	nc = len(cs)
	# Assume that there are 5 clusters at most, for the sake of simplicity in plotting color scheme.
	clist = ['r', 'g', 'b', 'm', 'k']
	mlist = ['*', '+', 's', 'H', 'D']
	
	# First plot mean of each cluster
	for i in range(nc): 
		#plt.scatter(means[i][0], means[i][1], s=60, c=clist[i], marker=mlist[i])
		plt.plot(cs[i][0], cs[i][1], clist[i]+mlist[i], markersize=10)
	
	for i in range(nc):
		plot2d(clusters[i], color=clist[i])
	plt.show()
	
def clusterInfo(clusters):
	'''
	Show clusters info.
	'''
	for id in clusters:
		print 'Cluster %d: ' %id
		print clusters[id]
		print 'Size: %d' % len(clusters[id])
		print '-'*30

if __name__ == '__main__':
	points = genData()
	#plot2d(points)
	# Compute clusters and their centroids
	clusters, cs = kmeans(points, k=4)
	# Plotting...
	plotClusters(clusters, cs)
	
	