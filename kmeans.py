'''
Simple k-means with random initial starting points (ver 0)
(c) Duong Nguyen nguyen@sg.cs.titech.ac.jp
'''

import math
import random
from collections import defaultdict
import matplotlib.pyplot as plt

def edist(p, q):
	"""
	Compute the Euclidean distance between two points p, q.
	"""
	if len(p) != len(q):
		raise ValueError, "lengths must match!"
	
	sqSum = sum(map(lambda x,y: (x-y)**2, p, q))
	return math.sqrt(sqSum)
	
def pearson(x, y):
	"""
	Compute the Pearson correlation between two points x, y. 
	This can be used a "distance" between x, y too.
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
		
def kmeans(points, distance=pearson, k=5, iter=1000):
	"""
	k-means algorithm of clustering data points into k clusters using distance metric.
	"""
	
	pdim = len(points[0])
	
	# Compute the [min, max] ranges for each coordinate of point. This can be used to choose initial random means.
	vrange = zip(map(min, points), map(max, points))
	
	# Select initial k random means
	means = []
	for i in range(k):
		means.append([random.random()*(vrange[i][1]-vrange[i][0]) + vrange[i][0] for j in range(pdim)])
	
	#print means
	
	clusters = None
	
	for t in range(iter):
		print 'Iteration %d' %t
		
		# First, assign each data point to a cluster specified by its nearest mean.
		tmpClusters = defaultdict(list)
		
		for p in points:
			pdist, cid = min([(distance(p, means[id]), id) for id in range(len(means))])
			tmpClusters[cid].append(p)
		
		# Stop if convergence
		if tmpClusters == clusters: break
		# Update clusters
		clusters = tmpClusters
		
		# Compute new mean for each cluster. 
		for i in range(k):
			# Get the list of points that belong to i-th cluster
			cPoints = tmpClusters[i]
			# Get the size of i-th cluster
			cSize = len(cPoints)
			
			# Compute new mean for i-th cluster by simply compute its average
			if cSize > 0:
				total = map(sum, zip(*cPoints))
				avg = map(lambda x: x/cSize, total)
				means[i] = avg # new mean of i-th cluster
		
	clusters = tmpClusters
	return clusters, means
			
def genPoints(n=100, d=2):
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
	
def clusterStat(clusters):
	for id in clusters:
		print 'Cluster %d: ' %id
		print clusters[id]
		print 'Size: %d' % len(clusters[id])
		print '-'*30

def plotClusters(clusters, means):
	nc = len(means)
	# Assume that there are 5 clusters at most, for the sake of simplicity in plotting color scheme.
	clist = ['r', 'g', 'b', 'm', 'k']
	mlist = ['*', '+', 's', 'H', 'D']
	
	# First plot mean of each cluster
	for i in range(nc): 
		#plt.scatter(means[i][0], means[i][1], s=60, c=clist[i], marker=mlist[i])
		plt.plot(means[i][0], means[i][1], clist[i]+mlist[i], markersize=10)
	
	for i in range(nc):
		plot2d(clusters[i], color=clist[i])
	plt.show()
	

if __name__ == '__main__':
	points = genPoints()
	#plot2d(points)
	clusters, means = kmeans(points, distance=edist, k=2)
	plotClusters(clusters, means)
	
	