import math
import random
from collections import defaultdict
import myfeeder

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
		
def kmeans(points, distance=edist, k=5, iter=100):
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
		
	clusters = None
	
	for t in range(iter):
		print 'Iteration %d' %t
		
		# First, assign each data point to a cluster specified by its nearest mean.
		tmpClusters = defaultdict(list)
		
		for pid in range(len(points)):
			_, cid = min([(distance(points[pid], means[id]), id) for id in range(len(means))])
			tmpClusters[cid].append(pid)
		
		# Stop if convergence
		if tmpClusters == clusters: break
		# Update clusters
		clusters = tmpClusters
		
		# Compute new mean for each cluster. 
		for i in range(k):
			# Get the list of points that belong to i-th cluster
			cPoints = [points[id] for id in tmpClusters[i]]
			# Get the size of i-th cluster
			cSize = len(cPoints)
			
			# Compute new mean for i-th cluster by simply compute its average
			if cSize > 0:
				total = map(sum, zip(*cPoints))
				avg = map(lambda x: x/cSize, total)
				means[i] = avg # new mean of i-th cluster
		
	clusters = tmpClusters
	return clusters, means
		
		
def clustering(fname='feedlist.txt', distance=edist, nc=2):
	datafile = 'data.txt'
	myfeeder.parseFeedList(fname, out=datafile)
	rows, cols, data = myfeeder.readData(datafile)
	
	cls, means = kmeans(data, distance=distance, k=nc)
	fclusters = {}
	for i in range(nc):
		fclusters[i] = [rows[j] for j in cls[i]]
		
	return fclusters
	
def showClusters(clusters):
	for i in clusters:
		print '-'*30
		print 'Cluster %d: %d feed(s)' %(i, len(clusters[i]))
		for feed in clusters[i]:
			print feed
		print '-'*30
		
	
if __name__ == '__main__':
	fclusters = clustering()
	showClusters(fclusters)
	