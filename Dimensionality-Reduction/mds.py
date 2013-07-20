"""
Simple implementation of classical MDS.
See http://www.stat.cmu.edu/~ryantibs/datamining/lectures/09-dim3-marked.pdf for more details.
"""

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def square_points(size):
	nsensors = size**2
	return np.array([(i/size, i%size) for i in range(nsensors)])
	
def norm(vec):
	return np.sqrt(np.sum(vec**2))
	
def mds(D, dim=2):
	"""
	Classical multidimensional scaling algorithm.
	Given a matrix of interpoint distances D, find a set of low dimensional points
	that have a similar interpoint distances.
	"""
	(n,n) = D.shape
	A = (-0.5 * D**2)
	M = np.ones((n,n))/n
	I = np.eye(n)
	B = np.dot(np.dot(I-M, A),I-M)
	
	'''Another way to compute inner-products matrix B
	Ac = np.mat(np.mean(A, 1))
	Ar = np.mat(np.mean(A, 0))
	B = np.array(A - np.transpose(Ac) - Ar + np.mean(A))
	'''
	
	[U,S,V] = linalg.svd(B)
	Y = U * np.sqrt(S)
	return (Y[:,0:dim], S)
	
def test():
	points = square_points(10)
	distance = np.zeros((100,100))
	for (i, pointi) in enumerate(points):
		for (j, pointj) in enumerate(points):
			distance[i,j] = norm(pointi-pointj)
	
	Y, eigs = mds(distance)
	
	plt.figure()
	plt.plot(Y[:,0], Y[:,1], '.')
	plt.figure(2)
	plt.plot(points[:,0], points[:,1], '.')
	plt.show()
	
def main():
	import sys, os, getopt, pdb
	def usage():
		print sys.argv[0] + "[-h] [-d]"
		
	try:
		(options, args) = getopt.getopt(sys.argv[1:], 'dh', ['help', 'debug'])
	except getopt.GetoptError:
		usage()
		sys.exit(2)
		
	for o, a in options:
		if o in ('-h', '--help'):
			usage()
			sys.exit()
		elif o in ('-d', '--debug'):
			pdb.set_trace()
			
	test()
	
	
if __name__ == '__main__':
	main()