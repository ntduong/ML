"""
Simple implementation of classical MDS.
See http://www.stat.cmu.edu/~ryantibs/datamining/lectures/09-dim3-marked.pdf for more details.
"""

import numpy as np
import numpy.linalg as linalg

def mds(D, dim=2):
	""" Classical multidimensional scaling algorithm.
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

def main():
    """ @Todo: Adding simple test code for mds."""
	pass
    
if __name__ == '__main__':
	main()