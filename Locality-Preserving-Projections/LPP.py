# Locality Preserving Projection (LPP)
# Author: Duong Nguyen
# Email: nguyen@sg.cs.titech.ac.jp

import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

def loadData(filename):
	""" Load data from file.
		Returns:
			Data matrix X of size n x d
			n: the number of samples
			d: the dimensionality of data sample.
	"""
	
	X = []
	with open(filename, 'rt') as fin:
		for line in fin:
			row = map(float, line.strip().split(','))
			X.append(row)
			
	X = np.asarray(X, dtype='float64')
	return X
	
def plotData(data, figname):
	n, d = data.shape
	assert d <= 3, "Up to 3D data!"
	
	if d == 2:
		plt.scatter(data[:,0], data[:,1], c='r', marker='o', label='origin 2d data')
		plt.legend(loc='best')
		plt.savefig(figname)
	elif d == 3:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(data[:,0], data[:,1], data[:,2], 'ro', label='3d')
		ax.legend(loc='best')
		ax.view_init(30,50)
		plt.savefig(figname)
	else:
		pass

def compute_sqdist_mat(X):
	""" Compute the pair-wise distance matrix between row vectors of X.
		
		Args:
			X of size n x d
		Returns:
			sqD of size n x n, where D_{i,j} = || xi - xj ||^2
	"""
	n, d = X.shape
	tX = X.T # tX has size d x n
	sq = np.sum(tX**2, axis=0)
	A = np.tile(sq, (n,1))
	B = A.T
	C = np.dot(X, tX)
	sqD = A + B - 2*C
	return sqD

def check(X):
	""" Brute-force to check compute_sqdist_mat."""
	def sqdist(x, y):
		return np.sum((x-y)**2)

	n, d = X.shape
	sqD = []
	for i in range(n):
		row = [sqdist(X[i], X[j]) for j in range(n)]
		sqD.append(row)
	
	sqD = np.asarray(sqD, dtype='float64')
	return sqD
	
def construct_sim_mat(sqD, gamma=0.5):
	""" Construct distance-based similarity matrix W. 
		
		W_ij = exp(-|| xi - xj ||^2 / (gamma^2)).
		By using square distance matrix sqD above, we can compute W easily.
		Args:
			gamma: tuning parameter > 0.
		Returns:
			Similarity matrix W of size n x n, where n is the number of samples.
	""" 
	W = np.exp(-sqD/(gamma**2))
	return W

def lpp(X, W):
	""" Locality Preserving Projection (LPP).
		
		Args:
			X: data matrix of size n x d (n samples, dimensionality d)
			W: similarity(affinity) matrix of size n x n (pair-wise similarity between two samples)
		Returns:
			B = [y1|y2|...|ym] of size d x m, where:
		
		y1(e1), y2(e2),...ym(em) are solutions (eigenvector,eigenvalue) 
		of a generalized eigenvalue problem: X L tX y = e X D tX y
		and e1 <= e2 <= .... <= em (the m smallest eigenvalues).
	"""
	D = np.diag(np.sum(W, axis=1))
	L = D - W # construct graph-laplacian matrix
	
	def matprod(*args):
		return reduce(np.dot, args)
		
	A = matprod(X.T, L, X)
	B = matprod(X.T, D, X)
	w, V = eig(A,B) # w are sorted in increasing order. Then, yi = v[:,i] = i-th column of v
	
	return w, V
	
def lpp_transform(X, V, ncomp=2):
	""" 
		Args:
		--------------
		X: n x d. Data matrix
		V: d x m. Each column of V is a LPP direction.
		ncomp (<= m <= d): The dimension of transformed data
		
		Returns:
		--------------
		tr_X: n x ncomp
	"""
	
	_, m = V.shape
	if ncomp > m:
		ncomp = m
		
	tr_X = np.dot(X, V[:,0:ncomp])
	return tr_X
	
	
def main(X, figname):
	sqD = compute_sqdist_mat(X)
	W = construct_sim_mat(sqD)
	w, V = lpp(X, W)
	# tr_X size: n x m. 
	# In practice, we choose m << d (the dimension of origin samples) 
	# to perform dimensionality reduction.
	tr_X = np.dot(X, V) 
	
	xmean = np.mean(X, axis=0)
	def plot_lpp_dir(dir):
		plt.plot([xmean[0]+k*dir[0] for k in np.linspace(-0.7,0.7)], 
				[xmean[1]+k*dir[1] for k in np.linspace(-0.7,0.7)],
				'b-', lw=2, label='1st LPP direction')
	
	first_dir = V[:,0]
	fig = plt.figure()
	fig.clf()
	plt.scatter(X[:,0], X[:,1], c='r', marker='o', label='origin data')
	plot_lpp_dir(first_dir)
	plt.legend(loc='best')
	plt.savefig(figname)
	
def wineDataAnalysis():
	""" Apply LPP to Wine Quality Dataset (UCI Machine Learning Data Repository).
	"""
	import csvloader
	data, _ = csvloader.main()
	cdata = np.vstack((data[0], data[1]))
	
	#print cdata.shape
	#print data[0].shape
	cdata = cdata[:5000]
	
	sqD = compute_sqdist_mat(cdata)
	
	W = construct_sim_mat(sqD)
	w, V = lpp(cdata, W)
	
	
	def lpp2d():
		tr_2d_data = lpp_transform(cdata, V, ncomp=2)
		assert tr_2d_data.shape[1] == 2, "2D"
	
		nred = data[0].shape[0]
		fig = plt.figure()
		fig.clf()
		plt.scatter(tr_2d_data[0:nred][:,0], tr_2d_data[0:nred][:,1], c='r', marker='o', label='red wine')
		plt.scatter(tr_2d_data[nred:][:,0], tr_2d_data[nred:][:,1], c='b', marker='o', label='white wine')
		plt.legend(loc='best')
		plt.savefig('wine-lpp-2d.png')
	
	def lpp3d():
		tr_3d_data = lpp_transform(cdata, V, ncomp=3)
		assert tr_3d_data.shape[1] == 3, "3D"
		
		nred = data[0].shape[0]
		fig = plt.figure()
		fig.clf()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(tr_3d_data[0:nred][:,0], tr_3d_data[0:nred][:,1], tr_3d_data[0:nred][:,2],
			c='r', marker='o', label='red wine')
		ax.plot(tr_3d_data[nred:][:,0], tr_3d_data[nred:][:,1], tr_3d_data[nred:][:,2],
			c='b', marker='o', label='white wine')
	
		ax.legend(loc='best')
		ax.view_init(30,50)
		plt.savefig('wine-lpp-3d.png')
	
	lpp2d()
	lpp3d()
	
if __name__ == '__main__':
	#X = loadData('2d-2.txt')
	#plotData(X, 'origin-2d-2.png')
	#main(X, 'lpp-2d-2.png')
	wineDataAnalysis()

