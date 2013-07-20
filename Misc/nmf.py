# NMF Factorization
# See http://hebb.mit.edu/people/seung/papers/nmfconvergence.pdf for more details.

import numpy as np

def distance(a, b):
	return np.sum((a-b)**2)
		
def factorize(V, pc=10, iter=50):
	"""
		Approximate V by W*H by minimizing distance(V, W*H).
		V: r x c matrix,
		W: r x pc matrix,
		H: pc x c matrix.
	"""
	
	r = V.shape[0]
	c = V.shape[1]
	
	# Initialize W, H matrices
	W = np.random.random((r,pc))
	H = np.random.random((pc,c))
	
	for i in range(iter):
		WH = np.dot(W, H)
		
		dist = distance(V, WH)
		
		if i % 10 == 0:
			print dist
			
		if dist == 0: break
		
		# Update H
		Hnu = np.dot(W.T, V)
		Hde = np.dot(np.dot(W.T, W), H)
		H = H*Hnu/Hde
		
		# Update W
		Wnu = np.dot(V, H.T)
		Wde = np.dot(W, np.dot(H, H.T))
		W = W*Wnu/Wde
		
	return W, H
		
if __name__ == '__main__':
	V = np.random.random((4,6))
	W, H = factorize(V, pc=3, iter=10000)
	print np.dot(W,H)
	print V
	