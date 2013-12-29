'''
Created on 2013/12/29
@author: Duong Nguyen
@note: Nonnegative Matrix Factorization (NMF) with multiplicative update rules.
@version: Frobenius

V = W*H, where V, W, H >= 0. Here, ">=" means element-wise relation
V: n x m matrix
W: n x b matrix
H: b x m matrix 
Minimize_over_W,H || V - W*H || ^ 2 (Frobenius norm)

See also, http://hebb.mit.edu/people/seung/papers/nmfconverge.pdf for more details.
'''

import numpy as np
import matplotlib.pyplot as plt


def fro_distance(A, B):
    """ Compute the Frobenius distance between two matrices A, B.
        d(A, B) = ||A-B||_2 = sqrt{\sum_{i,j} (A_ij-B_ij)^2}
    """
    return np.sqrt(np.sum((A-B)**2))

def factorize(V, b, init="uniform", max_iter=100, tol=1e-8, verbose=False):
    """ Factorization with multiplicative update rules.
        @param init: specify how to initialize W, H. Default: uniformly random in [0,1)
        @param max_iter: Maximum number of iterations of updating
        @param tol: Tolerance value to check if converge   
    """
    
    assert init in ("uniform", "normal"), "Unsupported initialization!"
    n, m = V.shape
    
    if init == "uniform":
        W, H = np.random.random((n,b)), np.random.random((b,m))
    else:
        W, H = np.random.randn(n,b), np.random.randn(b,m)
    
    dists = []
    for i in xrange(max_iter):
        WH = W.dot(H)
        
        cur_dist = fro_distance(V, WH)
        dists.append(cur_dist)
        
        if verbose and i % 10 == 0:
            print "%d iteration: %f" %(i, cur_dist)
            
        if cur_dist <= tol:
            break
        
        # Update H
        nu = (W.T).dot(V)
        de = (W.T).dot(WH)
        H = H*nu/de # element-wise update
        
        # Update W
        nu = V.dot(H.T)
        de = W.dot(H).dot(H.T)
        W = W*nu/de # element-wise update
        
    return W, H, dists
    
def test(plot=False):
    """ Test Frobenius norm-based NMF."""
    
    V = np.random.random((10, 9))
    W, H, dists = factorize(V, 3, init="uniform", max_iter=200, tol=1e-8, verbose=True)
    
    if plot:
        plt.figure()
        plt.clf()
        plt.plot(dists, "b-", lw=2)
        plt.xlabel("Iteration")
        plt.ylabel("Frobenius distance as cost function")
        plt.title("Frobenius NMF Demo")
        plt.show()
        plt.savefig("fro_nmf_demo.png") # save figure at current working directory
        
    print "V =", V
    print "W * H =", W.dot(H)
    print "Frobenius distance d =", dists[-1] 
    
if __name__ == "__main__":
    test(plot=True)