'''
Created on 2013/12/29
@author: Duong Nguyen
@note: Nonnegative Matrix Factorization (NMF) with multiplicative update rules.
@version: Kullback-Leibler

V = W*H, where V, W, H >= 0. Here, ">=" means element-wise relation
V: n x m matrix
W: n x b matrix
H: b x m matrix 
Minimize_over_{W,H} KL-divergence(V, W*H)

See also, http://hebb.mit.edu/people/seung/papers/nmfconverge.pdf for more details.
'''

import numpy as np
import matplotlib.pyplot as plt


def kl_distance(A, B):
    """ Compute the KL divergence between two matrices A, B.
        KL(A, B) = \sum_{i,j} (A_ij * log(A_ij / B_ij) - A_ij + B_ij)
    """
    return np.sum(A * np.log(A/B) - A + B)
    
def factorize(V, b, init="uniform", max_iter=100, tol=1e-8, verbose=False):
    """ Factorization with multiplicative update rules.
        @param init: specify how to initialize W, H. Default: uniformly random in [0,1)
        @param max_iter: Maximum number of iterations of updating
        @param tol: Tolerance value to check if converge   
        
        @return: n x b matrix W, and b x m matrix H such that KL(V, W*H) is minimized.
                The values of KL-divergence at each iteration step are also returned as dists.
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
        
        cur_dist = kl_distance(V, WH)
        dists.append(cur_dist)
        
        if verbose and i % 10 == 0:
            print "%d iteration: %f" %(i, cur_dist)
            
        if cur_dist <= tol:
            break
        
        # Update H
        nu = (W.T).dot(V/(WH))
        de = np.tile(np.sum(W,0), (m,1)).T
        H = H*nu/de # element-wise update
        
        # Update W
        nu = (V/(W.dot(H))).dot(H.T)
        de = np.tile(np.sum(H,1), (n,1))
        W = W*nu/de # element-wise update
        
    return W, H, dists
    
def test(plot=False):
    """ Test KL-based NMF."""
    
    V = np.random.random((10, 9))
    W, H, dists = factorize(V, 3, init="uniform", max_iter=300, tol=1e-8, verbose=True)
    
    if plot:
        plt.figure()
        plt.clf()
        plt.plot(dists, "b-", lw=2)
        plt.xlabel("Iteration")
        plt.ylabel("KL divergence as cost function")
        plt.title("KL-based NMF Demo")
        plt.show()
        plt.savefig("kl_nmf_demo.png") # save figure at current working directory
        
    print "V =", V
    print "W * H =", W.dot(H)
    print "KL divergence =", dists[-1] 
    
if __name__ == "__main__":
    test(plot=True)