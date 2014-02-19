"""
Simple implementation of (linear) k-means algorithm with:
1. Random initial centroids
2. Initial centroids selected by kmeans++ heuristic.
"""

import random
from collections import defaultdict
import numpy as np
from utils import WeightedRandomSelector

def edist(x, y):
    """ Compute the Euclidean distance between two points p, q."""
    
    if len(x) != len(y):
        raise ValueError, "Lengths must match!"
    
    sqSum = sum(map(lambda a,b: (a-b)**2, x, y))
    #sqSum = np.sum((x-y)**2)
    return np.sqrt(sqSum)
    
def random_init_from_data(X, k, dist=None):
    """ Choose k initial *different* centroids randomly from input data X. 
        
        Params: 
            X:  n x d matrix of n samples. Type: np.ndarray  
                Each sample is d-dimensional.
            k: Number of centroids
        Returns:
            List of k centroids.
    """
    n = X.shape[0] 
    if k > n: 
        k = n
    k_indices = random.sample(xrange(n), k)
    cs = X[k_indices]
    return cs.tolist()
    
def random_init2(X, k, dist=None):
    """ Choose k initial *different* centroids randomly and not necessarily from data X.
        
        Params: 
            X:  n x d matrix of n samples. Type: np.ndarray 
                Each sample is d-dimensional.
            k:  Number of centroids
        Returns:
            List of k centroids.
    """
    
    n, d = X.shape
    if k > n:
        k = n
        
    # val_range[i] = (min_i, max_i) -> [min, max] range for i-th coordinate of X.    
    val_range = zip(np.min(X, 0), np.max(X, 0))
    
    # Select k random *different* initial centroids.
    set_c = set()
    
    while len(set_c) < k:
        set_c.add((random.random() * (val_range[i][1] - val_range[i][0]) + val_range[i][0] for i in range(d)))
    
    assert len(set_c) == k
    cs = map(list, set_c)
    return cs
    
def init_plusplus(X, k, dist=edist):
    """ Choose k initial *different* centroids randomly using the k-means++ heuristic. 
        See the paper: Sergei Vassilvitskii, and David Arthur, K-means++: The advantages of careful seeding.
        This often gives better clustering results, but it is slower than random initial version.
    """
    
    X = X.tolist()
    set_c = set()
    
    # Choose the first centroid randomly from data X
    cid = random.randrange(len(X))
    set_c.add(tuple(X[cid]))
    del X[cid]
    
    i = 0
    while len(set_c) < k and i < k*5:
        min_dists = [min(dist(c, p) for c in set_c) for p in X]
        selector = WeightedRandomSelector(min_dists)
        cid = selector()
        set_c.add(tuple(X[cid]))
        i += 1
        del X[cid]
    
    cs = map(list, set_c)
    if len(cs) < k:
        cs.extend([cs[0]]*(k-len(cs)))    
    return cs
    
def kmeans(X, init=init_plusplus, distance=edist, k=5, n_iter=1000, tol=1e-10):
    """    k-means algorithm of clustering data X into k clusters.
        
        Params:
            X:     n x d matrix of n samples. [np.ndarray type]
                Each sample is d-dimensional.
            distance: Distance metric function (Euclidean, Pearson, etc.)
            init: Specify how to select k initial centroids (random, k-means++, etc.)
            n_iter: The maximum number of iterations.
            tol: Tolerance value for convergence
        
        Returns:
            clusters: {cluster_id: [list of indices of samples in cluster]} for cluster_id = 0,1,...,k-1
            cs: list of centroids.
    """
    
    n = X.shape[0] # number of samples
    
    # Get k initial centroids
    cs = init(X, k, dist=distance)
    
    for _ in range(n_iter):    
        # First, assign each data point to a cluster specified by its nearest centroid.
        clusters = defaultdict(list)
        for xid in range(n):
            _, cid = min([ (distance(X[xid], cs[i]), i) for i in range(k) ])
            clusters[cid].append(xid)
        
        oldcs = cs[:] # save old centroids
        
        # Compute new centroid for each cluster. 
        for i in range(k):
            cPoints = clusters[i] # list of indices of samples that belong to the i-th cluster 
            cSize = len(cPoints) # size of the i-th cluster
            if cSize > 0:
                cX = X[cPoints]
                cs[i] = np.mean(cX, axis=0) # new centroid of the i-th cluster
        
        # Check convergence
        diff = 0.0
        for i in range(k):
            diff += distance(cs[i], oldcs[i])
        if diff <= tol: 
            break
            
    return clusters, cs
    
if __name__ == '__main__':
    pass
    
    