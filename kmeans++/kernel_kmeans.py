"""
Simple implementation of kernel k-means algorithm.

Author: Duong Nguyen
Email: nguyen@sg.cs.titech.ac.jp
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import shuffle
from sklearn.metrics.pairwise import euclidean_distances


def gaussian_kernel(x, y, c):
    """ K(x, y) = exp(-||x-y||^2 / (c^2)). """
    return np.exp(-np.dot(x-y, x-y)/(c*c))

def kernel_kmeans(X, k=2, c=0.6, n_iter=1000, tol=1e-10):
    """ Kernel k-means algorithm with Gaussian kernel K(x,y) = exp(-||x-y||^2 / (c*c))
    
        Params:
            X:  n x d matrix of n samples.
                Each sample is d-dimensional
            k: #clusters
            c: Gaussian kernel bandwidth
            iter: #iteration
            tol: tolerance value to check convergence.
            
        Returns: 
            Set of clusters. Each cluster contains indices of samples belong to that cluster.
    """
    
    n, d = X.shape
    
    # First, randomly initialize cluster partition
    X_ind = np.arange(0,n)
    
    shuffle(X_ind)
    split_ind = np.arange(n/k, (n/k)*k, n/k)
    assert len(split_ind) == k-1
    
    clusters = np.split(X_ind, split_ind)   
    assert len(clusters) == k 
    
    # Next compute the kernel matrix K
    sqD = euclidean_distances(X, X, squared=True)
    K = np.exp(-sqD/(c*c))
    
    for i in range(n_iter):
        tmp_clusters = [[] for _ in range(k)]
        
        for xi in range(n):
            min_ci = -1
            best_min = None
            for ci in range(k):
                c_size = len(clusters[ci])
                
                tmp1 = sum([ K[xi, cxi] for cxi in clusters[ci] ])
                tmp2 = sum([ K[cxi, cxj] for cxi in clusters[ci] for cxj in clusters[ci] ])
                if best_min == None:
                    best_min = -2.0 * tmp1 / c_size + 1.0 * tmp2 / (c_size**2)
                    min_ci = ci
                elif best_min > -2.0 * tmp1 / c_size + 1.0 * tmp2 / (c_size**2):
                    best_min = -2.0 * tmp1 / c_size + 1.0 * tmp2 / (c_size**2)
                    min_ci = ci
            
            # Store new assignment in temporary list        
            tmp_clusters[min_ci].append(xi)
        
        # Check if converge
        # diff is sum of distance between k old and new centroids 
        diff = 0 
        for ci in range(k):
            old_ci_size = len(clusters[ci])
            new_ci_size = len(tmp_clusters[ci])
            tmp1 = sum([ K[a,b] for a in clusters[ci] for b in clusters[ci] ])
            tmp2 = sum([ K[a,b] for a in tmp_clusters[ci] for b in tmp_clusters[ci] ])
            tmp3 = sum([ K[a,b] for a in clusters[ci] for b in tmp_clusters[ci] ])
            diff += np.sqrt(1.0 * tmp1 / (old_ci_size**2) + 1.0 * tmp2 / (new_ci_size**2) - 2.0 * tmp3 / (old_ci_size*new_ci_size))
            
        if diff <= tol: # break if converge
            break
        
        # Update new clusters
        for ci in range(k):
            clusters[ci] = np.array(tmp_clusters[ci])
        
        
    assert len(clusters) == k, "We need k clusters!"
    return clusters
        
def loadData(fname):
    X = []
    with open(fname, 'rt') as fin:
        for line in fin:
            row = map(float, line.strip().split(','))
            X.append(row)
            
    X = np.array(X, dtype='float64')
    return X

def with_dataset(fname='2d-6.txt',figname='2d-6.png'):
    X = loadData(fname)
    n_clusters = 2
    X_cluster = kernel_kmeans(X, k=n_clusters, c=0.1, n_iter=1000, tol=1e-7)
    
    cs = ['r', 'b']
    ms = ['o', 'x']
    fig = plt.figure()
    fig.clf()
    for i in range(n_clusters):
        plt.plot(X[X_cluster[i], 0], X[X_cluster[i], 1], cs[i]+ms[i])
    
    plt.axis('off')
    plt.savefig(figname)
    plt.show()
    
if __name__ == '__main__':
    with_dataset(fname='2d-7.txt', figname='2d-7_c_01.png')
    