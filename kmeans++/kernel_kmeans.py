"""
Simple implementation of kernel k-means algorithm.

Author: Duong Nguyen
Email: nguyen@sg.cs.titech.ac.jp
"""

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from numpy.random import shuffle
from sklearn.metrics.pairwise import euclidean_distances

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
    
    #np.random.seed(1)
    
    shuffle(X_ind)
    split_ind = np.arange(n/k, (n/k)*k, n/k)
    assert len(split_ind) == k-1
    
    clusters = np.split(X_ind, split_ind)   
    assert len(clusters) == k 
    
    # Next compute the kernel matrix K
    sqD = euclidean_distances(X, X, squared=True)
    K = np.exp(-sqD/(c*c))
    
    def obj_function(aClusters):
        """ Compute the objective function value given a cluster configuration.
            Used in convergence checking.
            
            Params:
                aClusters: [[cluster_1], [cluster_2],..., [cluster_k]]
                
            Returns:
                The value of objective function defined as:
                sum_{j=1}^k sum_{x \in C_j} || psi(x) - mu_j ||^2
        """
        
        obj_val = float(0)
        for ci in range(k):
            ci_size = len(aClusters[ci])
            temp1 = sum([K[a,a] for a in aClusters[ci]])
            temp2 = sum([K[a,b] for a in aClusters[ci] for b in aClusters[ci]]) * 1.0 / ci_size
            obj_val += temp1 - temp2
            
        return obj_val
                
    
    obj_val_list = []
    
    for i in range(n_iter):
        obj_val_list.append(obj_function(clusters))
        
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
        
        '''
        # One way to check if converge or not
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
        '''
        
        '''
        # Different way to check convergence: If the value of objective function is not unchanged
        # or changed too little under some tolerance, then stop.
        old_obj_val = obj_function(clusters)
        new_obj_val = obj_function(tmp_clusters)
        
        obj_val_list.append(old_obj_val) # to see if the value of objective function decrease at each iteration
        
        if np.abs(new_obj_val-old_obj_val) <= tol: # convergence
            break
        '''
        
        # Simple convergence checking
        is_converge = True
        for ci in range(k):
            old_ci_size = len(clusters[ci])
            new_ci_size = len(tmp_clusters[ci])
            if old_ci_size != new_ci_size:
                is_converge = False
                break
            for xi in clusters[ci]:
                if xi not in tmp_clusters[ci]:
                    is_converge = False
                    break
            
        if is_converge:
            break
        
        # Update new clusters
        for ci in range(k):
            clusters[ci] = np.array(tmp_clusters[ci])
        
        
    assert len(clusters) == k, "We need k clusters!"
    return clusters, obj_val_list
        
def loadData(fname):
    X = []
    with open(fname, 'rt') as fin:
        for line in fin:
            row = map(float, line.strip().split(','))
            X.append(row)
            
    X = np.array(X, dtype='float64')
    return X

def with_dataset(c, dat_fname, fig_name, obj_fname):
    X = loadData(dat_fname)
    n_clusters = 2
    n_iter = 100
    X_cluster, obj_val_list = kernel_kmeans(X, k=n_clusters, c=c, n_iter=n_iter, tol=1e-7)
    
    cs = ['r', 'b']
    ms = ['o', 'x']
    fig = plt.figure(1)
    fig.clf()
    for i in range(n_clusters):
        plt.plot(X[X_cluster[i], 0], X[X_cluster[i], 1], cs[i]+ms[i])
    
    plt.axis('off')
    plt.savefig(fig_name)
    
    fig = plt.figure(2)
    fig.clf()
    
    plt.plot(range(1, len(obj_val_list)+1), obj_val_list, 'b-', lw=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Objective function')
    plt.savefig(obj_fname)
    #plt.show()
    
if __name__ == '__main__':
    with_dataset(0.6, '2d-6/2d-6.txt', '2d-6/2d-6_c_06.png', '2d-6/2d-6_obj_c_06.png')
    #with_dataset(0.3, '2d-7/2d-7.txt', '2d-7/2d-7_c_03.png', '2d-7/2d-7_obj_c_03.png')