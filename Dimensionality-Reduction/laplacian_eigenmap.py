'''
Created on 2013/05/23
@author: duong
@todo: 
+ Try sparse eigen solver
+ Compare with LLE
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for 3D plot
from scipy.linalg import eig
from scipy.sparse.linalg import eigs # sparse eigen solver
from scipy import spatial # for kdtree


def loadData(fname='s-curve.txt'):
    X = []
    with open(fname) as fin:
        for line in fin:
            row = map(float, line.strip().split(','))
            X.append(row)
            
    X = np.asarray(X, dtype='float64')
    
    def plot(fname='s-curve.png', show=False):
        fig = plt.figure()
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(X[:,0], X[:,1], X[:,2], 'bo', label='s-curve')
        ax.legend(loc='best', prop={'size':15})
        ax.view_init(30,50)
        plt.savefig(fname)
        if show:
            plt.show()
        
    #plot()
    
    return X

def lap_eigenmap(X, m=2, type='knn', k=7, gamma=3):
    """ Laplacian eigenmap.
        
        Solve generalized eigenvalue problem: Ly = aDy.
        y_n, ..., y_1 are eigenvectors corresponding to a_n <= ... <= a_1
        Note that: a_n = 0 and y_n ~ 1_n
        Eigenmap: G = [y_n-m,...y_n-1]
        
        Params:
            X: n x d matrix of n samples
            m: #eigenvectors to take
            type: similarity matrix type,
                    'knn' for knn based 
                    'local' for local scaling based
                    'dist' for distance based
            k: #nearest neigbors parameter, ignored when type='dist'
            gamma: bandwidth parameter, ignored when type='knn' or 'local'
        
        Returns:
            G: n x m matrix
    """
    
    n, _ = X.shape
    
    def knn_based_sim(k):
        tree = spatial.KDTree(X)
        knn_ind = []
        for i in range(n):
            knn_ind.append(np.asarray(tree.query(X[i], k=k)[1]))
        
        W = np.zeros((n,n), dtype='float64')
        for i in range(n):
            for j in range(n):
                if (i in knn_ind[j]) or (j in knn_ind[i]):
                    W[i][j] = W[j][i] = float(1)
                    
        return W
    
    def dist_based_sim(gamma):
        from sklearn.metrics.pairwise import euclidean_distances
        
        sqDist = euclidean_distances(X, X, squared=True)
        W = np.exp(-sqDist/(gamma**2))
        
        return W
    
    def local_scaling_based_sim(k):
        from sklearn.metrics.pairwise import euclidean_distances
        
        tree = spatial.KDTree(X) # use KDTree for knn fast queries. 
        gammas = []
        for i in range(n):
            gammas.append(np.asarray(tree.query(X[i], k=k)[0])[-1]) 
        gammas = np.asarray(gammas, dtype='float64')
        localMat = np.dot(gammas.reshape(n,1), gammas.reshape(1,n)) # localMat[i,j] = gamma_i x gamma_j
        sqDist = euclidean_distances(X, X, squared=True)
        W = np.exp(-sqDist/localMat)
        
        return W

    if type == 'knn':
        W = knn_based_sim(k)
    elif type == 'local':
        W = local_scaling_based_sim(k)
    elif type == 'dist':
        W = dist_based_sim(gamma)
    else:
        raise ValueError, "Invalid type!"
    
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    # eigvals, eigvecs = eig(L, D) --> terribly slow
    eigvals, eigvecs = eigs(L, M=D, which='SM')

    sorted_indices = np.argsort(eigvals)    
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    return eigvecs[:, m:0:-1]

def main(fname='unfold2.png', show=True):
    X = loadData()
    G = lap_eigenmap(X, m=2, type='knn', k=10, gamma=3)
    fig = plt.figure()
    fig.clf()
    
    plt.plot(G[:,0], G[:,1], 'ro', label='unfold')
    plt.legend(loc='best', prop={'size':15})
    plt.savefig(fname)
    if show:
        plt.show()
    
if __name__ == '__main__':
    main()
            

