'''
Created on 2013/04/25
@author: duongnt
'''

import numpy as np
from scipy.linalg import eig
from scipy import spatial # for kdtree
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

def loadData(filename):
    """ Load data from file.
        Returns:
        -------------------
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
    """ Plot data and save the figure in file.
        Args:
        --------------
        data: n x d matrix (ndarray)
        figname: figure file name
    """
    
    d = data.shape[1]
    assert d <= 3, "Up to 3D data!"
    
    if d == 2: # 2D
        plt.scatter(data[:,0], data[:,1], c='r', marker='o', label='2d')
        plt.legend(loc='best')
        plt.axis('equal')
        plt.savefig(figname)
    elif d == 3: # 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data[:,0], data[:,1], data[:,2], 'ro', label='3d')
        ax.legend(loc='best')
        ax.view_init(30,50) # view angle
        plt.savefig(figname)
    else:
        pass

def sqDist(X):
    """ Compute the pair-wise distance matrix between row vectors of X (samples).    
        Args:
        ------------------
            X: n x d matrix of n samples
        Returns:
        --------------------
            sqD of size n x n, where sqD_{i,j} = || xi - xj ||^2
    """
    
    n = X.shape[0] # number of samples
    tX = X.T # tX has size d x n
    sq = np.sum(tX**2, axis=0)
    A = np.tile(sq, (n,1))
    B = A.T
    C = np.dot(X, tX)
    sqD = A + B - 2*C
    return sqD

def sqDist2(X):
    """ Compute the pair-wise distance matrix between row vectors of X,
        using sklearn.metrics.pairwise.euclidean_distances.
        Args:
        -----------
            X: n x d
        Returns:
        ------------
            sqD of size n x n, where sqD_{i,j} = || xi - xj ||^2
    """
    return euclidean_distances(X, X, squared=True)
    
def dist_based_sim(X, gamma=0.5):
    """ Construct distance-based similarity matrix W. 
        W_ij = exp(-|| xi - xj ||^2 / (gamma^2)).
        By using square distance matrix sqD above, we can compute W easily.
        Args:
        --------------------------------
            X: n x d matrix of n samples
            gamma: tuning parameter > 0
        Returns:
        ----------------------------------------
            Similarity matrix W of size n x n
    """ 
    sqD = sqDist(X)
    W = np.exp(-sqD/(gamma**2))
    return W

def local_scaling_based_sim(X, kv=7):
    """ Compute distance-based similarity matrix,
        using local scaling heuristic.
        See L. Zelnik-Manor & P. Perona, Self-tuning spectral clustering,
        Advances in NIPS 17, 1601-1608, MIT Press, 2005 for heuristically choosing k value.
    """
    
    tree = spatial.KDTree(X) # use KDTree for knn fast queries. 
    n = X.shape[0]
    gammas = []
    for i in range(n):
        gammas.append(tree.query(X[i], k=kv)[0][-1]) # compute the distance btw X[i] and its k-th nearest neighbor
    gammas = np.asarray(gammas, dtype='float64')
    localMat = np.dot(gammas.reshape(n,1), gammas.reshape(1,n)) # localMat[i,j] = gamma_i x gamma_j
    sqD = sqDist(X)
    W = np.exp(-sqD/localMat)
    return W

def knn_based_sim(X, kv=7):
    """ Compute k-nearest-neighbor-based similarity matrix W
        Args:
        ---------
            X: n x d matrix of n samples
            kv: k value in knn.
        Returns:
            W: n x n matrix, where
            W_{i,j} = 1 if x_i is knn of x_j or x_j is knn of x_i
            W_{i,j} = 0, otherwise.
    """
    
    tree = spatial.KDTree(X)
    n = X.shape[0]
    knn_idx = []
    for i in range(n):
        knn_idx.append(tree.query(X[i], k=kv)[1])
    
    W = np.zeros((n,n), dtype='int')
    for i in range(n):
        for j in range(i,n):
            if (i in knn_idx[j]) or (j in knn_idx[i]):
                W[i,j] = W[j,i] = 1
    
    return W

def lpp(X, W):
    """ Locality Preserving Projection (LPP).
        
        Args:
            X: data matrix of size n x d (n samples, dimensionality d)
            W: similarity(affinity) matrix of size n x n (pair-wise similarities matrix)
        Returns:
            B = [y1|y2|...|ym] of size d x m, where:
        
        y1(e1), y2(e2),...ym(em) are solutions (eigenvector,eigenvalue) 
        of a generalized eigenvalue problem: X L tX y = e X D tX y
        and e1 <= e2 <= .... <= em (the m smallest eigenvalues).
    """
    D = np.diag(np.sum(W, axis=1))
    L = D - W # construct graph-Laplacian matrix
    
    def matprod(*args):
        return reduce(np.dot, args)
        
    A = matprod(X.T, L, X)
    B = matprod(X.T, D, X)
    evals, V = eig(A,B) # w are sorted in INCREASING order. y_i = V[:,i] = i-th column of V
    
    return evals, V 
    
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
    
    
def main(X, figname, sim_type='dist_based'):
    """ Main program: Locality Preserving Projection Algorithm
        Args:
        -----------
            X: n x d matrix of n samples.
            figname: filename to save a plot figure.
            sim_type: similarity matrix type,
                    'dist_based' for distance based similarity
                    'local_scaling' for local scaling based similarity
                    'knn' for knn based similarity.    
    """
    
    if sim_type == 'dist_based':
        W = dist_based_sim(X, gamma=0.5)
    elif sim_type == 'local_scaling':
        W = local_scaling_based_sim(X, kv=7)
    elif sim_type == 'knn':
        W = knn_based_sim(X, kv=50)
    else:
        raise ValueError, "Invalid similarity type!" 
    
    _, V = lpp(X, W)
    
    # tr_X size: n x m. 
    # In practice, we choose m << d (the dimension of origin samples) 
    # to perform dimensionality reduction.
    # tr_X = np.dot(X, V) 
    
    xmean = np.mean(X, axis=0)
    
    def plot_lpp_dir(direction):
        """ Plot LPP direction. """
        plt.plot([xmean[0]+k*direction[0] for k in np.linspace(-0.8,0.8)], 
                [xmean[1]+k*direction[1] for k in np.linspace(-0.8,0.8)],
                'b-', lw=2, label='1st LPP direction')
    
    first_dir = V[:,0]
    fig = plt.figure()
    fig.clf()
    plt.scatter(X[:,0], X[:,1], c='r', marker='o', label='original data')
    plot_lpp_dir(first_dir)
    plt.legend(loc='best')
    plt.axis('equal')
    plt.savefig(figname)
    
def lpp_for_digit_dataset():
    from sklearn.datasets import load_digits
    X = load_digits().data
    
    W = local_scaling_based_sim(X, kv=7)
    _, V = lpp(X, W)
    tr_digits = lpp_transform(X, V, ncomp=2)
    plotData(tr_digits, figname='digits2d.png')
    
    
if __name__ == '__main__':
    #X = loadData('2d-1.txt')
    #plotData(X, 'original-2d-1.png')
    #main(X, 'dist_based_lpp_2d_1.png', sim_type='dist_based')
    lpp_for_digit_dataset()

