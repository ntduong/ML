"""
Naive implementation of Local Fisher Discriminant Analysis
Demo with 2D toy data

Author: Duong Nguyen @TokyoTech
Contact: nguyen@sg.cs.titech.ac.jp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy import spatial # for kdtree
from sklearn.metrics.pairwise import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D # for 3d plot

#------------------------------------------#
#    Load data from file, plotting code    # 
#------------------------------------------#
def loadData(filename, label):
    """ Load data from file.
        Assign given label to each samples.
        
        Returns:
        ----------------------
        X: n x d matrix of n samples
        Y: n x 1 label vector
        n_samples: number of samples
        dim: data dimension
    """
    X = []
    with open(filename, 'r') as fin:
        for line in fin:
            row = map(float, line.strip().split(','))
            X.append(row)
            
    n_samples = len(X)
    dim = len(X[0])
    Y = [label] * n_samples
    X = np.asarray(X, dtype='float64')
    Y = np.asarray(Y, dtype='int16')
    return X, Y, n_samples, dim

def plot_2d_xx(infile1, infile2, outfile=None, show=True):
    """ Plot 2d data. Just for testing."""
    X1, Y1, n1, d1 = loadData(infile1, 0)
    X2, Y2, n2, d2 = loadData(infile2, 1)
    d1 = X1.shape[1]
    d2 = X2.shape[1]
    assert d1 == d2, "Dimension not match!"
    assert d1 == 2, "2D!"
    
    plt.plot(X1[:,0], X1[:,1], 'rx', label='Class 1')
    plt.plot(X2[:,0], X2[:,1], 'go', label='Class 2')
    plt.legend(loc='best', prop={'size':20})
    
    if outfile: # save to file
        plt.savefig(outfile)
    if show: # show figure
        plt.show()

#--------------------------#
#    Similarity matrix     # 
#--------------------------#
def sqDist(X):
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

#-----------------#
#    LFDA code    # 
#-----------------#
def LFDA(X, Y, n_class, m, sim_type='knn'):
    """ Local Fisher Discriminant Analysis.
        n classes: 0,1,...n-1
        Params:
        ----------------------------
        X: n x d matrix of n d-dimensional samples
        Y: n x 1 class label vector for n samples
        n_class: number of classes
        sim_type: similarity matrix type,
         + knn for knn-based
         + dist for dist-based
         + local for local-based
        m: reduced dimensionality
        
        Returns:
        ------------------------------
        B_LFDA: m x d matrix (LFDA embedding matrix)
    """
    n, d = X.shape
    # Compute similarity matrix
    if sim_type == 'knn':
        W = knn_based_sim(X, kv=50)
    elif sim_type == 'dist':
        W = dist_based_sim(X, gamma=0.5)
    elif sim_type == 'local':
        W = local_scaling_based_sim(X, kv=7)
    else:
        print 'Invalid sim_type!'
        W = np.ones((n,n), dtype='float64')
        
    # nc = [n0, n1,...], where ni = # samples from class i
    nc = [len(Y[Y==i]) for i in range(n_class)]
    
    # Compute Qw matrix
    Qw = np.zeros((n,n), dtype='float64')
    for i in range(n):
        for j in range(n):
            if Y[i] == Y[j]:
                Qw[i,j] = W[i,j] / float(nc[Y[i]])
    
    # Compute Sw matrix
    Sw = np.zeros((d,d), dtype='float64')
    for i in range(n):
        for j in range(n):
            r = np.reshape(X[i]-X[j], (1,d))
            c = np.reshape(X[i]-X[j], (d,1))
            Sw += 0.5 * Qw[i,j] * np.dot(c,r)
            
    # Compute Qb matrix
    Qb = 1./n * np.ones((n,n), dtype='float64')
    for i in range(n):
        for j in range(n):
            if Y[i] == Y[j]:
                Qb[i,j] = W[i,j] * (1./n - 1./nc[Y[i]])
     
    # Compute Sb matrix
    Sb = np.zeros((d,d), dtype='float64')
    for i in range(n):
        for j in range(n):
            r = np.reshape(X[i]-X[j], (1,d))
            c = np.reshape(X[i]-X[j], (d,1))
            Sb += 0.5 * Qb[i,j] * np.dot(c,r)
            
    eigvals, V = eig(Sb, Sw)
    sorted_indices = np.argsort(eigvals)[::-1] # sorted in decreasing order of eigenvalues
    V = V[:, sorted_indices]
    return V[:, 0:m]

def testLFDA(outfile='lfda-2d-L3.png', show=True, debug=False):
    """ Test LFDA."""
    
    X1, Y1, _, d1 = loadData('2d-L3-c1.txt', 0)
    X2, Y2, _, d2 = loadData('2d-L3-c2.txt', 1)
    assert d1 == d2, "Dimensions must match!"
    
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2))
    n_class = 2
    
    lfda_dir = LFDA(X, Y, n_class, m=1, sim_type='knn')
    
    if debug: # In debug mode, print out lfda_dir vector
        print lfda_dir
    
    x_mean = np.mean(X, 0)
    
    plt.plot(X1[:,0], X1[:,1], 'rx', markersize=10, mew='2', label='Class 1')
    plt.plot(X2[:,0], X2[:,1], 'bo', markersize=10, mew='2', mfc='w', mec='b', label='Class 2')
    
    plt.plot([x_mean[0] + k * lfda_dir[0] for k in np.linspace(-9,9)], 
                [x_mean[1] + k * lfda_dir[1] for k in np.linspace(-9,9)],
                'k-', lw=4, label='LFDA')
    
    plt.legend(loc='best', prop={'size':18})
    plt.axis('equal')
    plt.savefig(outfile)
    if show:
        plt.show()
    
if __name__ == '__main__':
    testLFDA()
    