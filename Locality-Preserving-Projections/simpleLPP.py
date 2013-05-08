'''
Created on 2013/04/25
@author: duongnt
'''

import numpy as np
from scipy.linalg import eig
from scipy import spatial # for kdtree
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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
        plt.scatter(data[:,0], data[:,1], c='r', marker='x', label='2-dim')
        plt.legend(loc='best', prop={'size':20})
        plt.axis('equal')
        plt.savefig(figname)
    elif d == 3: # 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(data[:,0], data[:,1], data[:,2], 'rx', label='3-dim')
        ax.legend(loc='best', prop={'size':20})
        ax.view_init(50,80) # view angle
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
    sqD = sqDist2(X)
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
    sqD = sqDist2(X)
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
    
    #d = B.shape[0]
    #Id = np.eye(d)
    
    evals, V = eig(A, B) # w are sorted in INCREASING order. y_i = V[:,i] = i-th column of V
    
    # Need to sort in an increasing order
    sorted_indices = np.argsort(evals)
    V = V[:,sorted_indices]
    
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
    
    
    xmean = np.mean(X, axis=0)
    
    pca = PCA(n_components=2, whiten=True).fit(X)
    pmean = pca.mean_
    pcs = pca.components_
    
    def plot_pca_dir(pc):
        plt.plot([k*pc[0]+pmean[0] for k in np.linspace(-0.5,0.5)], 
                 [k*pc[1]+pmean[1] for k in np.linspace(-0.5,0.5)], 
                 'm-', lw=4, label='PCA')
    
    def plot_lpp_dir(direction):
        """ Plot LPP direction. """
        plt.plot([xmean[0]+k*direction[0] for k in np.linspace(-0.8,0.8)], 
                [xmean[1]+k*direction[1] for k in np.linspace(-0.8,0.8)],
                'g-', lw=4, label='LPP')
    
    first_dir = V[:,0]
    fig = plt.figure()
    fig.clf()
    #plt.scatter(X[:,0], X[:,1], c='r', marker='x', label='original data')
    plt.scatter(X[:,0], X[:,1], c='r', marker='x')
    
    plot_pca_dir(pcs[0])
    plot_lpp_dir(first_dir)
    
    plt.legend(loc='best', prop={'size':20})
    plt.axis('equal')
    plt.savefig(figname)
    
def lpp_for_4d_data(filename='4d-x.txt'):
    X = []
    with open(filename, 'r') as fin:
        for line in fin:
            X.append(map(float, line.strip().split(',')))
    X = np.asarray(X, dtype='float64')
    
    #W = knn_based_sim(X, kv=2)
    #W = dist_based_sim(X, gamma=0.5)
    W = local_scaling_based_sim(X, kv=7)
    _, V = lpp(X, W)
    trX = lpp_transform(X, V, ncomp=2)
    plotData(trX, figname='local7-lpp-4dto2d.eps')

def lpp4Iris():
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = iris.data
    Y = iris.target
    
    cs = ['b', 'g', 'r']
    ms = ['+', 'o', 'x']
    lb = ['setosa', 'versicolor', 'virginica']
    
    k = 5
    ncomp = 2
    
    #W = local_scaling_based_sim(X, kv=k)
    W = knn_based_sim(X, kv=k)
    _, V = lpp(X, W)
    trX = lpp_transform(X, V, ncomp=ncomp)
    
    fig = plt.figure()
    fig.clf()
    if ncomp == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(3):
            ax.plot(trX[Y==i,0], trX[Y==i,1], trX[Y==1,2], cs[i]+ms[i], label=lb[i])
    
        ax.view_init(50, 70)
        plt.legend(loc='best', prop={'size':20})
        #plt.savefig('3d/local-%dnn-iris-3d.eps' % k)
        plt.savefig('3d/%dnn-iris-3d.png' % k)
    elif ncomp == 2:
    
        for i in range(3):
            plt.plot(trX[Y==i,0], trX[Y==i,1], cs[i]+ms[i], label=lb[i])
        plt.legend(loc='best', prop={'size':20})
        #plt.savefig('2d/local-%dnn-iris-2d.eps' % k)
        plt.savefig('2d/%dnn-iris-2d.png' % k)
    else:
        pass
    
def lpp4Digits():
    from sklearn.datasets import load_digits
    digit = load_digits(3)
    X = digit.data
    Y = digit.target
    lb = ['0', '1', '2']
    cs = ['b', 'g', 'r']
    ms = ['+', 'o', 'x']
    ncomp = 2
    
    #W = knn_based_sim(X, kv=7)
    W = local_scaling_based_sim(X, kv=20)
    _, V = lpp(X, W)
    tr_data = lpp_transform(X, V, ncomp=ncomp)

    fig = plt.figure()
    fig.clf()
    if ncomp == 2:
        for i in range(3):
            plt.plot(tr_data[Y==i,0], tr_data[Y==i,1], cs[i]+ms[i], label=lb[i])
        plt.legend(loc='best', prop={'size':20})
        plt.savefig('lpp-digit-2d.png')
    elif ncomp == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(3):
            ax.plot(tr_data[Y==i,0], tr_data[Y==i,1], tr_data[Y==i,2], cs[i]+ms[i], label=lb[i])
        ax.legend(loc='best', prop={'size':20})
        ax.view_init(50, 70)
        plt.savefig('lpp-digit-3d.png')

         
if __name__ == '__main__':
    #X = loadData('2d-2.txt')
    #plotData(X, 'original-2d-1.png')
    #main(X, 'knn_based_lpp_2d_1.eps', sim_type='knn')
    #main(X, '2d_2.eps', sim_type='knn')
    #lpp_for_4d_data()
    lpp4Iris()
    
