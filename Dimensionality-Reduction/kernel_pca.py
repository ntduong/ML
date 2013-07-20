'''
Created on May 18, 2013
@author: Duong Nguyen
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_circles, make_moons, load_iris, load_digits

def myKPCA(X, Y, kernel_type='gauss', c=3, deg=2, ncomp=2, dataset='wine', show=False):
    """ Kernel PCA.
        Params:
        --------------
        X: n x d samples matrix
        Y: n x 1 vector of class labels
        kernel_type: Kernel type (Polynomial, RBF, etc.)
        c: Gaussian width (Ignored if kernel_type is polynomial)
        deg: Degree (Ignored if kernel_type is Gaussian kernel.)
        ncomp: #principal components
        dataset: Name of dataset
        show: True to show the embedding.
    """
    
    def poly_kernel(deg):
        """ Polynomial kernel.
            Params:
            ------------
            deg: Degree of polynomial.
            
            Returns:
            ------------
            K: kernel matrix of size n x n, where K_ij = k(x_i,x_j) = <x_i, x_j> ^ deg.
        """
        return np.dot(X,X.T)**deg
        
    def gaussian_kernel(bw):
        """ Gaussian kernel.
            Params:
            ------------
            bw: bandwidth parameter of Gaussian kernel
            
            Returns:
            -------------
            K: kernel matrix of size n x n, where K_ij = exp(-||x_i-x_j||^2 / bw^2).
        """
        sqDist = euclidean_distances(X, X, squared=True)
        return np.exp(-sqDist/(bw*bw))
    
    n, d = X.shape
    if ncomp > d: 
        ncomp = d
    
    H = np.eye(n, dtype='float64') - 1.0/n * np.ones((n,n),dtype='float64') # centering matrix
    
    if kernel_type == 'gauss':
        K = gaussian_kernel(c)
    elif kernel_type == 'poly':
        K = poly_kernel(d)
    else:
        raise ValueError, "Invalid kernel type!"
    
    HKH = np.dot(np.dot(H,K),H)
    w, v = np.linalg.eig(HKH)
    sorted_ind = np.argsort(w)[::-1]
    w = w[sorted_ind]
    v = v[:,sorted_ind]
    A = v[:,0:ncomp]
    D = np.diag(w[0:ncomp]**(-0.5))
    P1 = np.dot(np.dot(D, A.T),H) # D tA H
    X_kpca = np.dot(P1, K) - 1.0/n * np.dot(P1, np.dot(K, np.ones((n,n))))
    X_kpca = X_kpca.T
    
    def myplot():
        """ Plot Kernel PCA embedding.
            For simplicity of color scheme, assume up to 3 classes.
        """
        fig = plt.figure()
        fig.clf()
        n_class = len(np.unique(Y))
        assert n_class <= 3, "Up to 3 classes!"
        cs = ['r', 'g', 'b']
        ms = ['x', 'o', 's']
        labels = ['Class 1', 'Class 2', 'Class 3']
        
        if ncomp == 2:
            for i in range(n_class):
                plt.plot(X_kpca[Y==i, 0], X_kpca[Y==i, 1], cs[i]+ms[i], label=labels[i])
        
            plt.legend(loc='best', prop={'size':20})
            plt.axis('equal')
            plt.axis('off')
        elif ncomp == 3:
            ax = fig.add_subplot(111, projection="3d")
            for i in range(n_class):
                ax.plot(X_kpca[Y==i, 0], X_kpca[Y==i, 1], X_kpca[Y==i, 2], cs[i]+ms[i], label=labels[i])
        
            ax.legend(loc='best', prop={'size':20})
            ax.axis('equal')
            ax.axis('off')
            ax.view_init(50,70)
        
        fname='kpca-%s-%s-%dd-c_%d-deg_%d.png' %(dataset, kernel_type, ncomp, c, deg)
        plt.savefig(fname)
        if show:
            plt.show()    
    myplot()
    
def testPCA(X, Y, ncomp=2, dataset='wine'):
    """ PCA. To compare with KPCA."""
    pca = PCA(n_components=ncomp, whiten=True)
    pca.fit(X)
    X_pca = pca.transform(X)
    n_class = len(np.unique(Y))
    
    ms = ['x', 'o', 's']
    cs = ['r', 'g', 'b']
    labels = ['Class 1', 'Class 2', 'Class 3']
    
    fig = plt.figure()
    fig.clf()

    if ncomp == 2:
        for i in range(n_class):
            plt.plot(X_pca[Y==i, 0], X_pca[Y==i, 1], cs[i]+ms[i], label=labels[i])
        plt.legend(loc='best', prop={'size':20})
        plt.axis('equal')
        plt.axis('off')
    elif ncomp == 3:
        ax = fig.add_subplot(111, projection="3d")
        for i in range(n_class):
            ax.plot(X_pca[Y==i, 0], X_pca[Y==i, 1], X_pca[Y==i, 2], cs[i]+ms[i], label=labels[i])
        
        ax.legend(loc='best', prop={'size':20})
        ax.axis('equal')
        ax.axis('off')
        ax.view_init(50,70)
        
    figname = 'pca-%s-%dd' %(dataset, ncomp)
    plt.savefig(figname)
    
def plotData(X, Y, fname):
    """ Use to plot original 2d dataset."""
    n_class = len(np.unique(Y))
    assert n_class <= 3, "Dim <= 3"
    ms = ['x', 'o', 's']
    cs = ['r', 'g', 'b']
    labels = ['Class 1', 'Class 2', 'Class 3']
    fig = plt.figure()
    fig.clf()
    for i in range(n_class):
        plt.plot(X[Y==i, 0], X[Y==i, 1], cs[i]+ms[i], label=labels[i])
    plt.legend(loc='best', prop={'size':20})
    plt.axis('equal')
    plt.savefig(fname)
    
def withWineData():
    def loadWineData(fnames=['normalized-wine-c1.txt', 'normalized-wine-c2.txt', 'normalized-wine-c3.txt']):
        X = []
        Y = []
        label = 0
        for fname in fnames:
            with open(fname, 'rt') as fin:
                for line in fin:
                    row = map(float, line.strip().split(','))
                    X.append(row)
                    Y.append(label)
            label += 1
    
        X = np.asarray(X, dtype='float64')
        Y = np.asarray(Y, dtype='int16')
        return X, Y
    
    X, Y = loadWineData()
    testPCA(X, Y, ncomp=2, dataset='wine')
    myKPCA(X, Y, kernel_type='gauss', c=3, deg=2, ncomp=2, dataset='wine')
    #myKPCA(X, Y, kernel_type='poly', c=3, deg=1, ncomp=2, dataset='wine')

def withCircleData():
    np.random.seed(0)
    X, Y = make_circles(n_samples=400, noise=.05, factor=.3)
    #plotData(X, Y, 'original-circle.png')
    #testPCA(X, Y, ncomp=2, dataset='circles')
    #myKPCA(X, Y, kernel_type='gauss', c=1, deg=2, ncomp=2, dataset='circles')
    myKPCA(X, Y, kernel_type='poly', c=1, deg=10, ncomp=2, dataset='circles')
    
def withMoonData():
    np.random.seed(0)
    X, Y = make_moons(n_samples=400, noise=.05)
    plotData(X, Y, 'original-moon.png')
    testPCA(X, Y, ncomp=2, dataset='moons')
    myKPCA(X, Y, kernel_type='gauss', c=3, deg=2, ncomp=2, dataset='moons')

def withIrisData():
    iris = load_iris()
    X = iris.data
    Y = iris.target
    #testPCA(X, Y, ncomp=2, dataset='iris')
    myKPCA(X, Y, kernel_type='poly', c=1, deg=3, ncomp=2, dataset='iris')

def withDigitData(n_class):
    digits = load_digits(n_class=n_class)
    X = digits.data
    Y = digits.target
    #testPCA(X, Y, ncomp=2, dataset='digit')
    myKPCA(X, Y, kernel_type='gauss', c=10, deg=3, ncomp=2, dataset='digits')
if __name__ == '__main__':
    #withWineData()
    #withCircleData()
    #withMoonData()
    #withIrisData()
    withDigitData(3)