'''
Created on May 18, 2013
@author: Administrator
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets import make_circles, make_moons

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

def myKPCA(X, Y, c=3, ncomp=2, dataset='wine'):
    n = X.shape[0]
    H = np.eye(n, dtype='float64') - 1.0/n * np.ones((n,n),dtype='float64')
    sqDist = euclidean_distances(X, X, squared=True)
    K = np.exp(-sqDist/(c*c))
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
        fig = plt.figure()
        fig.clf()
        n_class = len(np.unique(Y))
        cs = ['r', 'g', 'b']
        ms = ['x', 'o', '+']
        labels = ['C1', 'C2', 'C3']
        for i in range(n_class):
            plt.plot(X_kpca[Y==i, 0], X_kpca[Y==i, 1], cs[i]+ms[i], label=labels[i])
        
        plt.legend(loc='best', prop={'size':20})
        plt.axis('equal')
        fname='KPCA-%s-%dd-%d.png' %(dataset, ncomp, c)
        plt.savefig(fname)
        #plt.show()
        
    myplot()
    #return X_kpca.T
    
def testKPCA(X, Y, ncomp=2):
    kpca = KernelPCA(n_components=ncomp, kernel="rbf", gamma=3)
    kpca.fit(X)
    X_kpca = kpca.transform(X)
    n_class = len(np.unique(Y))
    
    def plotKPCA():
        fig = plt.figure()
        fig.clf()
        cs = ['r', 'g', 'b']
        ms = ['x', 'o', '+']
        labels = ['Class 1', 'Class 2', 'Class 3']
        for i in range(n_class):
            plt.plot(X_kpca[Y==i, 0], X_kpca[Y==i, 1], cs[i]+ms[i], label=labels[i])
            
        plt.legend(loc='best', prop={'size':20})
        plt.axis('equal')
        plt.show()
    
    plotKPCA()
    
def testPCA(X, Y, ncomp=2, fname='PCA.png'):
    pca = PCA(n_components=ncomp, whiten=True)
    pca.fit(X)
    X_pca = pca.transform(X)
    n_class = len(np.unique(Y))
    ms = ['x', 'o', '+']
    cs = ['r', 'g', 'b']
    labels = ['Class 1', 'Class 2', 'Class 3']
    fig = plt.figure()
    fig.clf()
    for i in range(n_class):
        plt.plot(X_pca[Y==i, 0], X_pca[Y==i, 1], cs[i]+ms[i], label=labels[i])
    plt.legend(loc='best', prop={'size':20})
    #plt.axis('equal')
    plt.savefig(fname)
    
def plotData(X, Y, fname):
    n_class = len(np.unique(Y))
    assert n_class <= 3, "Dim <= 3"
    ms = ['x', 'o', '+']
    cs = ['r', 'g', 'b']
    labels = ['Class 1', 'Class 2', 'Class 3']
    fig = plt.figure()
    fig.clf()
    for i in range(n_class):
        plt.plot(X[Y==i, 0], X[Y==i, 1], cs[i]+ms[i], label=labels[i])
    plt.legend(loc='best', prop={'size':20})
    #plt.axis('equal')
    plt.savefig(fname)
    
if __name__ == '__main__':
    #X, Y = loadWineData()
    #X, Y = make_circles(n_samples=400, noise=.05, factor=.3)
    #X, Y = make_moons(n_samples=400, noise=0.05)
    #myKPCA(X, Y, c=6, ncomp=2, dataset='circle')
    from sklearn.datasets import load_iris, load_digits
    #iris = load_iris()
    #X = iris.data
    #Y = iris.target
    digit = load_digits(3)
    X = digit.data
    Y = digit.target
    #testPCA(X,Y,ncomp=2,fname='digit-pca.png')
    myKPCA(X, Y, c=6, ncomp=2, dataset='digit')