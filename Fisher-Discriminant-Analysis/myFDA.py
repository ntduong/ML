'''
Created on 2013/05/08
@author: duong
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def loadData(filename, label):
    X = []
    with open(filename, 'r') as fin:
        for line in fin:
            row = map(float, line.strip().split(','))
            X.append(row)
            
    n_samples = len(X)
    dim = len(X[0])
    Y = [label] * n_samples
    X = np.asarray(X, dtype='float64')
    Y = np.asarray(Y, dtype='float64')
    return X, Y, n_samples, dim
    
def plot_2d_xx(infile1, infile2, outfile=None, show=True):
    X1, Y1, n1, d1 = loadData(infile1, 0)
    X2, Y2, n2, d2 = loadData(infile2, 1)
    d1 = X1.shape[1]
    d2 = X2.shape[1]
    assert d1 == d2, "Dimension not match!"
    assert d1 == 2, "2D!"
    
    plt.plot(X1[:,0], X1[:,1], 'rx', label='Class 1')
    plt.plot(X2[:,0], X2[:,1], 'go', label='Class 2')
    plt.legend(loc='best', prop={'size':20})
    
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()

def FDA(X, n1, n2, ncomp=2):
    """ Fisher Discriminant Analysis.
        Assumption: 2 classes Y1 = 1, Y2 = 2.
        
        Params:
        -------------
        X: n x d matrix of samples.
        Y: n x 1 label vector
        n1: number of samples from class 1
        n2: number of samples from class 2
        ncomp: number of red-dims
        
        Returns:
        ---------------
        B_FDA: m x d matrix
    """
    assert n1+n2 == X.shape[0]
    d = X.shape[1]
    X1 = X[:n1]
    X2 = X[n1:]
    mu1 = np.mean(X1, 0)
    mu2 = np.mean(X2, 0)
    mu = np.mean(X, 0)

    # compute Sb
    Sb = n1 * np.dot(np.reshape(mu1-mu, (d,1)), np.reshape(mu1-mu, (1,d))) + \
        n2 * np.dot(np.reshape(mu2-mu, (d,1)), np.reshape(mu2-mu, (1,d)))
    
    # compute Sw
    Sw = np.dot((X1-mu1).T, (X1-mu1)) + np.dot((X2-mu2).T, (X2-mu2))
    
    eigvals, V = eig(Sb, Sw)
    #trX = np.dot(X, V[:,0:ncomp])
    sorted_indices = np.argsort(eigvals)
    V = V[:, sorted_indices]
    return V[:, -ncomp:]
    
 
def testFDA(outfile='fda-2d-L3.png', show=False):
    """ Test FDA."""
    X1, Y1, n1, d1 = loadData('2d-L3-c1.txt', 0)
    X2, Y2, n2, d2 = loadData('2d-L3-c2.txt', 1)
        
    assert d1 == d2, "Dimension must match!"
    
    X = np.vstack((X1, X2))
    fda_dir = FDA(X, n1, n2, ncomp=1)
    
    print fda_dir
    
    x_mean = np.mean(X, 0)
    
    plt.plot(X1[:,0], X1[:,1], 'rx', label='Class 1')
    plt.plot(X2[:,0], X2[:,1], 'bo', label='Class 2')
    
    plt.plot([x_mean[0] + k*fda_dir[0] for k in np.linspace(-9,8)], 
                [x_mean[1] + k*fda_dir[1] for k in np.linspace(-9,8)],
                'k-', lw=4, label='FDA')
    
    plt.legend(loc='best', prop={'size':20})
    plt.axis('equal')
    plt.savefig(outfile)
    if show:
        plt.show()
    
def LFDA():
    pass

def testLFDA():
    """ Test LFDA."""
    pass

if __name__ == '__main__':
    #plot_2d_xx('2d-L1-c1.txt', '2d-L1-c2.txt', '2d-L1.png')
    #plot_2d_xx('2d-L2-c1.txt', '2d-L2-c2.txt', '2d-L2.png')
    #plot_2d_xx('2d-L3-c1.txt', '2d-L3-c2.txt', '2d-L3.png')
    testFDA()
    