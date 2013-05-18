"""
Simple implementation of Fisher Discriminant Analysis
Demo with 2D toy data

Author: Duong Nguyen @TokyoTech
Contact: nguyen@sg.cs.titech.ac.jp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig # for generalized eigensystem solver
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

#----------------#
#    FDA code    # 
#----------------#
def FDA(X, Y, n_class, m):
    """ Fisher Discriminant Analysis.
        Params:
        -------------
        X: n x d matrix of samples
        Y: n x 1 label vector
        n_class: number of classes
        m: reduced dimensionality
        
        Returns:
        ---------------
        B_FDA: m x d matrix (FDA embedding matrix)
    """
    n, d = X.shape
    
    nc = [] # number of samples from each class 
    muc = [] # list of class sample means  
    for i in range(n_class):
        nc.append(len(Y[Y==i]))
        muc.append(np.mean(X[Y==i], 0)) 
    
    mu_X = np.mean(X, 0) # sample mean of the whole data

    # Compute Sb
    Sb = np.zeros((d,d), dtype='float64')
    
    for i in range(n_class):
        Sb += nc[i] * np.dot(np.reshape(muc[i]-mu_X, (d,1)), np.reshape(muc[i]-mu_X, (1,d))) 
    
    # compute Sw
    Sw = np.zeros((d,d), dtype='float64')
    for i in range(n_class):
        Sw += np.dot((X[Y==i]-muc[i]).T, (X[Y==i]-muc[i]))
    
    eigvals, V = eig(Sb, Sw)
    sorted_indices = np.argsort(eigvals)[::-1]
    V = V[:, sorted_indices] # sort in decreasing order of eigenvalues
    return V[:, 0:m] # return the m largest component
    
def testFDA(outfile='fda-2d-L3.png', show=True, debug=False):
    """ Test FDA."""
    X1, Y1, n1, d1 = loadData('2d-L3-c1.txt', 0)
    X2, Y2, n2, d2 = loadData('2d-L3-c2.txt', 1)
        
    assert d1 == d2, "Dimensions must match!"
    
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2))
    fda_dir = FDA(X, Y, n_class=2, m=1)
    
    if debug: # print fda_dir in debug mode
        print fda_dir
    
    x_mean = np.mean(X, 0)
    
    plt.plot(X1[:,0], X1[:,1], 'rx', markersize=10, mew='2', label='Class 1')
    plt.plot(X2[:,0], X2[:,1], 'bo', markersize=10, mew='2', mfc='w', mec='b', label='Class 2')
    
    plt.plot([x_mean[0] + k*fda_dir[0] for k in np.linspace(-9,9)], 
                [x_mean[1] + k*fda_dir[1] for k in np.linspace(-9,9)],
                'k-', lw=4, label='FDA')
    
    plt.legend(loc='best', prop={'size':20})
    plt.axis('equal')
    plt.savefig(outfile)
    if show:
        plt.show()

if __name__ == '__main__':
    testFDA()
    