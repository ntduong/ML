'''
Created on 2013/06/11
Author: Duong Nguyen
Email: nguyen@sg.cs.titech.ac.jp
Simple implementation of projection pursuit and demo.
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from sklearn.decomposition import PCA

def loadData(datfile, pre_process=True, plot=True):
    """ Load samples data from file.
        
        Returns: X as n x d matrix (ndaray), where
                    n: # of samples
                    d: dimensionality
    """
    X = []
    with open(datfile, 'rt') as fin:
        for line in fin:
            row = map(float, line.strip().split(','))
            X.append(row)
    
    X = np.array(X, dtype='float64')
    
    if pre_process:
        X = pre_processing(X)
        
    if plot:
        fig = plt.figure()
        fig.clf()
        plt.plot(X[:,0], X[:,1], 'rx')
        plt.axis('off')
        figname = datfile.split('.')[0] + '.png'
        if pre_process: figname = 'pre'+figname
        plt.savefig(figname)
        plt.show()
        
    return X
     
def pre_processing(X):
    """ Center and sphere data."""
    eps = 1e-18
    n = X.shape[0]
    cX = X - np.mean(X, axis=0) # centering
    cov_mat = 1.0/n * np.dot(cX.T, cX)
    eigvals, eigvecs = eigh(cov_mat)
    D = np.diag(1./np.sqrt(eigvals+eps)) 
    W = np.dot(np.dot(eigvecs, D), eigvecs.T) # whitening matrix
    wcX = np.dot(cX, W)
    return wcX

def check(X, datname, DEBUG=False):
    """ Check if data is already centering and sphering."""
    n = X.shape[0]
    if DEBUG:
        print 1./n * np.dot(X.T, X)
    fig = plt.figure()
    fig.clf()
    plt.plot(X[:,0], X[:,1], 'rx')
    plt.axis('off')
    figname = datname + '.png'
    plt.savefig(figname)
    plt.show()
    
def proj_pursuit(X, n_iter=1000, learning_rate=0.1, tol=1e-8):
    """ Gradient ascent algorithm for projection pursuit.
        The kurtosis function G(s) = s^4 is used as non-Gaussian measurement.
    """
    
    n, d = X.shape
    #init_b = np.asarray([1.0/np.sqrt(d) for _ in xrange(d)], dtype='float64') 
    init_b = np.random.random((d,))
    init_b /= np.sqrt(sum(init_b*init_b)) # normalization
    
    def obj_func(b):
        tmp = (np.dot(X, b))**4
        return (tmp.mean(axis=0) - 3) ** 2
    
    def gradient(b):
        t1 = (np.dot(X, b))**4
        t1 = t1.mean(axis=0) - 3
        t2 = (np.dot(X, b))**3
        t3 = np.mean(np.array([t2[i] * X[i] for i in xrange(n)]), axis=0)
        return 2.0 * t1 * 4.0 * t3
    
    obj_val = obj_func(init_b)
    proj_b = init_b
    
    for _ in xrange(n_iter):
        proj_b += learning_rate * gradient(proj_b) # update
        proj_b /= np.sqrt(np.sum(proj_b**2)) # normalization
        
        new_obj_val = obj_func(proj_b) # compute new objective value
        
        if np.abs(new_obj_val - obj_val) <= tol: # check convergence
            break
        else:
            obj_val = new_obj_val
        
    return proj_b

def proj_pursuit_2(X, G, g, g_der, n_iter=1000, tol=1e-8):
    """ Approximated Newton-based Projection Pursuit 
        with general non-Gaussian measurement function G(s).
        Note: Consider the effect of outlier to performance of PP
    """
    
    n, d = X.shape
    init_b = np.random.random((d,))
    init_b /= np.sqrt(sum(init_b*init_b)) # normalization
    
    def obj_fun(b):
        # Redundant
        return sum(G(np.dot(b, X[i])) for i in range(n)) / float(n)
    
    def update(b):
        first_term = sum(g_der(np.dot(b, X[i])) for i in range(n)) * b / float(n)
        second_term = np.sum(X[i] * g(np.dot(b, X[i])) for i in range(n)) / float(n)
        return first_term - second_term
    
    proj_b = init_b
    for cnt in xrange(n_iter):
        old_b = proj_b
        proj_b = update(proj_b)
        proj_b /= np.sqrt(np.dot(proj_b, proj_b)) # normalization
        
        if np.dot(proj_b - old_b, proj_b - old_b) <= tol: # check convergence || new_b - old_b ||^2 <= tolerance
            break
                
    #print cnt
    return proj_b
        
def main(datfile):
    """ Testing projection pursuit with kurtosis."""
    X = loadData(datfile, pre_process=True, plot=False)
    proj_b = proj_pursuit(X, n_iter=100, learning_rate=0.01, tol=1e-8)
    
    fig = plt.figure()
    fig.clf()
    
    plt.plot(X[:,0], X[:,1], 'rx', mfc='white', mec='r', ms=7, mew=1) # plot samples
    
    # plot projection pursuit direction
    X_mean = X.mean(axis=0)    
    plt.plot( [proj_b[0] * i + X_mean[0] for i in np.linspace(-7,7)], 
              [proj_b[1] * i + X_mean[1] for i in np.linspace(-7,7)], 'b', lw=4)
    
    figname = 'proj_pursuit' + datfile.split('.')[0] + '.png'
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('fig1/' + figname)
    plt.show()
    
def main2(datfile):
    """ Testing a general projection pursuit. """ 
    X = loadData('data/'+datfile, pre_process=True, plot=False)
    
    G = [lambda s: s**4, lambda s: np.log(np.cosh(s)), lambda s: -np.exp(-(s**2)/2.)]
    g = [lambda s: 4 * (s**3), lambda s: np.tanh(s), lambda s: s*np.exp(-(s**2)/2)]
    g_der = [lambda s: 12 * (s**2), lambda s: 1 - (np.tanh(s))**2, lambda s: (1-s**2)*np.exp(-(s**2)/2)]
    
    func_name = ('s^4', 'log_cosh', 'expo')
    choice = 2
    proj_b = proj_pursuit_2(X, G[choice], g[choice], g_der[choice], n_iter=1000, tol=1e-8)
    
    fig = plt.figure()
    fig.clf()
    plt.plot(X[:,0], X[:,1], 'rx', mfc='white', mec='r', ms=7, mew=1) # plot samples
    
    # plot projection pursuit direction
    X_mean = X.mean(axis=0)    
    plt.plot( [proj_b[0] * i + X_mean[0] for i in np.linspace(-5,5)], 
              [proj_b[1] * i + X_mean[1] for i in np.linspace(-5,5)], 'b', lw=4)
    
    figname = 'general_proj_pursuit_with_%s' % func_name[choice] + datfile.split('.')[0] + '.png'
    plt.axis('equal')
    plt.axis('off')
    plt.savefig('fig2/' + figname)
    plt.show()

if __name__ == '__main__':
    main2('2d-3.txt')
    