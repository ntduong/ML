# Simple implementation of perceptron algorithm in online learning
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

def linear_kernel(x1, x2):
    return np.dot(x1, x2)
    
def polynomial_kernel(x1, x2, d=3):
    return (1 + np.dot(x1, x2)) ** d
    
def gaussian_kernel(x1, x2, sigma=5.0):
    return np.exp(-linalg.norm(x1-x2)**2/(2*(sigma**2)))
    

class Perceptron(object):
    def __init__(self, T=1):
        self.T = T
    
    def fit(self, X, y):
        n_samples, dim = X.shape
        self.w = np.zeros(dim, dtype=np.float64)
        self.b = 0.0
        
        for t in range(self.T):
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i]
                    self.b += y[i] * 1
                    
    def project(self, x):
        return np.dot(x, self.w) + self.b
        
    def predict(self, x):
        x = np.atleast_2d(x)
        return np.sign(self.project(x))
        
class KernelPerceptron(object):
    def __init__(self, kernel=linear_kernel, T=1):
        self.T = T
        self.kernel = kernel
        
    def fit(self, X, y):
        n_samples, dim = X.shape
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        
        # Compute Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
                
        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0
                    
        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" %(len(self.alpha), n_samples)
        
    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict
        
    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, dim = X.shape
        return np.sign(self.project(X))
        
def gen_lin_separable_data(n=100):
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[0.8, 0.6],[0.6, 0.8]])
    
    X1 = np.random.multivariate_normal(mean1, cov, n)
    y1 = np.ones(n)
    X2 = np.random.multivariate_normal(mean2, cov, n)
    y2 = np.ones(n) * (-1)
    
    return X1, y1, X2, y2

def gen_non_lin_separable_data(n=100):
    mu1 = [-1,2]
    mu2 = [1,-1]
    mu3 = [4,-4]
    mu4 = [-4,4]
    cov = [[1.0, 0.8],[0.8, 1.0]]
    X1_1 = np.random.multivariate_normal(mu1, cov, n/2)
    X1_2 = np.random.multivariate_normal(mu3, cov, n/2)
    X1 = np.vstack((X1_1, X1_2))
    y1 = np.ones(n)
    
    X2_1 = np.random.multivariate_normal(mu2, cov, n/2)
    X2_2 = np.random.multivariate_normal(mu4, cov, n/2)
    X2 = np.vstack((X2_1, X2_2))
    y2 = np.ones(n) * (-1)
    
    return X1, y1, X2, y2
    
def split_data(X1, y1, X2, y2, n=100):
    ntr = int(n*0.9)
    X1_tr, X1_te = X1[:ntr], X1[ntr:]
    y1_tr, y1_te = y1[:ntr], y1[ntr:]
    X2_tr, X2_te = X2[:ntr], X2[ntr:]
    y2_tr, y2_te = y2[:ntr], y2[ntr:]
    
    X_tr = np.vstack((X1_tr, X2_tr))
    y_tr = np.hstack((y1_tr, y2_tr))
    X_te = np.vstack((X1_te, X2_te))
    y_te = np.hstack((y1_te, y2_te))
    
    return X_tr, y_tr, X_te, y_te
    
def plot_margin(X1_train, X2_train, clf):
    def f(x, w, b, c=0):
    """
    Line: w[0].x + w[1].y + b = c. Compute y.
    """
        return (c - w[0] * x - b) / w[1]
    
    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    
    # plot line w.x + b = 0
    a0, b0 = -4, 4
    a1 = f(a0, clf.w, clf.b)
    b1 = f(b0, clf.w, clf.b)
    plt.plot([a0,b0], [a1,b1], "k")
    plt.axis("tight")
    plt.show()
        
def plot_contour(X1_train, X2_train, clf):
    plt.plot(X1_train[:,0], X1_train[:,1], "ro")
    plt.plot(X2_train[:,0], X2_train[:,1], "bo")
    
    # plot support vectors
    plt.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")
    
    X1, X2 = np.meshgrid(np.linspace(-6,6,50),np.linspace(-6,6,50))
    X = np.array([[x1,x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.project(X).reshape(X1.shape)
    plt.contour(X1, X2, Z, [0.0], colors="k", linewidths=1, origin="lower")
    plt.axis("tight")
    plt.show()
    

def test_linear():
    X1, y1, X2, y2 = gen_lin_separable_data()
    X_tr, y_tr, X_te, y_te = split_data(X1, y1, X2, y2)
    
    clf = Perceptron(T=3)
    clf.fit(X_tr, y_tr)
    y_predict = [clf.predict(x)[0] for x in X_te]
    correct = np.sum(y_predict == y_te)
    print "%d out of %d predictions correct!" %(correct, len(y_predict))
    
    plot_margin(X_tr[y_tr==1], X_tr[y_tr==-1], clf)

def test_kernel():
    X1, y1, X2, y2 = gen_non_lin_separable_data()
    X_tr, y_tr, X_te, y_te = split_data(X1, y1, X2, y2)
    
    clf = KernelPerceptron(kernel=gaussian_kernel, T=20)
    clf.fit(X_tr, y_tr)
    y_predict = clf.predict(X_te)
    correct = np.sum(y_predict == y_te)
    print "%d out of %d predictions correct" %(correct, len(y_predict))
    
    plot_contour(X_tr[y_tr==1], X_tr[y_tr==-1], clf)
    
if __name__ == '__main__':
    #test_linear()
    test_kernel()