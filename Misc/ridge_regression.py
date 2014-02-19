import numpy as np
import matplotlib.pyplot as plt

class RidgeRegressor(object):
    """Simple implementation of ridge regression.
        TODO: Add cross-validation procedure to select regularization param reg.
    """
    
    def __init__(self, reg=0.0):
        """ Argument
                reg: regularization parameter.
                In the case of ordinary least-squares regression, reg = 0.
        """
        self.reg = reg
    
    def train(self, X, y):
        """ Learning parameters (model) with training samples.
            Arguments
                X: nxd matrix of n d-dimensional samples 
                y: output vector corresponding to training samples
        """
        n, d = X.shape
        # Adding bias terms: X -> n x (d+1) matrix
        X = np.hstack((np.ones((n, 1)), X))
        R = self.reg * np.eye(d+1)
        # No regularization for bias term.
        R[0,0] = 0
        
        # Learning params: (X.T X + reg I)w = X.T y  
        A = np.dot(X.T, X) + R
        b = np.dot(X.T, y)
        self.params = np.linalg.solve(A, b)
        
    def predict(self, X):
        """ Predict output of new test samples using learned params.
            Arguments
                X: m x d matrix of m d-dimensional new samples
            Return
                Output vector (m x 1) corresponding to m given test samples.
        """
        m, d = X.shape
        # Adding bias terms
        X = np.hstack((np.ones((m, 1)), X))
        return np.dot(X, self.params)

def test(n=1000):
    """ Test RidgeRegressor with synthetic 1d data.
    """
    X = np.linspace(-np.pi, np.pi, n)
    y = 2*np.sin(X) + 3*np.cos(X)
    # Adding Gaussian noise N(0, 0.5^2)
    y_noise = y + 0.5*np.random.normal(size=n)
    
    # Create design/feature matrix
    X2 = np.power(X, 2)
    X3 = np.power(X, 3)
    Xtr = np.c_[X, X2, X3]
    
    regressor = RidgeRegressor(reg=0.0)
    regressor.train(Xtr, y_noise)
    output = regressor.predict(Xtr)
    
    plt.figure()
    plt.plot(X, y, 'g', label='True')
    plt.plot(X, y_noise, 'r+', label='Noisy data')
    plt.plot(X, output, 'bx', label='Ridge regression(reg=%.1f)' %regressor.reg)
    plt.legend(loc='best')
    plt.show()
    
if __name__ == '__main__':
    test()
