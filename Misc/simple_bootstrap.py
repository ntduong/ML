# Simple bootstrapping snippet
import numpy as np

def bootstrap(X, n=None):
    """ Sampling n items from X with replacement.
        @param X: n x d of n d-dimensional samples
    """
    
    if n == None:
        n = len(X)
        
    resample_id = np.floor(np.random.rand(n)*len(X)).astype(int)
    rX = X[resample_id]
    
    return rX
 