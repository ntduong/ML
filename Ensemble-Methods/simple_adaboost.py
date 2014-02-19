# Simple implementation of Adaboost

from random import choice
import numpy as np
import matplotlib.pyplot as plt
 
def plot_data(data, label):
    """
    Helper function to plot data.
    """
    n, d = data.shape
    if d > 2: 
        raise ValueError, "don't support plotting for data with dimensionality > 2."
    pind = (label == 1)
    nind = (label == -1)
    plt.plot(data[pind,0], data[pind,1], 'b*')
    plt.plot(data[nind,0], data[nind,1], 'rx')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.show()
    
def linear_data(n):
    """
    Generate n 2-d samples for linearly-separable case.
    """
    data = np.random.randn(n, 2)
    pind = (data[:,0] > data[:,1])
    nind = (data[:,0] <= data[:,1])
    label = np.ones(n, dtype=int)
    label[nind] = -1
    
    return data, label
    
def non_linear_data(n):
    """
    Generate n 2-d samples for non-linearly separable case.
    """
    data = np.random.randn(n, 2)
    pind = (data[:,0]**2 + data[:,1]**2 <= 1)
    nind = (data[:,0]**2 + data[:,1]**2 > 1)
    label = np.ones(n, dtype=int)
    label[nind] = -1
    
    return data, label

class WeakClassifier(object):
    """
    Dummy classifier :)
    """
    def __init__(self, threshold, dim, dir):
        self.threshold = threshold
        self.dim = dim
        self.dir = dir
        
    def classify(self, x):
        """
        Classify a sample x into {+1,-1}.
        """
        dir = self.dir # 1 or -1
        dim = self.dim # 0 or 1 for 2-d data
        threshold = self.threshold
        if x[dim] >= threshold:
            return 1*dir
        else:
            return -1*dir

def adaboost(data, label, T=10):
    """
    T: #weak classifiers.
    """
    
    # Get #training samples and dimensionality of data
    ntr, dim = data.shape
    # Sanity check data size
    assert ntr == label.size, "size must match!!"
    
    # First, initialize all sample-weights to 1./ntr
    D = np.ones(ntr) / ntr
    # Classifier-weights
    alpha = []
    # Classifier dict
    h = {}
    
    error = np.ones(T) * np.inf
    # threshold candidate for weak classifier
    thresholds = np.arange(-5,5,0.1)
    
    for t in range(T):    
        for ti in range(len(thresholds)): # classifier threshold
            for d in range(2): # classifier dim
                for dir in [-1,1]: # classifier dir
                    tmp_wc = WeakClassifier(thresholds[ti], d, dir)
                    ind = (np.asarray(tmp_wc.classify(x) for x in data) != label)
                    tmpe = np.sum(D[ind])
                    if tmpe < error[t]:
                        error[t] = tmpe
                        h[t] = tmp_wc                            
        
        # Stop even we still don't have T classifiers yet!
        if error[t] >= 0.5:
            print 'Stop because error rate >= 1./2!!!'
            break
        
        # Compute weight for t-th classifier: lower error rate --> higher weight
        alpha.append(0.5 * np.log((1.0-error[t])/error[t]))
        
        # Update samples weight based on t-th classification result 
        res = np.asarray([h[t].classify(x) for x in data])
        
        # misclassified samples get higher weights, because label*res = -1, alpha[t] > 0
        D = D * np.exp(-alpha[t] * label * res)
        # normalize D to have total sum = 1
        D /= np.sum(D)
        
    finalLabel = np.zeros(ntr)
    
    for i in range(ntr):
        finalLabel[i] = np.sign(np.sum(alpha[t] * h[t].classify(data[i]) for t in range(len(alpha))))
        if finalLabel[i] == 0: finalLabel[i] = choice([-1,1]) # random choice when sign = 0.
    
    # Compute misclassification rate on training data
    miss_rate = float(np.sum(finalLabel != label)) / ntr
    print "Misclassification rate:", miss_rate
    
    # Also, plotting final classification result by adaboosted classifier
    #plot_data(data, finalLabel)
    
def test_WeakClassifier(data, label, thresholds, dirs, dim=2):
    th = choice(thresholds)
    d = choice(range(dim))
    dir = choice(dirs)
    random_WC = WeakClassifier(th, d, dir)
    clabel = np.asarray([random_WC.classify(x) for x in data])
    mis_rate = float(np.sum(clabel != label)) / len(label)
    print "Misclassification rate:", mis_rate
    #plot_data(data, clabel)
    
if __name__ == '__main__':
    data, label = non_linear_data(1000)
    #plot_data(data, label)
    adaboost(data, label)
    #test_WeakClassifier(data, label, np.arange(-3,3,0.1), [-1,1])