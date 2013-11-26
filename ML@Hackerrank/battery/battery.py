'''
Created on 2013/10/27
@author: Duong Nguyen @TokyoTech
@problem: Laptop Battery Life
'''

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

def train(fname="trainingdata.txt"):
    X, Y = [], []
    with open(fname, "r") as fin:
        for line in fin:
            x, y = map(float, line.strip().split(","))
            X.append(x)
            Y.append(y)
            
    X = np.array(X)
    X = X[:, np.newaxis]
    Y = np.array(Y)
    gbreg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, 
                                      max_depth=1, random_state=0, loss="ls").fit(X,Y)
    #print mean_squared_error(Y, gbreg.predict(X))
    return gbreg

if __name__ == "__main__":
    gbreg = train()
    test_input = float(raw_input())
    print "%0.2f" % gbreg.predict(test_input)