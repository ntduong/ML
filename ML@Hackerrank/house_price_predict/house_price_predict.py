'''
@author: Duong Nguyen @TokyoTech
@contact: ntduong268(at).gmail.com
@note: Predicting House Prices for Charlie @Hackerrank.com
'''
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV

def read_input():
    F, N = map(int, raw_input().strip().split())
    Xtr, Ytr = [], []
    for _ in xrange(N):
        inp = map(float, raw_input().strip().split())
        assert len(inp) == F+1, "Invalid input!"
        Xtr.append(inp[:F])
        Ytr.append(inp[F])
        
    Xtr = np.array(Xtr)
    Ytr = np.array(Ytr)
    
    T = int(raw_input())
    Xte = []
    for _ in xrange(T):
        inp = map(float, raw_input().strip().split())
        assert len(inp) == F, "Invalid input!"
        Xte.append(inp)
    
    Xte = np.array(Xte)
    
    return Xtr, Ytr, Xte

def train_ridge(Xtr, Ytr):
    reg = RidgeCV(alphas=np.linspace(0.1, 1, 5, endpoint=True))
    reg.fit(Xtr, Ytr)
    return reg
    
def train_lasso(Xtr, Ytr):
    reg = LassoCV(alphas=np.linspace(0.1, 1, 5, endpoint=True))
    reg.fit(Xtr, Ytr)
    return reg

def predict(reg, Xte):
    return reg.predict(Xte)
    
def main(method="ridge", fromfile=False, fname=None):
    if fromfile and fname:
        import sys
        sys.stdin = open(fname, "r")
        
    Xtr, Ytr, Xte = read_input()
    if method == "ridge":
        reg = train_ridge(Xtr, Ytr)
    elif method == "lasso":
        reg = train_lasso(Xtr, Ytr)
        
    for y in predict(reg, Xte):
        print "%.2f" %y
    
if __name__ == "__main__":
    main(method="lasso", fromfile=True, fname="house_in.txt")