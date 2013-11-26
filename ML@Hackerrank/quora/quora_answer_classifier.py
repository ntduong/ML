'''
Created on 2013/10/28
@author: Duong Nguyen @TokyoTech
@problem: Quora Answer Classifier
'''

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def read_input():
    m, n = map(int, raw_input().strip().split())
    Xtr = []
    Ytr = []
    for i in xrange(m):
        line = raw_input().strip().split()
        row = []
        #name = line[0]
        label = int(line[1])
        Ytr.append(label)
        assert n == len(line)-2, "Invalid input!"
        for ft in line[2:]:
            id, val = ft.split(":")
            row.append(float(val))
        Xtr.append(row)
    
    Xtr = np.array(Xtr)
    Ytr = np.array(Ytr)
    
    Xte = []
    names = []
    q = int(raw_input())
    for i in xrange(q):
        line = raw_input().strip().split()
        name = line[0]
        names.append(name)
        row = []
        for ft in line[1:]:
            id, val = ft.split(":")
            row.append(float(val))
        Xte.append(row)
    Xte = np.array(Xte)
    
    return Xtr, Ytr, Xte, names
    
            
def train_rfc(Xtr, Ytr):
    clf = RandomForestClassifier(n_estimators=15, random_state=0, n_jobs=-1).fit(Xtr,Ytr)
    return clf

def train_lr(Xtr, Ytr):
    lr = LogisticRegression().fit(Xtr, Ytr)
    return lr
    
def train_dtree(Xtr, Ytr):
    dtree = DecisionTreeClassifier().fit(Xtr, Ytr)
    return dtree
    
def predict(clf, Xte, names):
    answers = clf.predict(Xte)
    for i, name in enumerate(names):
        print "%s %+d" %(name, answers[i])
    
def main(method):
    Xtr, Ytr, Xte, names = read_input()
    if method == "dtree":
        clf = train_dtree(Xtr, Ytr)
        predict(clf, Xte, names)
    elif method == "logistic":
        clf = train_lr(Xtr, Ytr)
        predict(clf, Xte, names)
    elif method == "rf":
        clf = train_rfc(Xtr, Ytr)
        predict(clf, Xte, names)
        
if __name__ == "__main__":
    main("rf")
    