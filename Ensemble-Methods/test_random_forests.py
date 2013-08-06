""" First try on ensemble methods in sklearn: Random Forests, Gradient Tree Boosting
    See also, http://scikit-learn.org/stable/modules/ensemble.html for more details
    
    (c) Duong Nguyen @Tokyotech
"""

import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import cross_validation as CV

def shuffle_ids(n):
    assert n > 0, "n should be a positive integer!"
    indices = np.arange(n)
    np.random.shuffle(indices)
    return indices
    
def split_ids(ids, percent=0.8):
    n = len(ids)
    return ids[0:int(n*percent)], ids[int(n*percent):]

def test_with_iris():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    n, d = X.shape
    assert n == len(y), "Missing label(s)!"
    
    ids = shuffle_ids(n)
    tr_ids, te_ids = split_ids(ids, percent=0.8)
    
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    rf.fit(X[tr_ids], y[tr_ids])
    print "Score of Random Forests on test data:", rf.score(X[te_ids], y[te_ids])
    pred_probs = rf.predict_proba(X[te_ids])
    print pred_probs # pred_probs is a list of [p1, p2, p3] item, where pi is prob of class i
    
    '''
    gtb = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, 
                                    max_depth=1, random_state=0).fit(X[tr_ids], y[tr_ids])
                                    
    print "Score of Gradient Tree Boosting on test data:", gtb.score(X[te_ids], y[te_ids])
    '''
    
def test_cv():
    """ Test cross validation in sklearn."""
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    n, d = X.shape
    assert n == len(y), "Missing labels!"
    
    rfc = RandomForestClassifier(n_estimators=100)
    
    folds = CV.KFold(n, n_folds=5, indices=False) # indices=False --> true, false indices, not number indices
    cv_scores = []
    
    for tr, te in folds:
        score = rfc.fit(X[tr], y[tr]).score(X[te], y[te])
        cv_scores.append(score)
        
    print "CV Scores:", cv_scores
    print "Mean:", np.array(cv_scores).mean()
    print "Std:", np.array(cv_scores).std()
    
if __name__ == '__main__':
    test_cv()