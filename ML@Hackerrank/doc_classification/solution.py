'''
Created on 2013/10/26
@author: duong
'''

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def load_training_data(fname="trainingdata.txt"):
    corpus = []
    cat = []
    with open(fname, "r") as fin:
        n_doc = int(fin.readline())
        for line in fin:
            lb, doc = line.strip().split(" ", 1)
            corpus.append(doc.lower())
            cat.append(int(lb))
            
    assert n_doc == len(corpus), "Invalid training data!"
    return cat, corpus
    
def classifier_1(corpus, cat):
    """ Multinomial Naive Bayes with raw word count as features."""
    count_vectorizer = CountVectorizer(min_df=1, stop_words=['the', 'a'], ngram_range=(1,2))
    counts = count_vectorizer.fit_transform(corpus) # counts is n_samples x n_features matrix
    cf = MultinomialNB()
    cf.fit(counts, cat)
    return cf, count_vectorizer

def predict(cf, count_vec, fname="test.txt"):
    test_samples = []
    with open(fname, "r") as fin:
        n_test = int(fin.readline())
        for line in fin:
            test_samples.append(line.strip())
    
    assert len(test_samples) == n_test, "Malformed test file!"
    test_counts = count_vec.transform(test_samples)
    return cf.predict(test_counts) 
    
def pipelization(corpus, cat, fname="test.txt"):
    pipeline = Pipeline([('vectorization', CountVectorizer(ngram_range=(1,2),min_df=1)),
                         ('tfidf_transformer', TfidfTransformer()),
                         ('classifier', MultinomialNB())
                         ])
    pipeline.fit(corpus, cat)
    test = []
    with open(fname, "r") as fin:
        n_test = int(fin.readline())
        for line in fin:
            test.append(line.strip().lower())
            
    return pipeline.predict(test)
    
def main():
    cat, corpus = load_training_data()
    test = []
    n_test = int(raw_input())
    for i in xrange(n_test):
        test.append(raw_input().strip().lower())
    
    pipeline = Pipeline([('vectorization', CountVectorizer(ngram_range=(1,2),min_df=1)),
                         ('tfidf_transformer', TfidfTransformer()),
                         ('classifier', MultinomialNB())
                         ])
    pipeline.fit(corpus, cat)
    for ans in pipeline.predict(test):
        print ans
        
    
if __name__ == "__main__":
    cat, corpus = load_training_data()
    for ans in pipelization(corpus, cat):
        print ans