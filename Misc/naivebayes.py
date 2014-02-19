"""    Simple implementation of mutinomial Naive Bayes for text classfification.
    TODO: Apply to 20 Newsgroups, Reuters-21578 datasets
"""

__author__ = 'Duong Nguyen'
__version__ = '0.0'

import math
import sys
from collections import defaultdict

class NaiveBayes(object):
    """ Multinomial Naive Bayes"""
    def __init__(self):
        self.categories = set()
        self.vocabularies = set()
        self.wordcount = {}
        self.catcount = {}
        self.denom = {}
        
    def train(self, data):
        for d in data:
            cat = d[0]
            self.categories.add(cat)
        
        for cat in self.categories:
            self.wordcount[cat] = defaultdict(int)
            self.catcount[cat] = 0
            
        for d in data:
            cat, doc = d[0], d[1:]
            self.catcount[cat] += 1
            for word in doc:
                self.vocabularies.add(word)
                self.wordcount[cat][word] += 1
        
        for cat in self.categories:
            self.denom[cat] = sum(self.wordcount[cat].values()) + len(self.vocabularies)
    
    def wordProb(self, word, cat):
        """ Compute P(word|cat) with Laplace smoothing.
        """
        return float(self.wordcount[cat][word] + 1) / self.denom[cat]
        
    def docProb(self, doc, cat):
        """ Compute log P(cat|doc) = log P(cat) + sum_i log P(word_i|cat)
        """
        total = sum(self.catcount.values()) # number of docs in training data
        score = math.log(float(self.catcount[cat])/total) # log P(cat)
        for word in doc:
            score += math.log(self.wordProb(word, cat)) # + sum_i log P(word_i|cat)
            
        return score
        
        
    def classify(self, doc):
        """ Classify doc by argmax_cat log P(cat|doc).
        """
        best = None
        maxP = -sys.maxint
        for cat in self.categories:
            p = self.docProb(doc, cat)
            if p > maxP:
                maxP = p
                best = cat
                
        return best
    
if __name__ == '__main__':
    pass