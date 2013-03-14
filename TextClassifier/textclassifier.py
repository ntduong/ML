# Simple document classifer with Naive-Bayes
import re
import math
from collections import defaultdict

def getWords(doc):
	splitter = re.compile(r'\W*')
	words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]
	return set(words)
	
class Classifier(object):
	def __init__(self, getFeatures):
		self.fc = defaultdict(dict)
		self.cc = defaultdict(int)
		self.getFeatures = getFeatures
		
	def incf(self, f, c):
		self.fc[f].setdefault(c, 0)
		self.fc[f][c] += 1
		
	def incc(self, c):
		self.cc[c] += 1
		
	def fcount(self, f, c):
		"""
		Return the number of occurences of feature f in category c.
		"""
		if f in self.fc and c in self.fc[f]:
			return float(self.fc[f][c])
		else:
			return 0.0
	
	def catcount(self, c):
		"""
		Return the number of documents classified into category c.
		"""
		if c in self.cc:
			return float(self.cc[c])
		else:
			return 0
	
	def totalcount(self):
		"""
		Return the number of documents
		"""
		return sum(self.cc.values())
		
	def categories(self):
		"""
		Return list of categories, e.g: spam/not spam, etc
		"""
		return self.cc.keys()
		
	def train(self, doc, cat):
		"""
		Train classifier with doc that belongs to category cat.
		"""
		features = self.getFeatures(doc)
		for f in features:
			self.incf(f, cat)
		self.incc(cat)
		
	def fprob(self, f, cat):
		"""
		Return P(feature = f | c = cat) ~ freq(f in c) / freq(c)
		"""
		if self.catcount(cat) == 0.0: return 0
		return self.fcount(f, cat) / self.catcount(cat)
	
	def wfprob(self, f, cat, prf, weight=1.0, ap=0.5):
		"""
		Compute weighted probability P(feature = f | c = cat).
		Think this as smoothing technique...?
		"""
		basicprob = prf(f, cat)
		total = sum([self.fcount(f, cat) for cat in self.categories()])
		
		wp = (weight * ap + total * basicprob) / (weight + total)
		return wp
		
class NBClassifier(Classifier):
	def __init__(self, getFeatures):
		Classifier.__init__(self, getFeatures)
		self.name = 'Naive Bayes'
		self.thresholds = {}
		
	def docprob(self, doc, cat):
		"""
		Compute P(d = doc | c = cat) = product(P(feature = f | c = cat)) over f in features of doc.
		Note that in NaiveBayes, we assume that all feature are independent given category.
		"""
		features = self.getFeatures(doc)
		return reduce(lambda f: self.wfprob(f, cat, self.fprob), features, 1.0)
		
	def prob(self, doc, cat):
		"""
		Compute P(c = cat | d = doc) using Bayes rule.
		"""
		catprob = self.catcount(cat) / self.totalcount()
		docprob = self.docprob(doc, cat)
		return docprob * catprob
	
	def setThreshold(self, cat, t):
		self.thresholds[cat] = t
		
	def getThreshold(self, cat):
		if cat not in self.thresholds: return 1.0
		return self.thresholds[cat]
		
	def classify(self, doc, default=None):
		probs = {}
		maxp = 0.0
		for cat in self.categories():
			probs[cat] = self.prob(doc, cat)
			if maxp < probs[cat]:
				best = cat
				maxp = probs[cat]
				
		for cat in self.categories():
			if cat == best: continue
			if probs[cat] * self.getThreshold(best) > probs[best]: return default
		
		return best
		
if __name__ == '__main__':
	pass