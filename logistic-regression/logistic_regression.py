"""
	Simple Implementation of Logistic Regression (LR) for binary classification.
	Label (class): y \in {+1, -1}.
	Model: p(y = +1 | x) = g(w'x),	 
	where g is a logistic function, w' means w transpose.
""" 

import numpy as np
import scipy.optimize

def logistic(z):
	""" Logistic (sigmoid) function."""

	return 1.0/(1.0 + np.exp(-z))
	
def predict(w,x):
	""" If p(y = +1 | x) > 0.5, then class y = +1. Otherwise y = -1."""
	return logistic(np.dot(w,x)) > 0.5 or -1

"""
	Training phase: MLE + L2-regularized log-likelihood.
	To maximize the log-likelihood, we use gradient ascent for simplicity.
"""
def log_likelihood(X, Y, w, C=0.1):
	""" 
		Log likelihood function with regularization parameter C >= 0 (default: C = 0.1)
		See README for more details.
	"""
	return np.sum(np.log(logistic(Y*np.dot(X,w))) - C/2*np.dot(w,w))

def log_likelihood_grad(X, Y, w, C=0.1):
	""" Compute the gradient of log-likelihood function above."""
	
	# K is dimensionality of a sample.
	K = len(w)
	# N is #training samples.
	N = len(X)
	s = np.zeros(K)
	
	for i in range(N):
		s += Y[i] * X[i] * logistic(-Y[i]*np.dot(X[i],w))
		
	s -= C*w
	return s
	
def grad_num(X, Y, w, f, eps=0.00001):
	""" Compute gradient numerically."""
	
	K = len(w)
	ident = np.identity(K)
	g = np.zeros(K)
	
	for i in range(K):
		g[i] += f(X,Y,w+eps*ident[i])
		g[i] -= f(X,Y,w-eps*ident[i])
		g[i] /= 2*eps
	return g
	
def train(X, Y, C=0.1):
	"""
		Training phase by using BFGS algorithm to minimize -log-likelihood function.
	"""
	def f(w):
		return -log_likelihood(X,Y,w,C)
	def fprime(w):
		return -log_likelihood_grad(X,Y,w,C)
	
	K = X.shape[1]
	initial_guess = np.zeros(K)
	
	return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)
	
def accuracy(X,Y,w):
	"""
		First, evaluate the classifier on training data.
	"""
	n_correct = 0
	for i in range(len(X)):
		if predict(w,X[i]) == Y[i]:
			n_correct += 1
	return n_correct*1.0 / len(X)

# Splitting data in ith fold.
def fold(arr, K, i):
	""" 
		Split origin data to training data + validation data.
		Return: Tuple (hold-out samples, training samples).
	"""
	N = len(arr)
	size = np.ceil(1.0 * N / K)
	arange = np.arange(N)
	holdout = np.logical_and(i*size <= arange, arange < (i+1)*size)
	rest = np.logical_not(holdout)
	return arr[holdout], arr[rest]
	
def kfold(arr, K):
	""" K-fold training data splitting."""
	return [fold(arr, K, i) for i in range(K)]

def avg_accuracy(all_x, all_y, C):
	"""
		Calculate the CV score for each setup (C).
	"""
	s = 0
	# nfolds is number of folds.
	nfolds = len(all_x)
	for i in range(nfolds):
		x_holdout, x_rest = all_x[i]
		y_holdout, y_rest = all_y[i]
		
		# Training phase to get w
		w = train(x_rest, y_rest, C)
		
		# Evaluate with hold-out samples
		s += accuracy(x_holdout, y_holdout, w)
	return s * 1.0 / nfolds
	
def train_C(X, Y, K=10):
	""" Using k-fold CV to optimize C."""
	# C candidates
	all_C = np.arange(0, 1, 0.1)
	all_X = kfold(X,K)
	all_Y = kfold(Y,K)
	all_acc = np.array([avg_accuracy(all_X, all_Y, C) for C in all_C])
	return all_C[all_acc.argmax()]

def read_data(filename, sep=",", aFilter=int):
	""" 
		Data format(ith line): label_i,x_i1,x_i2,...x_id
	"""
	def split_line(line):
		return line.split(sep)
	def process_line(line):
		return map(aFilter, split_line(line))
		
	with open(filename) as f:
		lines = map(process_line, f.readlines())
		# adding x[0] = 1 for all rows
		X = np.array([ [1]+line[1:] for line in lines ])
		# the first column is class label of training data
		Y = np.array([line[0] or -1 for line in lines])
	return X, Y

def main(training_file='SPECT.train.txt', test_file='SPECT.test.txt'):
	# Read data
	X_train, Y_train = read_data(training_file)
	
	# Training phase
	C = train_C(X_train, Y_train)
	print "Optimized C was", C
	w = train(X_train, Y_train, C)
	print 'w was', w
	
	# Testing phase
	X_test, Y_test = read_data(test_file)
	print 'Accuracy was', accuracy(X_test, Y_test, w)
	
if __name__ == '__main__':
	main()
		