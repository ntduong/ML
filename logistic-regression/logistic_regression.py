# Logistic regression for binary classification y is in {-1,+1}
# p(y=+1|x) = g(wTx), here g is logistic function: R->[0,1], 1-g(z) = g(-z)
import numpy as np
def logistic(z):
	return 1.0/(1.0 + np.exp(-z))
	
def predict(w,x):
	"""if p(y=+1|x) >= 0.5 --> classify to class y=+1. otherwise y=-1."""
	return logistic(np.dot(w,x)) > 0.5 or -1

# training phase with MLE + L2-regularized log-likelihood
# if we choose C = 0, we get non-regularized log likelihood, and get normal MLE solution. It might be sensitive to overfiiting
def log_likelihood(X, Y, w, C = 0.1):
	return np.sum(np.log(logistic(Y*np.dot(X,w)))-C/2*np.dot(w,w))

# There are no close-form solution for maximizing log likelihood above, so just use gradient ascent for simplicity
def log_likelihood_grad(X, Y, w, C = 0.1):
	K = len(w)
	N = len(X)
	s = np.zeros(K)
	
	for i in range(N):
		s += Y[i] * X[i] * logistic(-Y[i]*np.dot(X[i],w))
		
	s -= C*w
	return s
	
def grad_num(X, Y, w, f, eps = 0.00001):
	"Compute gradient numerically."
	K = len(w)
	ident = np.identity(K)
	g = np.zeros(K)
	
	for i in range(K):
		g[i] += f(X,Y,w+eps*ident[i])
		g[i] -= f(X,Y,w-eps*ident[i])
		g[i] /= 2*eps
	return g
	
import scipy.optimize
def train_w(X,Y,C=0.1):
	def f(w):
		return -log_likelihood(X,Y,w,C)
	def fprime(w):
		return -log_likelihood_grad(X,Y,w,C)
	K = X.shape[1]
	initial_guess = np.zeros(K)
	
	return scipy.optimize.fmin_bfgs(f, initial_guess, fprime, disp=False)
	
# Evaluation for training data
def accuracy(X,Y,w):
	n_correct = 0
	for i in range(len(X)):
		if predict(w,X[i]) == Y[i]:
			n_correct += 1
	return n_correct*1.0 / len(X)

# Using k-fold CV
def fold(arr, K, i):
	"Split origin data to training data + validation data."
	N = len(arr)
	size = np.ceil(1.0*N/K)
	arange = np.arange(N)
	heldout = np.logical_and(i*size <= arange, arange < (i+1)*size)
	rest = np.logical_not(heldout) # we can use rest and heldout as indices to get elements satisfy condition.
	return arr[heldout], arr[rest]
	
def kfold(arr, K):
	"Do k-fold training data splitting."
	return [fold(arr, K, i) for i in range(K)]
def avg_accuracy(all_x, all_y, C):
	s = 0
	K = len(all_x)
	for i in range(K):
		x_heldout, x_rest = all_x[i]
		y_heldout, y_rest = all_y[i]
		w = train_w(x_rest, y_rest, C)
		s += accuracy(x_heldout, y_heldout, w)
	return s*1.0 / K
	
def train_C(X, Y, K = 10):
	"Using k-fold CV to optimize C"
	# C candidates
	all_C = np.arange(0,1,0.1)
	all_X = kfold(X,K)
	all_Y = kfold(Y,K)
	all_acc = np.array([avg_accuracy(all_X, all_Y, C) for C in all_C])
	return all_C[all_acc.argmax()]

def read_data(filename, sep=",", filt=int):
	def split_line(line):
		return line.split(sep)
		
	def apply_filt(values):
		"Filtering(mapping) all v in values and make it integer"
		return map(filt, values)
	def process_line(line):
		return apply_filt(split_line(line))
		
	f = open(filename)
	lines = map(process_line, f.readlines())
	# adding x[0] = 1 for all rows
	X = np.array([ [1]+line[1:] for line in lines ])
	# the first column is class label of training data
	Y = np.array([line[0] or -1 for line in lines])
	
	f.close()
	
	return X, Y

def main(training_file='SPECT.train.txt', test_file='SPECT.test.txt'):
	X_train, Y_train = read_data(training_file)
	C = train_C(X_train, Y_train)
	print "Optimized C was", C
	w = train_w(X_train, Y_train, C)
	print 'w was', w
	
	X_test, Y_test = read_data(test_file)
	print 'Accuracy was', accuracy(X_test, Y_test, w)
	
if __name__ == '__main__':
	main()
		