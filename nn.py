'''
nn.py
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def shuffle_together(a, b):
	'''
	shuffle two arrays in unison
	helpful code from
	https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
	'''
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]

def add_bias(x):
	ones = np.ones((len(x),1))
	x = np.append(x, ones, 1)
	return x

def normalize(x):
	norms = np.broadcast_to(np.reshape(np.sum(x, 1), (len(x), 1)), x.shape)
	return x/norms

def get_training_data(size = 1, holdout = 0.1, xname = "TrainDigitX.csv.gz", yname = "TrainDigitY.csv.gz",
	xt1name = "TestDigitX.csv.gz", yt1name = "TestDigitY.csv.gz", xt2name = "TestDigitX2.csv.gz"):
	
	print "Loading training data..."
	# load
	xdata = np.loadtxt(xname, delimiter = ",")
	ydata = np.loadtxt(yname)
	xtest1 = np.loadtxt(xt1name, delimiter = ",")
	ytest1 = np.loadtxt(yt1name)
	xtest2 = np.loadtxt(xt2name, delimiter = ",")
	print "Creating training and holdout sets..."
	# shuffle
	(xdata, ydata) = shuffle_together(xdata, ydata)
	# add bias neuron
	xdata  = add_bias(xdata)
	xtest1 = add_bias(xtest1)
	xtest2 = add_bias(xtest2)
	print xdata.shape
	# normalize
	xdata  = normalize(xdata)
	xtest1 = normalize(xtest1)
	xtest2 = normalize(xtest2)
	# get sizes
	assert size <= 1
	assert size > 0
	size = int(len(ydata)*size)
	hsize = int(size*holdout)
	print size
	print hsize
	# update vectors
	xhold = xdata[0:hsize]
	yhold = ydata[0:hsize]
	xdata = xdata[hsize:size]
	ydata = ydata[hsize:size]
	
	print ""
	return xdata, ydata, xhold, yhold, xtest1, ytest1, xtest2



def sig(x):
	'''
	sigmoid function
	'''
	return 1.0/(1.0 + np.exp(-x))

def dsig(x):
	'''
	derivative of sigmoid function
	'''
	return x*(1-x)

### network

def init(hidden_sizes = [28], in_size = 785, out_size = 10):
	'''
	creates initial weight vectors
	'''
	layers = [in_size] + hidden_sizes + [out_size]
	weights = []
	for i in xrange(1,len(layers)):
		weights.append(np.random.rand(layers[i-1], layers[i])*2 - 1)
	return weights


def think(input_vec, weights):
	'''
	gives an output based on input
	'''
	results = [input_vec]
	for w in weights:
		y = np.dot(results[-1], w)
		results.append(sig(y))
	return results

def get_digit(results):
	'''
	gives the best guess for which digit based on results
	'''
	return np.argmax(results[-1], axis = 1)


def make_answers(y, out_size = 10):
	'''
	converts vector of digits into array of
	[0,...,1,...,0] vectors
	'''
	answers = np.zeros((len(y), out_size))
	for i in xrange(0,len(y)):
		answers[i,int(y[i])] = 1
	return answers


def get_accuracy(input_vec, answer_vec, weights):
	results = think(input_vec, weights)
	guess = get_digit(results)
	correct = np.equal(guess, answer_vec)
	return float(sum(correct))/len(correct)


def train(input_vec, answer_vec, xhold, yhold, weights, learn_rate, max_epochs):
	'''
	given starting weights
	'''
	n = len(input_vec)
	accuracy = []
	print "Training over " + str(n) + " examples"
	for e in xrange(0,max_epochs):
		oldweights = weights
		for i in xrange(0,n):
			results = think(input_vec[i], weights)
			nsteps = len(weights)
			# last layer
			output = results[nsteps] # matrix of 10xb
			delta = output - answer_vec[i] # vector of 10xb
			dlast = delta*dsig(output) # vector of 10xb
			weights[nsteps-1] = weights[nsteps-1] - learn_rate*np.outer(results[nsteps-1], dlast) # matrix of 28x10
			# other layers
			for l in xrange(nsteps-1, 0, -1):
				temp = np.sum(weights[l]*np.broadcast_to(dlast, weights[l].shape), 1)
				dlast = temp*dsig(results[l])
				weights[l-1] = weights[l-1] - learn_rate*np.outer(results[l-1], dlast)
		accuracy.append(get_accuracy(xhold, yhold, weights))
		print "\tepoch " + str(e) + " finished with holdout accuracy " + str(accuracy[-1])
		if e == 0:
			oldweights = weights
		elif accuracy[e] < 0.95*accuracy[e-1]:
			weights = oldweights
			print "\tstopping training due to low accuracy"
			break
		else:
			oldweights = weights
	return weights, accuracy


### plotting and outputting

def export(guess, filename):
	predictions = open(filename, "w")
	l = len(guess)
	for g in xrange(0,l-1):
		predictions.write(str(guess[g]) + ", ")
	predictions.write(str(guess[l-1]) + "\n")
	predictions.close()


def learnplot(xdata, ydata, xhold, yhold, xtest, ytest, size, rates, max_epochs, save_weights = False, log10 = False):
	'''
	runs and plots results of accuracy over different learning rates
	'''
	answers = make_answers(ydata)
	x = []
	y = []
	for l in rates:
		t1 = time.time()
		print "Learning rate " + str(l)
		x.append(l)
		w = init([size])
		w, accuracy = train(xdata, answers, xhold, yhold, w, l, max_epochs)
		t2 = time.time()
		print t2-t1
		y.append(get_accuracy(xtest1, ytest1, w))
		print y[-1]
		print accuracy[-1]
		if save_weights:
			np.save("weights_l"+str(l), w)
	plt.plot(x, y)
	plt.grid(True)
	plt.title("Accuracy over learning rate (hidden layer of " + str(size) + ")")
	if log10:
		x = np.log10(x)
		plt.xlabel("Learning rate (log 10)")
	else:
		plt.xlabel("Learning rate")
	plt.ylabel("Test set accuracy")
	plt.show()
	return x, y

def layerplot(xdata, ydata, xhold, yhold, xtest, ytest, sizes, learn_rate, max_epochs, save_weights = False):
	'''
	runs and plots results of accuracy over different layer sizes
	'''
	answers = make_answers(ydata)
	x = []
	y = []
	for s in sizes:
		t1 = time.time()
		print "Hidden layer size " + str(s)
		x.append(s)
		w = init([s])
		w, accuracy = train(xdata, answers, xhold, yhold, w, learn_rate, max_epochs)
		t2 = time.time()
		y.append(get_accuracy(xtest, ytest, w))
		print "Accuracy: " +str(y[-1])
		print t2-t1
		if save_weights:
			np.save("weights_s"+str(s), w)
	plt.plot(x, y)
	plt.grid(True)
	plt.title("Accuracy over layer size (learning rate " + str(learn_rate) + ")")
	plt.xlabel("Hidden layer size")
	plt.ylabel("Test set accuracy")
	plt.show()
	return x, y

def surfaceplot(xdata, ydata, xhold, yhold, answers):
	'''
	runs and plots results of accuracy based on both layer size and learning rate
	don't use this it's very slow
	'''
	L = []
	S = []
	A = []
	l = 0.25
	while l < 2.5:
		print "Learning rate " + str(l)
		A.append([])
		L.append(l)
		S = []
		for s in xrange(20,100,5):
			print "Hidden layer size " + str(s)
			S.append(s)
			w = init([s])
			w, accuracy = train(xdata, answers, xhold, yhold, w, l, 50)
			A[-1].append(np.amax(accuracy))
		l += 0.25
	Z = np.asarray(A)
	X, Y = np.meshgrid(S, L)
	print X.shape
	print Y.shape
	print Z.shape
	
	hf = plt.figure()
	ha = hf.add_subplot(111, projection='3d')
	ha.plot_surface(X, Y, Z)
	plt.show()


if __name__ == "__main__":
	t1 = time.time()
	xdata, ydata, xhold, yhold, xtest1, ytest1, xtest2 = get_training_data(1)
	t2 = time.time()
	print t2-t1
	answers = make_answers(ydata)
	test_answers = make_answers(ytest1)
	w = np.load("good.npy")
	w, accuracy = train(xdata, answers, xhold, yhold, w, 0.01, 100)
	np.save("good2", w)
	x = np.arange(len(accuracy))
	plt.plot(x, accuracy)
	plt.grid(True)
	plt.title("Accuracy over number of epochs (hidden layer size 32, learning rate 0.25)")
	plt.xlabel("Epoch")
	plt.ylabel("Holdout set accuracy")
	r1 = think(xtest1, w)
	r2 = think(xtest2, w)
	export(get_digit(r1), "test1predictions2.csv")
	export(get_digit(r2), "test2predictions2.csv")
	plt.show()


