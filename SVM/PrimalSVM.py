import numpy as np

class PrimalSVM():
	def __init__(self, C, lr):
		self.C = C
		self.lr = lr
		self.weights = None

	def train_dataset(self, X, Y):
		self.num_examples = len(Y)
		if self.weights is None:
			self.weights = np.zeros((X.shape[1], 1))
			self.weights[-1] = 1

		loss = 0

		for (x, y) in zip(X, Y):
			x_p, y_p = x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)
			loss += self._train(x_p,y_p)

		return loss/self.num_examples

	def test_dataset(self, X, Y):
		corr = 0
		predictions = []
		for (x, y) in zip(X, Y):
			x_p, y_p = x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)
			y_prime = self._test_predict(x)
			predictions.append(y_prime)

			if y_prime == y:
				corr += 1
		err = 1 - corr/len(Y)

		return predictions, err

	def _train(self, x, y):
		loss = self._svm_loss(x, y)
		pred = self._predict(x, y)

		if pred <= 1:
			gradient = np.copy(self.weights)
			gradient[-1] = 0
			self.weights = ((1 - self.lr) * gradient) + (self.lr * self.C * (self.num_examples * y * x))
		else:
			self.weights = (1 - self.lr)* self.weights

		return loss
		
	def _svm_loss(self, x, y):
		pred = self._predict(x, y)
		maxx = np.maximum(0, 1 - np.asscalar(pred))
		J = self.weights.T.dot(self.weights)/2
		J = J + (self.C * self.num_examples * maxx)
		return J

	def _predict(self, x, y,):
		y_prime = y * np.dot(self.weights.T, x)
		return y_prime

	def _test_predict(self, x):
		y_prime = np.sign(np.dot(self.weights.T, x))
		return y_prime

