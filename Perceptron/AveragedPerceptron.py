from BasePerceptron import BasePerceptron
import numpy as np

class AveragedPerceptron(BasePerceptron):
	def __init__(self, lr, weights=None):
		super().__init__(lr, weights)
		self.a = None

	def train_dataset(self, X, Y):
		if self.weights is None:
			self.weights = np.zeros((X.shape[1], 1))
			self.a = np.zeros_like(self.weights)

		for (x,y) in zip(X,Y):
			x_p, y_p = x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)
			self._train(x_p,y_p)

	def _train(self, x, y):
		y_prime = self._predict(x)
		err = y - y_prime

		if y_prime != y:
			self.weights += self.lr * err * x
		
		self.a += self.weights

	def _test(self, x):
		y_prime = self._test_prediction(x)
		return y_prime

	def _test_prediction(self, x):
		y_prime = np.sign(np.dot(self.a.T, x))
		return y_prime
