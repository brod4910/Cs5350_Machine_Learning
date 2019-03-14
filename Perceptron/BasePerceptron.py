import numpy as np

class BasePerceptron():
	def __init__(self, lr, weights= None):
		self.lr = lr
		self.weights = weights

	def train_dataset(self, X, Y):
		if self.weights is None:
			self.weights = np.zeros((X.shape[1], 1))

		for (x,y) in zip(X,Y):
			x_p, y_p = x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)
			self._train(x_p,y_p)

	def _train(self, x, y):
		y_prime = self._predict(x)
		err = y - y_prime

		if y_prime != y:
			self.weights += self.lr * (err * x)


	def _test(self, x):
		y_prime = self._predict(x)
		return y_prime

	def test_dataset(self, X, Y):
		corr = 0
		predictions = []
		for (x, y) in zip(X, Y):
			y_prime = self._test(x)
			predictions.append(predictions)

			if y_prime == y:
				corr += 1
		err = 1 - corr/len(Y)

		return predictions, err

	def _predict(self, x, weights= None):
		if weights is None:
			y_prime = np.sign(np.dot(self.weights.T, x))
		else:
			y_prime = np.sign(np.dot(weights.T, x))
		return y_prime
