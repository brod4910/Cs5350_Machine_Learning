from BasePerceptron import BasePerceptron
import numpy as np

class VotedPerceptron(BasePerceptron):
	def __init__(self, lr, weights=None):
		super().__init__(lr, weights)
		self.m = 0
		self.m_weights = []
		self.C_m = 1

	def _train(self, x, y):
		y_prime = self._predict(x)
		err = y - y_prime

		if y_prime != y:
			self.m_weights.append((self.C_m, self.weights.copy()))
			self.weights += self.lr * err * x
			self.m += 1
			self.C_m = 1
		else:
			self.C_m += 1

	def _test(self, x):
		y_prime = self._test_prediction(x)
		return y_prime

	def _test_prediction(self, x):
		votes = [c * self._predict(x, w) for (c, w) in self.m_weights]
		y_prime = np.sign(np.sum(votes))
		return y_prime
