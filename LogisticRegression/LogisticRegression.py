import numpy as np

class LogisticRegressionClassifier():
	def __init__(self, lr, variance, d, mode= 'mle'):
		self.lr = lr
		self.d = d
		self.variance = variance
		self.weights = None
		self.mode = mode

	def train_dataset(self, X, Y, epoch):
		if self.weights is None:
			self.weights = np.random.normal(scale= np.sqrt(self.variance), size= (X.shape[1]))
			# self.weights[-1] = 1

		if epoch == 1:
			lr = self.lr
		else:
			lr = self._lr_scheduler(self.lr, epoch, self.d)

		loss = 0
		for x, y in zip(X, Y):
			x = np.expand_dims(x, axis= 0)
			# calculate prior
			if self.mode != 'mle':
				prior = self.gaussian_prior(self.weights, self.variance)
				grad_prior = self.grad_gaus_prior(self.weights, self.variance)
			# calculate loss
			loss += self.loss(x, y) if self.mode == 'mle' else self.loss(x, y) + prior
			# calculate gradients
			gradients = self.grad_loss(x, y) if self.mode == 'mle' else self.grad_loss(x, y) + grad_prior
			# print(self.grad_loss(x, y), grad_prior)
			gradients = np.squeeze(gradients)
			# update
			self.weights -= lr*gradients

		loss /= len(Y)
		return loss

	def test_dataset(self, X, Y):
		corr = 0
		predictions = []
		for x, y in zip(X, Y):
			x = np.expand_dims(x, axis= 0)
			y_prime = self.predict(x)
			predictions.append(predictions)

			if y_prime == y:
				corr += 1
		err = 1 - corr/len(Y)

		return predictions, err
	
	def _lr_scheduler(self, lr, epoch, d):
		return lr/(1+((lr/d)*epoch))

	def loss(self, x, y):
		return np.log(1 + np.exp(-y * (x.dot(self.weights))))

	def grad_loss(self, x, y):
		return (self.lr*y*x*np.log((1+np.exp(-y * x.dot(self.weights)))))

	def predict(self, x):
		return np.sign(x.dot(self.weights))

	def gaussian_prior(self, w, v):
		denom = 2*v
		return (1/np.sqrt(denom*np.pi))*np.exp(-(1/denom)*w.T.dot(w))

	def grad_gaus_prior(self, w, v):
		denom = 2*v
		return np.exp(-(1/denom)*w.T.dot(w))/(np.sqrt(2)*np.sqrt(np.pi*v))
