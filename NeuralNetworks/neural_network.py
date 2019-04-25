import numpy as np

class NeuralNetwork():
	def __init__(self, num_features, nodes, lr, d):
		self.network = []
		self.lr = lr
		self.d = d
		# add bias
		hidden_nodes = nodes + 1

		input_layer = {'weights' : np.random.normal(size=(num_features, hidden_nodes))}
		hidden_layers = [{'weights' : np.random.normal(size=(hidden_nodes, hidden_nodes))}, {'weights' : np.random.normal(size=(hidden_nodes, hidden_nodes))}]
		output_layer = {'weights' : np.random.normal(size=(hidden_nodes, 1))}

		self.network.append(input_layer)
		self.network.append(hidden_layers[0])
		self.network.append(hidden_layers[1])
		self.network.append(output_layer)

	
	def train_dataset(self, X, Y, epoch):
		loss = 0
		if epoch == 1:
			lr = self.lr
		else:
			lr = self._lr_scheduler(self.lr, epoch, self.d)
		for x, y in zip(X, Y):
			x = np.expand_dims(x, axis= 0)
			output = self._forward_pass(x, y)
			loss += self.square_loss(output, y)
			grad_weights = self._backward_pass(output, y)
			self._update_weights(grad_weights, lr)

		loss /= len(Y)
		return loss

	def test_dataset(self, X, Y):
		loss = 0
		corr = 0
		for x, y in zip(X, Y):
			x = np.expand_dims(x, axis= 0)
			y_pred = self._predict(x)
			y_pred = np.sign(y_pred)

			if y_pred == y:
				corr += 1

		err = 1 - corr/len(Y)
		return err

	def _predict(self, x):
		out = x
		for i, layer in enumerate(self.network):
			if i != len(self.network) - 1:
				out = self.sigmoid(out.dot(layer['weights']))
			else:
				out = out.dot(layer['weights'])
		return out

	def _forward_pass(self, x, y):
		out = x
		for layer in self.network:
			layer['pre_actv'] = out.dot(layer['weights'])
			out = self.sigmoid(layer['pre_actv'])
			layer['output'] = out
		return out

	def _backward_pass(self, output, y):
		layer = self.network[-1]
		layer['delta'] = self.grad_sq_loss(output, y) * self.derv_sigmoid(layer['pre_actv'])
		for i, layer in enumerate(reversed(self.network)):
			# print(i)
			if i == 0:
				prev_layer = layer
				continue
			else:
				layer['delta'] = prev_layer['weights'].dot(prev_layer['delta'])*self.derv_sigmoid(layer['pre_actv'])


		grad_weights = [layer['weights'].dot(layer['delta'].T) for layer in self.network]
		return grad_weights

	def _update_weights(self, grad_weights, lr):
		for layer, grad_w in zip(self.network, grad_weights):
			layer['weights'] += lr*grad_w

	def _lr_scheduler(self, lr, epoch, d):
		return lr/(1+((lr/d)*epoch))

	def square_loss(self, y_pred, y):
		return (y - y_pred)**2/2

	def grad_sq_loss(self, y_pred, y):
		return y-y_pred

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def derv_sigmoid(self, x):
		dsig = self.sigmoid(x)
		return dsig*(1-dsig)