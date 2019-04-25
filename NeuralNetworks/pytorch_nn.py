import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
	def __init__(self, in_features, out_features, hidden_nodes, hidden_layers, activation, weight_init):
		super(NeuralNetwork, self).__init__()
		layers = []
		self.weight_init = weight_init
		for i in range(hidden_layers):
			if i == 0:
				layers += [nn.Linear(in_features, hidden_nodes)]
			elif i == hidden_layers - 1:
				layers += [nn.Linear(hidden_nodes, out_features)]
			else:
				layers += [nn.Linear(hidden_nodes, hidden_nodes)]

			if activation == 'relu':
				layers += [nn.ReLU(inplace=True)]
			else:
				layers += [nn.Tanh()]

		self.model = nn.Sequential(*layers)
		self.model.apply(self.init_weights)

	def forward(self, x):
		out = self.model(x)
		return out

	def init_weights(self, l):
		if type(l) == nn.Linear:
			if self.weight_init == 'he':
				nn.init.kaiming_uniform_(l.weight, mode='fan_in', nonlinearity='relu')
			else:
				nn.init.xavier_uniform_(l.weight)