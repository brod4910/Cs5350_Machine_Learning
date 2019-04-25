import numpy as np
from LogisticRegression import LogisticRegressionClassifier
from utils.utils import prepare_continous_data, shuffle_data

def main():
	train_ex, train_labels = prepare_continous_data('./bank-note/train.csv')
	test_ex, test_labels = prepare_continous_data('./bank-note/test.csv')
	train_labels = np.array([[-1] if l == 0 else [l] for l in train_labels])
	test_labels = np.array([[-1] if l == 0 else [l] for l in test_labels])

	epochs = 100
	lr = .0001
	variance = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
	dp = [100/873, 500/873, 700/873]


	for v in variance:
		for d in dp:
			print('Variance: {:.2f}, \t Parameter d: {:.6f}'.format(v, d))
			LogisticClassifier = LogisticRegressionClassifier(lr, v, d, mode= 'map')
			for epoch in range(1, epochs + 1):
				X, Y = shuffle_data(train_ex, train_labels)

				loss = LogisticClassifier.train_dataset(X, Y, epoch)
				# print('Epoch: {}\n\t Loss: {:.6f}'.format(epoch, loss[0]))

			preds, train_loss = LogisticClassifier.test_dataset(train_ex, train_labels)
			preds, test_loss = LogisticClassifier.test_dataset(test_ex, test_labels)
			print('Train error: {:.6f}'.format(train_loss))
			print('Test error: {:.6f}'.format(test_loss))
			# print(LogisticClassifier.weights)

if __name__ == '__main__':
	main()