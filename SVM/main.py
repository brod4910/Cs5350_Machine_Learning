from PrimalSVM import PrimalSVM
from DualSVM import DualSVM
from utils.utils import prepare_continous_data
from utils.utils import shuffle_data
import numpy as np

def run_dual_svm(train_ex, train_labels, test_ex, test_labels, epochs, lr, C, kernel= None):
	for c in C:
		print("Training Dual SVM with C = {:.8f}".format(c))
		dual_svm = DualSVM(c, lr, kernel=kernel)

		x_s, y_s = shuffle_data(train_ex, train_labels)
		y_s = y_s.reshape((y_s.shape[0]))
		dual_svm.train_dataset(x_s, y_s)
		# print(primal_svm.weights)
		__, train_err = dual_svm.test_dataset(train_ex, train_labels)
		__, test_err = dual_svm.test_dataset(test_ex, test_labels)

		print("Train Error: ", train_err)
		print("Test Error: ", test_err)

def run_primal_svm(train_ex, train_labels, test_ex, test_labels, epochs, lr, C):
	for c in C:
		print("Training Primal SVM with C = {:.8f}".format(c))
		primal_svm = PrimalSVM(c, lr)

		for epoch in range(epochs):
			x_s, y_s = shuffle_data(train_ex, train_labels)
			primal_svm.train_dataset(x_s, y_s)
			# print(primal_svm.weights)
			__, train_err = primal_svm.test_dataset(train_ex, train_labels)
			__, test_err = primal_svm.test_dataset(test_ex, test_labels)
			primal_svm.lr = lr/(1+(lr/100)*epoch)

		print("Train Error: ", train_err)
		print("Test Error: ", test_err)
		print('Weights:')
		for i in range(len(primal_svm.weights)):
			print(primal_svm.weights[i][0])


def main():
	train_ex, train_labels = prepare_continous_data('./bank-note/train.csv')
	test_ex, test_labels = prepare_continous_data('./bank-note/test.csv')
	train_labels = np.array([[-1] if l == 0 else [l] for l in train_labels])
	test_labels = np.array([[-1] if l == 0 else [l] for l in test_labels])

	epochs = 100
	lr = .1
	C = [1/873, 10/873, 50/873, 100/873, 300/873, 500/873, 700/873]

	run_primal_svm(train_ex, train_labels, test_ex, test_labels, epochs, lr, C)

	C = [100/873, 500/873, 700/873]

	run_primal_svm(train_ex, train_labels, test_ex, test_labels, epochs, lr, C)
	run_dual_svm(train_ex, train_labels, test_ex, test_labels, epochs, lr, C)

	print("Running Dual SVM gaussian kernel:")

	run_dual_svm(train_ex, train_labels, test_ex, test_labels, epochs, lr, C, kernel='gaussian')

if __name__ == '__main__':
	main()