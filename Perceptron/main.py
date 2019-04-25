from utils.utils import prepare_continous_data
from utils.utils import shuffle_data
from BasePerceptron import BasePerceptron
from VotedPerceptron import VotedPerceptron
from AveragedPerceptron import AveragedPerceptron

import numpy as np


def run_base_perceptron(train_ex, train_labels, test_ex, test_labels, T, lr):
	perceptron = BasePerceptron(.01)
	for t in range(T):
		shuffled_x, shuffled_y = shuffle_data(train_ex, train_labels)
		perceptron.train_dataset(shuffled_x, shuffled_y)

	predictions, err = perceptron.test_dataset(test_ex, test_labels)
	print("STANDARD PERCEPTRON:")
	print("error: ", err)
	print("weights: ")
	print(perceptron.weights)

def run_voted_perceptron(train_ex, train_labels, test_ex, test_labels, T, lrlr):
	v_perceptron = VotedPerceptron(.01)
	for t in range(T):
		shuffled_x, shuffled_y = shuffle_data(train_ex, train_labels)
		v_perceptron.train_dataset(shuffled_x, shuffled_y)

	predictions, err = v_perceptron.test_dataset(test_ex, test_labels)

	print("VOTED PERCEPTRON:")
	print("error: ", err)
	print("voted weights: ")
	for (c, w) in v_perceptron.m_weights:
		print("counts: ", c)
		print("weight vector: \n", w)
	print("number of votes: ", v_perceptron.m)

def run_averaged_perceptron(train_ex, train_labels, test_ex, test_labels, T, lr):
	a_perceptron = AveragedPerceptron(.01)
	for t in range(T):
		shuffled_x, shuffled_y = shuffle_data(train_ex, train_labels)
		a_perceptron.train_dataset(shuffled_x, shuffled_y)

	predictions, err = a_perceptron.test_dataset(test_ex, test_labels)

	print("AVERAGED PERCEPTRON:")
	print("error: ", err)
	print("weights: ")
	print(a_perceptron.weights)



def main():
	train_ex, train_labels = prepare_continous_data('./bank-note/train.csv')
	test_ex, test_labels = prepare_continous_data('./bank-note/test.csv')
	train_labels = np.array([[-1] if l == 0 else [l] for l in train_labels])
	test_labels = np.array([[-1] if l == 0 else [l] for l in test_labels])

	T = 10
	lr = .01

	run_base_perceptron(train_ex, train_labels, test_ex, test_labels, T, lr)
	run_voted_perceptron(train_ex, train_labels, test_ex, test_labels, T, lr)
	run_averaged_perceptron(train_ex, train_labels, test_ex, test_labels, T, lr)

if __name__ == '__main__':
	main()
