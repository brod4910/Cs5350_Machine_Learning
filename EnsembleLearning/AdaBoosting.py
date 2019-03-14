from utils.prepare_data import prepare_data
from utils.error_functions import entropy, gini_index
from DecisionTree.DecisionTree import DecisionTree
import numpy as np
from math import log, exp
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

def main():
    attr_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays','previous' , 'poutcome']
    attr_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    train_examples, train_labels, attributes = prepare_data('../DecisionTree/bank/train.csv', 
        attr_names= attr_names,
        attr_numeric= attr_numeric)

    train_labels = [1 if label == "yes" else -1 for label in train_labels]
    train_labels = np.array(train_labels)

    test_examples, test_labels, __ = prepare_data('../DecisionTree/bank/test.csv', 
        attr_names= attr_names,
        attr_numeric= attr_numeric)

    test_labels = [1 if label == "yes" else -1 for label in test_labels]

    train_errors = []
    test_errors = []

    T = 100
    weights = np.ones(len(train_examples))/len(train_examples)
    hypotheses = []
    hypotheses_votes = []
    for t in range(1, T + 1):
        for e in range(t, t+1):
            h = DecisionTree(entropy, 1)
            h.create_tree(train_examples, attributes, train_labels, weights= weights)

            predicted = np.array(h.test(train_examples))

            err = h.test_error(predicted, train_labels)

            alpha = .5 * log((1 - err)/err, 2)
            
            weights = weights * np.exp(-alpha * (predicted * train_labels))
            weights = weights/weights.sum()

            hypotheses.append(h)
            hypotheses_votes.append(alpha)

        H_final_train = np.zeros(len(train_examples))
        H_final_test = np.zeros(len(test_examples))

        for h, alpha in zip(hypotheses,hypotheses_votes):
            # print(alpha * np.array(h.test(test_examples)))
            H_final_train += alpha * np.array(h.test(train_examples))

        for h, alpha in zip(hypotheses,hypotheses_votes):
            # print(alpha * np.array(h.test(test_examples)))
            H_final_test += alpha * np.array(h.test(test_examples))

        H_final_train = np.sign(H_final_train)
        H_final_test = np.sign(H_final_test)

        train_err = h.test_error(H_final_train, train_labels)
        test_err = h.test_error(H_final_test, test_labels)

        train_errors.append(train_err)
        test_errors.append(test_err)

    plt.plot(train_errors)
    plt.plot(test_errors)
    plt.legend(['train error', 'test error'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()