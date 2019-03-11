import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.prepare_data import prepare_data
from utils.error_functions import entropy, gini_index
from DecisionTree.DecisionTree import DecisionTree
import numpy as np
from math import log, exp
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

    hypotheses = []
    hypotheses_votes = []
    T = 100
    for t in range(T):
        for e in range(t,t+1):
            m_ex = []
            m_labels = []

            for i in range(len(train_examples)):
                n = np.random.randint(len(train_examples))
                m_ex.append(train_examples[n])
                m_labels.append(train_labels[n])

            h = DecisionTree(entropy, len(attributes))
            h.create_tree(m_ex, attributes, m_labels, weights= None)

            predicted = np.array(h.test(m_ex))

            err = h.test_error(predicted, m_labels)

            alpha = .5 * log((1 - err)/err, 2)

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

    # hypotheses = []
    # hypotheses_votes = []

    # for t in range(100):
    #     m_ex = []
    #     m_labels = []
    #     ht = []
    #     hv = []
    #     sample = [i for i in range(len(train_examples))]
    #     sample = np.random.choice(sample, 1000, replace= False)

    #     for i in range(10):
    #         for i in range(len(sample)):
    #             n = np.random.randint(len(sample))
    #             m_ex.append(train_examples[n])
    #             m_labels.append(train_labels[n])

    #         h = DecisionTree(entropy, len(attributes))
    #         h.create_tree(m_ex, attributes, m_labels, weights= None)

    #         predicted = np.array(h.test(m_ex))

    #         err = h.test_error(predicted, m_labels)
    #         if err == 0:
    #             err = .5
    #         alpha = .5 * log((1 - err)/err, 2)

    #         ht.append(h)
    #         hv.append(alpha)

    #     hypotheses.append(ht)
    #     hypotheses_votes.append(hv)

    # th = []
    # tv = []
    # preds = []
    # h_final = []

    # for t in range(100):
    #     th.append(hypotheses[t][0])
    #     tv.append(hypotheses_votes[t][0])

    # for h, v in zip(th, tv):
    #     hf = 0
    #     m = 0
    #     for i, (t, l) in enumerate(zip(test_examples, test_labels)):
    #         hf += v * np.array(h.test([t]))
    #         hf = hf / len(th)
    #         hf = hf - l
    #         hf = np.square(hf)
    #     h_final.append(hf)

    # plt.plot(h_final)
    # plt.legend(['Bias'], loc='upper left')
    # plt.show()

if __name__ == '__main__':
    main()
