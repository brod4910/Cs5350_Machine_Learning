from utils.prepare_data import prepare_data
from utils.error_functions import entropy
from AdaBoostedTree import AdaBoostedTree
from BaggedTree import BaggedTree
from RandomForest import RandomForest
import numpy as np

def run_adaboost(train_examples, train_labels, attributes, test_examples, test_labels, n_trees):
    adaboost = AdaBoostedTree(entropy, n_trees, 1)
    adaboost.train_dataset(train_examples, attributes, train_labels)

    preds, error = adaboost.test_dataset(test_examples, test_labels)

    return error

def run_baggedtree(train_examples, train_labels, attributes, test_examples, test_labels, n_trees):
    baggedtree = BaggedTree(entropy, n_trees, len(attributes))
    baggedtree.train_dataset(train_examples, attributes, train_labels)

    preds, error = baggedtree.test_dataset(test_examples, test_labels)
    
    return error

def run_randomforest(train_examples, train_labels, attributes, test_examples, test_labels, n_trees):
    rforest = RandomForest(entropy, 2, n_trees, len(attributes))
    rforest.train_dataset(train_examples, attributes, train_labels)

    preds, error = rforest.test_dataset(test_examples, test_labels)
    
    return error

def main():
    attr_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays','previous' , 'poutcome']
    attr_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    train_examples, train_labels, attributes = prepare_data('../DecisionTree/bank/train.csv', 
        attr_names= attr_names,
        attr_numeric= attr_numeric)

    train_labels = np.array([1 if label == "yes" else -1 for label in train_labels])

    test_examples, test_labels, __ = prepare_data('../DecisionTree/bank/test.csv', 
        attr_names= attr_names,
        attr_numeric= attr_numeric)

    test_labels = np.array([1 if label == "yes" else -1 for label in test_labels])

    n_trees = 10

    for n in range(1, n_trees + 1):
        print("Number of trees per experiment: {}".format(n))
        error = run_adaboost(train_examples, train_labels, attributes, test_examples, test_labels, n)
        print("Error of Adaboosted Tree on test set: {:.5f}".format(error))

        error = run_baggedtree(train_examples, train_labels, attributes, test_examples, test_labels, n)
        print("Error of Bagged Tree on test set: {:.5f}".format(error))

        error = run_randomforest(train_examples, train_labels, attributes, test_examples, test_labels, n)
        print("Error of Random Forest on test set: {:.5f}".format(error))

if __name__ == '__main__':
    main()