from collections import Counter
import numpy as np
from math import log

'''
    Calculates the gini index of the labels given
    params:
        labels: labels to calculate the gini index
'''
def gini_index(labels):
    n = len(labels)
    if isinstance(labels[0], tuple):
        counter = Counter([label[0] for label in labels])
        weighted_sums = [sum([label[1] for i, label in enumerate(labels) if label[0] == count]) for count in counter]
        return 1 - sum(weight*(counter[count]/n)**2 for count, weight in zip(counter, weighted_sums))
    else:
        counter = Counter(labels)
        return 1 - sum((counter[count]/n)**2 for count in counter)


    # probabilities = [counter[count]/n for count in counter]
    # for i, prob in enumerate(probabilities):
    #     err = prob**2
    #     gi -= err

'''
    Calculates the majority error of the labels given
    params:
        labels: labels to calculate the majority error
'''
def majority_error(labels):
    counter = Counter(labels)
    majority = counter.most_common(1)[0][1]
    err = 1 - (majority/len(labels))
    return err

'''
    Calculates the entropy of the labels given
    params:
        labels: labels to calculate the entropy
'''
def entropy(labels):
    n = len(labels)

    if isinstance(labels[0], tuple):
        counter = Counter([label[0] for label in labels])
        weighted_sums = [sum([label[1] for i, label in enumerate(labels) if label[0] == count]) for count in counter]
        return -sum(weight*(counter[count]/n) * log(weight*(counter[count]/n), 2) for count, weight in zip(counter, weighted_sums))
    else:
        counter = Counter(labels)
        return -sum(counter[count]/n * log(counter[count]/n, 2) for count in counter)

def least_mean_squares(preds, y_i):
    error = y_i - preds
    return (np.sum(error**2)) / (2*y_i.shape[0])

def derv_LMS(pred, y_i, x_i):
    error = y_i - pred
    error = error.reshape(1, error.shape[0])
    return -np.dot(error, x_i)
    # return -np.sum((y_i - pred).dot(x_i_j))/len(pred)



