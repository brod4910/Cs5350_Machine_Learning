from DecisionTree import DecisionTree, Node, entropy, gini_index, majority_error
import numpy as np
from collections import Counter

def main():
    train_examples = []
    train_labels = []
    label_names = ['yes', 'no']
    attributes = {}
    attribute_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays','previous' , 'poutcome']
    attribute_names_numeric = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    attribute_names_unknown = ['job', 'education', 'contact', 'poutcome']

    with open ('./bank/train.csv' ,'r') as file:
        for line in file:
            s = {}
            sample = line.strip().split(',')
            for i, item in enumerate(sample[:-1]):
                s[attribute_names[i]] = item
            train_examples.append(s)
            train_labels.append(sample[-1])

    medians = [[],[],[],[],[],[], []]
    for s in train_examples:
        for i, val in enumerate(attribute_names_numeric):
            num = float(s[val])
            medians[i].append(num)

    for i, median in enumerate(medians):
        medians[i] = np.median(median)

    for (attr, median) in zip(attribute_names_numeric, medians):
        for s in train_examples:
            # print(attr)
            s[attr] = 'bigger' if float(s[attr]) >= float(median) else 'less'

    unknowns = [[],[],[],[]]
    for s in train_examples:
        for i, unknown in enumerate(attribute_names_unknown):
            unknowns[i].append(s[unknown])
    
    unknowns = [Counter(unknown).most_common(1)[0][0] for unknown in unknowns]

    for ex in train_examples:
        for j, item in enumerate(ex):
            attrs = ex[item]
            if item not in attributes:
                attributes[item] = []
            if attrs not in attributes[item]:
                attributes[item].append(attrs)

    tree = DecisionTree(gini_index, 6)
    tree.create_tree(train_examples, attributes, train_labels)

    test_examples = []
    test_labels = []
    with open ('./bank/test.csv' ,'r') as file:
        for line in file:
            s = {}
            sample = line.strip().split(',')
            for i, item in enumerate(sample[:-1]):
                s[attribute_names[i]] = item
            test_examples.append(s)
            test_labels.append(sample[-1])

    for (attr, median) in zip(attribute_names_numeric, medians):
        for s in test_examples:
            s[attr] = 'bigger' if float(s[attr]) >= float(median) else 'less'

    error_fns = [gini_index, majority_error, entropy]
    names = ["gini_index", "majority_error", "entropy"]

    print("BANK DATA EXPERIMENTS")
    print('WITH_ UNKNOWNS INCLUDED')
    print('Tree depth\t|\tGini Index Error\t|\tMajority Error\t|\tEntropy Error\t|')
    test_running_averages = [0, 0, 0]
    train_running_averages = [0, 0, 0]
    for i in range(1, 17):
        test_errors = []
        train_errors = []
        for j, (err_fn, name) in enumerate(zip(error_fns, names)):
            tree = DecisionTree(err_fn, i)
            tree.create_tree(train_examples, attributes, train_labels)
            
            predicted = tree.test(train_examples)
            train_error = tree.test_error(predicted, train_labels)
            train_errors.append(train_error)
            train_running_averages[j] += train_error

            predicted = tree.test(test_examples)
            test_error = tree.test_error(predicted, test_labels)
            test_errors.append(test_error)
            test_running_averages[j] += test_error


        print('{}\t\t|\t\t{:.4f}\t\t|\t{:.4f}\t\t|\t{:.4f}\t\t|'.format(i, test_errors[0], test_errors[1], test_errors[2]))

    train_running_averages = [err / 16 for err in train_running_averages]
    test_running_averages = [err / 16 for err in test_running_averages]
    print("Running Averages of train error:")
    print("Gini Index: {:.4f}, Majority Error: {:.4f}, Entropy: {:.4f}".format(train_running_averages[0], train_running_averages[1], train_running_averages[2]))
    print("Running Averages of test error:")
    print("Gini Index: {:.4f}, Majority Error: {:.4f}, Entropy: {:.4f}\n".format(test_running_averages[0], test_running_averages[1], test_running_averages[2]))

    for (attr, unknown) in zip(attribute_names_unknown, unknowns):
        for s in train_examples:
            s[attr] = unknown

    for (attr, unknown) in zip(attribute_names_unknown, unknowns):
        for s in test_examples:
            s[attr] = unknown

    print('WITHOUT UNKNOWNS INCLUDED')
    print('Tree depth\t|\tGini Index Error\t|\tMajority Error\t|\tEntropy Error\t|')
    test_running_averages = [0, 0, 0]
    train_running_averages = [0, 0, 0]
    for i in range(1, 17):
        test_errors = []
        train_errors = []
        for j, (err_fn, name) in enumerate(zip(error_fns, names)):
            tree = DecisionTree(err_fn, i)
            tree.create_tree(train_examples, attributes, train_labels)
            
            predicted = tree.test(train_examples)
            train_error = tree.test_error(predicted, train_labels)
            train_errors.append(train_error)
            train_running_averages[j] += train_error

            predicted = tree.test(test_examples)
            test_error = tree.test_error(predicted, test_labels)
            test_errors.append(test_error)
            test_running_averages[j] += test_error


        print('{}\t\t|\t\t{:.4f}\t\t|\t{:.4f}\t\t|\t{:.4f}\t\t|'.format(i, test_errors[0], test_errors[1], test_errors[2]))

    train_running_averages = [err / 16 for err in train_running_averages]
    test_running_averages = [err / 16 for err in test_running_averages]
    print("Running Averages of train error:")
    print("Gini Index: {:.4f}, Majority Error: {:.4f}, Entropy: {:.4f}".format(train_running_averages[0], train_running_averages[1], train_running_averages[2]))
    print("Running Averages of test error:")
    print("Gini Index: {:.4f}, Majority Error: {:.4f}, Entropy: {:.4f}".format(test_running_averages[0], test_running_averages[1], test_running_averages[2]))



if __name__ == '__main__':
    main()