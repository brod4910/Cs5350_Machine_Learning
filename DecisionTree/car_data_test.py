from DecisionTree import DecisionTree, Node, entropy, gini_index, majority_error
import numpy as np

def main():
    train_examples = []
    train_labels = []
    label_names = ['unacc', 'acc', 'good', 'vgood']
    attributes = {'buying' : ['vhigh', 'high', 'med', 'low'],
    'maint' : ['vhigh', 'high', 'med', 'low'],
    'doors' : ['2', '3', '4', '5more'],
    'persons' : ['2', '4', 'more'],
    'lug_boot' : ['small', 'med', 'big'],
    'safety' : ['low', 'med', 'high']}
    attribute_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

    with open ('./car/train.csv' ,'r') as file:
        for line in file:
            s = {}
            sample = line.strip().split(',')
            for j, item in enumerate(sample[:-1]):
                s[attribute_names[j]] = item

            train_examples.append(s)
            train_labels.append(sample[-1])


    tree = DecisionTree(gini_index, 6)
    tree.create_tree(train_examples, attributes, train_labels)

    test_examples = []
    test_labels = []
    with open ('./car/test.csv' ,'r') as file:
        for line in file:
            s = {}
            sample = line.strip().split(',')
            for j, item in enumerate(sample[:-1]):
                s[attribute_names[j]] = item

            test_examples.append(s)
            test_labels.append(sample[-1])

    error_fns = [gini_index, majority_error, entropy]
    names = ["gini_index", "majority_error", "entropy"]
    print("CAR DATA EXPERIMENTS")
    print('Tree depth\t|\tGini Index Error\t|\tMajority Error\t|\tEntropy Error\t|')
    test_running_averages = [0, 0, 0]
    train_running_averages = [0, 0, 0]
    for i in range(1, 7):
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

    train_running_averages = [err / 6 for err in train_running_averages]
    test_running_averages = [err / 6 for err in test_running_averages]
    print("Running Averages of train error:")
    print("Gini Index: {:.4f}, Majority Error: {:.4f}, Entropy: {:.4f}".format(train_running_averages[0], train_running_averages[1], train_running_averages[2]))
    print("Running Averages of test error:")
    print("Gini Index: {:.4f}, Majority Error: {:.4f}, Entropy: {:.4f}\n".format(test_running_averages[0], test_running_averages[1], test_running_averages[2]))


    

if __name__ == '__main__':
    main()